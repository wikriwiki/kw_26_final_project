"""LoRA training for one-forward-pass restricted-choice EXAONE scoring.

--dry-run checks the dataset without importing torch or downloading weights.
Real training requires an explicitly selected CUDA device and optional ML deps.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .dataset import canonical_json, fingerprint, read_jsonl


def adapter_fingerprint(path: str | Path) -> str:
    """Bind calibration to actual saved adapter weights and configuration."""
    directory = Path(path)
    files = [directory / "adapter_config.json"]
    weights = [directory / name for name in ("adapter_model.safetensors", "adapter_model.bin") if (directory / name).is_file()]
    if not files[0].is_file() or len(weights) != 1:
        raise ValueError("expected one adapter weight file and adapter_config.json")
    digest = hashlib.sha256()
    for file in files + weights:
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_examples(rows: Iterable[dict], *, allow_synthetic: bool = False) -> dict:
    kinds: Counter = Counter()
    models: set[str] = set()
    datasets: set[str] = set()
    synthetic_count = 0
    groups: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    count = 0
    for index, row in enumerate(rows, 1):
        count += 1
        prefix = f"example {index}"
        if row.get("schema_version") != 1 or not row.get("group"):
            raise ValueError(f"{prefix}: missing schema version or actor group")
        provenance = row.get("provenance", {})
        if not isinstance(provenance, dict) or not isinstance(provenance.get("synthetic"), bool):
            raise ValueError(f"{prefix}: missing explicit synthetic provenance")
        synthetic = provenance["synthetic"]
        if synthetic and not allow_synthetic:
            raise ValueError(f"{prefix}: synthetic smoke data requires --allow-synthetic")
        if row.get("split") not in ({"train", "synthetic"} if allow_synthetic else {"train"}):
            raise ValueError(f"{prefix}: training may not consume calibration/test data")
        if synthetic and row.get("split") != "synthetic":
            raise ValueError(f"{prefix}: synthetic data must stay in its separate split")
        if not synthetic and row.get("split") != "train":
            raise ValueError(f"{prefix}: real data must stay in the training split")
        teacher_model = provenance.get("teacher_model_id", "")
        if not isinstance(teacher_model, str) or "exaone" not in teacher_model.lower():
            raise ValueError(f"{prefix}: only identified EXAONE teacher labels are supported")
        question = row.get("question", {})
        if not isinstance(question, dict):
            raise ValueError(f"{prefix}: incomplete choice question")
        options = question.get("options")
        if not isinstance(options, dict) or not 2 <= len(options) <= 52:
            raise ValueError(f"{prefix}: expected between 2 and 52 choice options")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in options.items()):
            raise ValueError(f"{prefix}: options must map strings to strings")
        if row.get("target") not in options:
            raise ValueError(f"{prefix}: target is not an allowed choice")
        if any(not isinstance(question.get(key), str) or not question[key] for key in ("key", "state", "instructions")):
            raise ValueError(f"{prefix}: incomplete choice question")
        if not row.get("dataset_fingerprint") or not provenance.get("source_fingerprint"):
            raise ValueError(f"{prefix}: missing dataset/source fingerprint")
        identity = (row.get("snapshot_id"), question["key"], row.get("kind"))
        if not all(isinstance(value, str) and value for value in identity) or identity in seen:
            raise ValueError(f"{prefix}: missing or duplicate training question identity")
        seen.add(identity)
        kinds[str(row.get("kind", "unknown"))] += 1
        models.add(teacher_model)
        datasets.add(row["dataset_fingerprint"])
        groups.add(str(row["group"]))
        synthetic_count += int(synthetic)
    if not count:
        raise ValueError("training dataset is empty")
    if len(datasets) != 1:
        raise ValueError("mixing dataset versions requires rebuilding a single split manifest")
    return {"examples": count, "groups": sorted(groups), "kinds": dict(kinds),
            "teacher_models": sorted(models), "dataset_fingerprint": next(iter(datasets)),
            "synthetic_examples": synthetic_count, "eligible_for_live": False}


def _restricted_logits(model: Any, input_ids: Any, attention_mask: Any, token_ids: list[int]) -> Any:
    """Read only final hidden-state label rows; never materialize [seq, vocab].

PEFT installs adapters in the underlying causal model. Calling its backbone
directly preserves those adapters and gradients, while avoiding unused LM logits.
"""
    import torch
    import torch.nn.functional as functional

    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = causal.base_model
    if backbone is causal:
        raise ValueError("model must expose a backbone separately from its language-model head")
    result = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    hidden = getattr(result, "last_hidden_state", None)
    if hidden is None:
        raise ValueError("backbone must return last_hidden_state")
    head = causal.get_output_embeddings()
    if hasattr(head, "lora_A"):
        raise ValueError("output-head adapters are unsupported by restricted-head scoring")
    selected = torch.tensor(token_ids, device=head.weight.device, dtype=torch.long)
    weight = head.weight.index_select(0, selected)
    bias = getattr(head, "bias", None)
    bias = bias.index_select(0, selected) if bias is not None else None
    # Training uses individual unpadded examples plus gradient accumulation.
    return functional.linear(hidden[:, -1, :].to(weight.dtype), weight, bias)


def train(rows: list[dict], output_dir: str | Path, *, model_id: str, revision: str,
          device: str = "cuda", epochs: int = 3, learning_rate: float = 2e-4,
          accumulation_steps: int = 8, rank: int = 8, max_tokens: int = 4096,
          seed: int = 42, allow_synthetic: bool = False, allow_download: bool = False,
          backend: Any = None) -> dict:
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("training output must be a new or empty directory")
    summary = validate_examples(rows, allow_synthetic=allow_synthetic)
    if not revision or revision in {"main", "master", "latest"}:
        raise ValueError("pin --revision to a model commit before training")
    if "exaone" not in model_id.lower():
        raise ValueError("the student must be an EXAONE model")
    if epochs <= 0 or accumulation_steps <= 0 or rank <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, accumulation, rank and learning rate must be positive")
    if backend is None and not device.startswith("cuda"):
        raise ValueError("real model training requires CUDA; use --dry-run until GPU is available")
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("GPU training needs the optional fast-decision ML requirements") from exc
    if backend is None and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --dry-run without downloading model weights")
    if backend is None:
        from .backend import ExaoneChoiceBackend
        backend = ExaoneChoiceBackend(model_id=model_id, revision=revision, device=device,
                                      max_tokens=max_tokens, allow_download=allow_download)
    from .contracts import ChoiceQuestion

    # Tokenize and verify every target before creating an optimizer.
    encoded = []
    for row in rows:
        item = backend.encode_question(ChoiceQuestion(**row["question"]))
        if row["target"] not in item["labels"]:
            raise ValueError("backend dropped the target label")
        if len(set(item["token_ids"])) != len(item["labels"]):
            raise ValueError("choice labels must map to distinct single tokens")
        if not item["input_ids"] or len(item["input_ids"]) > max_tokens:
            raise ValueError("training inputs must be nonempty and fit without truncation")
        encoded.append((item, item["labels"].index(row["target"])))
    torch.manual_seed(seed)
    model = get_peft_model(backend.model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    ))
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=learning_rate)
    model_device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(encoded), generator=generator).tolist()
        losses = []
        for begin in range(0, len(order), accumulation_steps):
            batch = order[begin:begin + accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                item, target = encoded[index]
                ids = torch.tensor([item["input_ids"]], device=model_device, dtype=torch.long)
                logits = _restricted_logits(model, ids, torch.ones_like(ids), item["token_ids"])
                loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([target], device=logits.device))
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss; adapter was not saved")
                (loss / len(batch)).backward()
                losses.append(float(loss.detach().cpu()))
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
            optimizer.step()
        history.append({"epoch": epoch + 1, "mean_restricted_choice_loss": sum(losses) / len(losses)})
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    backend.tokenizer.save_pretrained(destination)
    manifest = {"schema_version": 1, **summary, "model_id": model_id, "revision": revision,
                "loss": "cross_entropy_over_allowed_label_tokens", "head_position": "last_input_token",
                "epochs": epochs, "learning_rate": learning_rate, "lora_rank": rank,
                "accumulation_steps": accumulation_steps, "seed": seed,
                "max_tokens": max_tokens, "history": history,
                "training_rows_fingerprint": fingerprint(rows), "calibrated": False,
                "model_fingerprint": adapter_fingerprint(destination),
                "status": "trained_adapter_requires_independent_evaluation"}
    (destination / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--revision", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit loading uncached model weights.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a new or empty directory")
    rows = read_jsonl(args.input)
    if args.dry_run:
        manifest = {"schema_version": 1, **validate_examples(rows, allow_synthetic=args.allow_synthetic),
                    "status": "dataset_validated_no_model_loaded", "weights_loaded": False,
                    "tokenization_validated": False, "trained": False,
                    "next_step": "Pin an EXAONE revision, connect GPU, and rerun without --dry-run."}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "dry_run.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    else:
        manifest = train(list(rows), args.output, model_id=args.model_id, revision=args.revision,
                         device=args.device, epochs=args.epochs, learning_rate=args.learning_rate,
                         accumulation_steps=args.accumulation_steps, rank=args.rank,
                         max_tokens=args.max_tokens, seed=args.seed, allow_synthetic=args.allow_synthetic,
                         allow_download=args.allow_download)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
