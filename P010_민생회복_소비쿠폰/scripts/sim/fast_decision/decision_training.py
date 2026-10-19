"""Train EXAONE-SimDecision: LoRA backbone AND dedicated conditional heads.

--dry-run needs no torch, weights or GPU. Actual pretrained-base training waits
for an explicitly selected CUDA device. An injected tiny encoder is test-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .contracts import DEFAULT_MODEL
from .dataset import fingerprint, read_jsonl
from .decision_data import KINDS, model_examples
from .training import validate_examples


def prepare(rows, *, allow_synthetic=False):
    summary = validate_examples(rows, allow_synthetic=allow_synthetic)
    examples = model_examples(rows, allow_synthetic=allow_synthetic)
    summary.update(decision_examples=len(examples),
                   route_targets=dict(Counter("PROCEED" if example.targets["route"] == 0 else "DEFER" for example in examples)),
                   route_target_semantics="conservative_scope_proxy_not_correctness_or_real_behavior_probability",
                   architecture="exaone_conditional_decision_heads_v1",
                   model_name="EXAONE-SimDecision-1.2B")
    return summary, examples


def train_decision(rows, output_dir, *, revision, model_id=DEFAULT_MODEL, device="cuda",
                   epochs=3, learning_rate=1e-4, accumulation_steps=8, rank=8,
                   decision_width=256, max_tokens=4096, seed=42, allow_synthetic=False,
                   allow_download=False, encoder=None, tokenizer=None):
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("output must be a new or empty directory")
    summary, examples = prepare(rows, allow_synthetic=allow_synthetic)
    if not model_id.startswith("LGAI-EXAONE/") or not revision or revision in {"main", "master", "latest"}:
        raise ValueError("an official EXAONE model and pinned revision are required")
    if any(value < 1 for value in (epochs, accumulation_steps, rank, decision_width, max_tokens)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("training settings must be positive and finite")
    if (encoder is None) != (tokenizer is None):
        raise ValueError("inject encoder and tokenizer together for CPU tests")
    injected = encoder is not None
    if not injected and not device.startswith("cuda"):
        raise ValueError("pretrained decision training requires CUDA; use --dry-run for now")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from .decision_model import DecisionConfig, ExaoneDecisionModel, encode_text, save_checkpoint
    if not injected:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no weights downloaded, use --dry-run")
        from transformers import AutoModel, AutoTokenizer
        with torch.cuda.device(device):
            base_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                  local_files_only=not allow_download, trust_remote_code=False)
        encoder = AutoModel.from_pretrained(model_id, revision=revision,
                                            local_files_only=not allow_download, trust_remote_code=False,
                                            torch_dtype=base_dtype)
    torch.manual_seed(seed)
    encoded = [encode_text(tokenizer, example.text, max_tokens) for example in examples]
    if getattr(encoder.config, "model_type", None) != "exaone4":
        raise ValueError("EXAONE 4 backbone required")
    adapted = get_peft_model(encoder, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"))
    model = ExaoneDecisionModel(adapted, DecisionConfig(
        hidden_size=encoder.config.hidden_size, decision_width=decision_width,
        model_id=model_id, revision=revision)).to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    # FP16 backprop needs scaling. BF16 and CPU smoke tests use normal gradients.
    scaler = torch.amp.GradScaler("cuda", enabled=(not injected and base_dtype == torch.float16))
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), accumulation_steps):
            batch = order[start:start + accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                ids = torch.tensor([encoded[index]], device=device, dtype=torch.long)
                example = examples[index]
                target = {kind: torch.tensor([example.targets[kind]], device=device, dtype=torch.long) for kind in KINDS}
                result = model(ids, torch.ones_like(ids),
                               torch.tensor([len(example.candidate_ids)], device=device, dtype=torch.long), target)
                if result.loss is None or not torch.isfinite(result.loss):
                    raise RuntimeError("invalid decision loss; no checkpoint saved")
                scaler.scale(result.loss / len(batch)).backward()
                losses.append(float(result.loss.detach().cpu()))
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite decision gradient; no checkpoint saved")
            scaler.step(optimizer)
            scaler.update()
        history.append({"epoch": epoch + 1, "mean_conditional_head_loss": sum(losses) / len(losses)})
    model.eval()
    model_fingerprint = save_checkpoint(model, tokenizer, destination)
    manifest = {"schema_version": 1, **summary, "model_id": model_id, "revision": revision,
                "model_fingerprint": model_fingerprint, "pretrained_base_loaded": not injected,
                "injected_test_encoder": injected, "trained": True, "calibrated": False,
                "loss": "masked_conditional_head_cross_entropy", "encoder_forwards_per_event": 1,
                "teacher_forcing": "preceding_actions_and_head_targets_during_training_only",
                "lora_rank": rank, "decision_width": decision_width, "max_tokens": max_tokens,
                "base_dtype": str(next(encoder.parameters()).dtype), "head_dtype": str(next(model.heads.parameters()).dtype),
                "epochs": epochs, "learning_rate": learning_rate, "accumulation_steps": accumulation_steps,
                "seed": seed, "history": history, "training_rows_fingerprint": fingerprint(rows),
                "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "status": "trained_decision_model_requires_independent_evaluation",
                "limitations": ["New heads require representative supervision; random heads are unusable.",
                                "Route scope labels are not empirical error-risk labels.",
                                "Independent rollout, calibration and real-world validation remain required."]}
    (destination / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", default="")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--decision-width", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be a new or empty directory")
    rows = list(read_jsonl(args.input))
    if args.dry_run:
        summary, examples = prepare(rows, allow_synthetic=args.allow_synthetic)
        report = {**summary, "trained": False, "weights_loaded": False, "gpu_used": False,
                  "tokenization_validated": False, "status": "decision_supervision_validated_no_model_loaded",
                  "masked_targets": {kind: sum(example.targets[kind] == -100 for example in examples) for kind in KINDS}}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "dry_run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        report = train_decision(rows, args.output, revision=args.revision, model_id=args.model_id,
                                device=args.device, epochs=args.epochs, learning_rate=args.learning_rate,
                                accumulation_steps=args.accumulation_steps, rank=args.rank,
                                decision_width=args.decision_width, max_tokens=args.max_tokens, seed=args.seed,
                                allow_synthetic=args.allow_synthetic, allow_download=args.allow_download)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
