"""EXAONE backbone + trained conditional decision heads, without an LM head.

One backbone forward per event. Small heads preserve place -> spend ->
satisfaction -> factor dependencies without additional language-model passes.
This is our architecture, not a reproduction of TypeSafe's undisclosed RLCD.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .contracts import DEFAULT_MODEL
from .decision_data import IGNORE, KINDS, LABELS, MAX_CANDIDATES

ARCHITECTURE = "exaone_conditional_decision_heads_v1"


@dataclass(frozen=True)
class DecisionConfig:
    hidden_size: int
    decision_width: int = 256
    model_id: str = DEFAULT_MODEL
    revision: str = ""
    architecture: str = ARCHITECTURE

    def __post_init__(self):
        if self.hidden_size < 1 or self.decision_width < 1:
            raise ValueError("positive head dimensions required")
        if not self.model_id.startswith("LGAI-EXAONE/") or self.architecture != ARCHITECTURE:
            raise ValueError("unsupported decision model identity")
        if not self.revision or self.revision in {"main", "master", "latest"}:
            raise ValueError("a pinned base-model revision is required")


@dataclass
class DecisionScores:
    logits: dict[str, torch.Tensor]
    selected: dict[str, torch.Tensor]
    loss: torch.Tensor | None = None


class ConditionalHeads(nn.Module):
    def __init__(self, config: DecisionConfig):
        super().__init__()
        width = config.decision_width
        self.shared = nn.Sequential(nn.Linear(config.hidden_size, width), nn.GELU(), nn.LayerNorm(width))
        sizes = {"poi": MAX_CANDIDATES + 1, **{key: len(values) for key, values in LABELS.items()}}
        self.outputs = nn.ModuleDict({key: nn.Linear(width, sizes[key]) for key in KINDS})
        self.conditions = nn.ModuleDict({key: nn.Embedding(sizes[key], width)
                                        for key in ("poi", "spend", "satisfaction")})

    def forward(self, hidden, candidate_counts, targets=None):
        state = self.shared(hidden.to(self.shared[0].weight.dtype))
        logits, selected, losses = {}, {}, []
        for kind in KINDS:
            scores = self.outputs[kind](state).float()
            if kind == "poi":
                slots = torch.arange(MAX_CANDIDATES + 1, device=scores.device)[None, :]
                allowed = (slots < candidate_counts[:, None]) | (slots == MAX_CANDIDATES)
                scores = scores.masked_fill(~allowed, float("-inf"))
            logits[kind] = scores
            choice = scores.argmax(-1)
            if targets is not None:
                label = targets[kind]
                active = label != IGNORE
                if active.any():
                    if (label[active] < 0).any() or (label[active] >= scores.shape[1]).any():
                        raise ValueError(f"invalid {kind} target")
                    if not torch.isfinite(scores[active].gather(1, label[active, None])).all():
                        raise ValueError("target selects an unavailable candidate")
                    losses.append(nn.functional.cross_entropy(scores[active], label[active]))
                    choice = torch.where(active, label, choice)
            selected[kind] = choice
            if kind in self.conditions:
                state = state + self.conditions[kind](choice)
        loss = torch.stack(losses).mean() if losses else None
        return DecisionScores(logits, selected, loss)


class ExaoneDecisionModel(nn.Module):
    def __init__(self, encoder: nn.Module, config: DecisionConfig):
        super().__init__()
        if getattr(encoder.config, "model_type", None) != "exaone4":
            raise ValueError("this decision architecture requires an EXAONE 4 backbone")
        if encoder.config.hidden_size != config.hidden_size:
            raise ValueError("head and EXAONE hidden dimensions differ")
        self.encoder, self.config = encoder, config
        self.heads = ConditionalHeads(config)

    def forward(self, input_ids, attention_mask, candidate_counts, targets=None):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("input IDs and masks must be matching matrices")
        batch, width = input_ids.shape
        if width == 0 or ((attention_mask != 0) & (attention_mask != 1)).any() or (attention_mask.sum(-1) == 0).any():
            raise ValueError("every input needs a nonempty binary attention mask")
        if candidate_counts.shape != (batch,) or candidate_counts.dtype != torch.long or (candidate_counts < 0).any() or (candidate_counts > MAX_CANDIDATES).any():
            raise ValueError("invalid candidate counts")
        if targets is not None:
            if set(targets) != set(KINDS) or any(value.shape != (batch,) or value.dtype != torch.long for value in targets.values()):
                raise ValueError("all head targets must be long tensors, one per input")
            # No downstream supervision after a route/action DEFER or missing label.
            stopped = targets["route"] != 0
            for kind in KINDS[1:]:
                if (stopped & (targets[kind] != IGNORE)).any():
                    raise ValueError("downstream labels after DEFER must be masked")
                defer = MAX_CANDIDATES if kind == "poi" else len(LABELS[kind]) - 1
                stopped = stopped | (targets[kind] == IGNORE) | (targets[kind] == defer)
        positions = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        result = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              position_ids=positions, use_cache=False, return_dict=True)
        # Works with left and right padding; never pool a padding token.
        last = torch.arange(width, device=input_ids.device)[None, :].masked_fill(attention_mask == 0, -1).max(-1).values
        hidden = result.last_hidden_state[torch.arange(batch, device=input_ids.device), last]
        return self.heads(hidden, candidate_counts, targets)


def encode_text(tokenizer, text: str, max_tokens: int) -> list[int]:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"decision input has {len(ids)} tokens; limit={max_tokens}, no truncation")
    return ids


def checkpoint_fingerprint(directory: str | Path) -> str:
    from .training import adapter_fingerprint
    directory = Path(directory)
    digest = hashlib.sha256(adapter_fingerprint(directory / "adapter").encode())
    paths = [directory / name for name in ("decision_config.json", "decision_heads.safetensors")]
    tokenizer_paths = sorted(path for path in (directory / "tokenizer").rglob("*") if path.is_file())
    if not tokenizer_paths:
        raise ValueError("checkpoint tokenizer files are missing")
    for path in paths + tokenizer_paths:
        digest.update(path.relative_to(directory).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(model: ExaoneDecisionModel, tokenizer, directory: str | Path) -> str:
    from safetensors.torch import save_file
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("decision checkpoint needs a new or empty directory")
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(directory / "adapter")
    tokenizer.save_pretrained(directory / "tokenizer")
    config = {**asdict(model.config), "max_candidates": MAX_CANDIDATES,
              "labels": {key: list(value) for key, value in LABELS.items()}}
    (directory / "decision_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_file({key: value.detach().cpu().contiguous() for key, value in model.heads.state_dict().items()},
              str(directory / "decision_heads.safetensors"))
    return checkpoint_fingerprint(directory)


def load_checkpoint(directory: str | Path, *, device="cpu", allow_download=False, encoder=None):
    """Load trained heads AND their matching EXAONE LoRA; never random heads."""
    from peft import PeftModel
    from safetensors.torch import load_file
    directory = Path(directory)
    manifest = json.loads((directory / "training_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("architecture") != ARCHITECTURE or manifest.get("model_fingerprint") != checkpoint_fingerprint(directory):
        raise ValueError("decision checkpoint identity or weight fingerprint mismatch")
    if manifest.get("trained") is not True:
        raise ValueError("checkpoint has no completed decision training")
    if encoder is None and (manifest.get("injected_test_encoder") or manifest.get("synthetic_examples")):
        raise ValueError("test/synthetic checkpoint requires an explicitly injected test encoder")
    raw = json.loads((directory / "decision_config.json").read_text(encoding="utf-8"))
    if raw.pop("max_candidates") != MAX_CANDIDATES or raw.pop("labels") != {key: list(value) for key, value in LABELS.items()}:
        raise ValueError("checkpoint decision schema differs from runtime")
    config = DecisionConfig(**raw)
    if manifest.get("model_id") != config.model_id or manifest.get("revision") != config.revision:
        raise ValueError("checkpoint base-model identity differs from manifest")
    if encoder is None:
        from transformers import AutoModel
        dtypes = {"torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}
        if manifest.get("base_dtype") not in dtypes:
            raise ValueError("checkpoint lacks a supported backbone precision")
        dtype = torch.float32 if device == "cpu" else dtypes[manifest["base_dtype"]]
        encoder = AutoModel.from_pretrained(config.model_id, revision=config.revision,
                                            local_files_only=not allow_download, trust_remote_code=False,
                                            torch_dtype=dtype)
    encoder = PeftModel.from_pretrained(encoder, directory / "adapter")
    model = ExaoneDecisionModel(encoder, config)
    model.heads.load_state_dict(load_file(str(directory / "decision_heads.safetensors")), strict=True)
    return model.to(device).eval(), manifest
