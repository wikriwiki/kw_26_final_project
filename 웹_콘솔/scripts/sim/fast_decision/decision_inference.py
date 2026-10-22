"""Offline replay for the trained EXAONE decision-head model. Never live routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .contracts import validate_output
from .dataset import read_jsonl
from .decision_data import KINDS, LABELS, MAX_CANDIDATES, decision_text
from .planner import preflight


class DecisionPredictor:
    def __init__(self, model, tokenizer, *, max_tokens=4096):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.model, self.tokenizer, self.max_tokens = model.eval(), tokenizer, max_tokens

    def event(self, snapshot, picks, order):
        import torch
        from .decision_model import encode_text
        text, candidate_ids = decision_text(snapshot, picks, order)
        ids = encode_text(self.tokenizer, text, self.max_tokens)
        device = next(self.model.parameters()).device
        tensor = torch.tensor([ids], device=device, dtype=torch.long)
        with torch.inference_mode():
            result = self.model(tensor, torch.ones_like(tensor),
                                torch.tensor([len(candidate_ids)], device=device, dtype=torch.long))
        selected, scores = {}, {}
        for kind in KINDS:
            values = result.logits[kind][0]
            labels = LABELS.get(kind)
            if kind == "poi":
                indices = list(range(len(candidate_ids))) + [MAX_CANDIDATES]
                values = values[indices]
                labels = candidate_ids + ("DEFER",)
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite decision head scores")
            probabilities = values.softmax(-1)
            selected[kind] = labels[int(values.argmax())]
            scores[kind] = {"labels": list(labels), "logits": values.cpu().tolist(),
                            "probabilities": probabilities.cpu().tolist()}
        return selected, scores, len(ids)

    def propose(self, snapshot):
        reasons = preflight(snapshot)
        report = {"status": "deferred", "output": None, "reasons": reasons, "scores": [],
                  "eligible_for_live": False, "encoder_forwards": 0, "selection": "argmax",
                  "probability_semantics": "uncalibrated_conditional_head_scores"}
        if reasons:
            return report
        picks = []
        for order in sorted(int(key) for key, candidates in snapshot["candidates"].items() if candidates):
            selected, scores, tokens = self.event(snapshot, picks, order)
            report["encoder_forwards"] += 1
            report["scores"].append({"order": order, "heads": scores, "input_tokens": tokens})
            for kind in KINDS:
                if selected[kind] == "DEFER":
                    report["reasons"] = [f"decision_model_deferred:{kind}"]
                    return report
            if selected["poi"] in {pick["poi_id"] for pick in picks}:
                report["reasons"] = ["duplicate_poi_in_day"]
                return report
            pick = {"order": order, "poi_id": selected["poi"], "actual_spent": int(selected["spend"]),
                    "actual_satisfaction": float(selected["satisfaction"]),
                    "pick_factor": selected["factor"], "policy_spend": {}}
            pick["pick_reason"] = f"경량 실험 선택: 후보 {pick['poi_id']}; 요인 코드 {pick['pick_factor']}"
            picks.append(pick)
        output = {"picks": picks, "review_lookup_requests": []}
        errors = validate_output(snapshot, output)
        if errors:
            report["reasons"] = errors
        else:
            report.update(status="proposed", output=output)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("choose a new output path")
    from transformers import AutoTokenizer
    from .decision_model import load_checkpoint
    from .runtime import append_record
    model, manifest = load_checkpoint(args.checkpoint, device=args.device, allow_download=args.allow_download)
    if manifest.get("injected_test_encoder") or manifest.get("synthetic_examples"):
        parser.error("test/synthetic checkpoints cannot be used for actual capture replay")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True, trust_remote_code=False)
    predictor = DecisionPredictor(model, tokenizer, max_tokens=args.max_tokens)
    count = errors = 0
    for row in read_jsonl(args.input):
        started = time.perf_counter()
        try:
            student = predictor.propose(row["snapshot"])
        except Exception as exc:
            errors += 1
            student = {"status": "error", "error_type": type(exc).__name__, "error": str(exc), "eligible_for_live": False}
        student.update(latency_seconds=time.perf_counter() - started,
                       model_id=manifest["model_id"], revision=manifest["revision"],
                       model_fingerprint=manifest["model_fingerprint"], architecture=manifest["architecture"])
        row["student"] = student
        append_record(args.output, row)
        count += 1
    print(json.dumps({"records": count, "errors": errors, "eligible_for_live": False}))
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
