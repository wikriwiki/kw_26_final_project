"""Read recorded model provenance; never relabel past runs as today's default."""
from datetime import timedelta
import json
import os
from pathlib import Path


def recorded_model_label(start, days, output_dir=None):
    root = Path(output_dir or os.environ.get("SIM_OUTPUT_DIR", "~/sim_output")).expanduser()
    models = set()
    for offset in range(days):
        day = (start + timedelta(days=offset)).isoformat()
        path = root / "metrics" / f"day_{day}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                model = (row.get("decision_provenance") or {}).get("model_id")
                if isinstance(model, str) and model:
                    models.add(model)
    return ", ".join(sorted(models)) if models else "모델 실행 기록 없음"
