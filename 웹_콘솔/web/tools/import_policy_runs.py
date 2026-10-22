"""정책 본런 결과 zip 을 실험 미러로 들여온다 — 콘솔의 "실험" 화면이 그대로 읽는다.

왜 필요한가
-----------
본런(P010·P012·P013·거리두기·P017)은 A100·Colab 에서 돌았고, 결과는 구글 드라이브의
`KW_시뮬결과_공유_20261011/<런>/<런>_results_*.zip` 에 있다. zip 안은 실험 레지스트리와
같은 설계다 — `pre/`(정책 전 공통) · `off/`(정책 없음) · `on/`(정책 있음) 세 갈래, 갈래마다
`metrics/day_*.jsonl`(한 줄 = 시민 하루, `integrity_sha256` 봉인).

이 스크립트는 `web/experiments/<실험>.json` 의 `source` 를 따라 zip 에서 일자 기록만 꺼내
`<미러>/<실험>/arms/<팔>/metrics/` 에 둔다. 콘솔은 그 줄을 다시 읽어 봉인을 검사한다.

* zip 은 CRC 를 검사하며 읽는다. 드라이브의 덜 내려받은 파일은 여기서 실패로 드러난다.
* 같은 크기의 파일이 이미 있으면 건너뛴다(다시 돌려도 안전).
* 명단(`roster.json`)으로 그래프 원본 페르소나(`agents_final.json.bak_orig`)를 걸러
  `bundle/personas.json` 을 만든다. 명단에 있는데 페르소나에 없는 사람이 있으면 멈춘다.
* 팔마다 `experiment_run.json`(상태·모델·명단·엔진 지문)을 쓴다.

원본 zip 과 페르소나 파일은 읽기만 한다.

사용법
------
    python web/tools/import_policy_runs.py \\
        --src "G:/내 드라이브/KW_시뮬결과_공유_20261011" \\
        --personas "G:/내 드라이브/Kw/final_project/data/neo4j_load/agents/agents_final.json.bak_orig" \\
        [--out <미러 루트>] [--only p010_minsaeng_coupon ...]

미러 루트는 콘솔과 같은 규칙으로 정한다(`SIM_EXPERIMENTS_ROOT` → `SIM_DATA_ROOT/experiments`
→ `~/sim_data/experiments`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from web.api.experiments import REGISTRY_DIR, experiments_root  # noqa: E402

DAY_FILE = re.compile(r"^(?:metrics/day_(\d{4}-\d{2}-\d{2})\.jsonl|night2_completed_(\d{4}-\d{2}-\d{2})\.json)$")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_one(definition: dict, src_root: Path, personas: dict[str, dict], out_root: Path) -> dict:
    eid = definition["experiment_id"]
    source = definition["source"]
    zip_path = src_root / source["zip"]
    root = source["root"]
    target = out_root / eid
    stats = {"experiment": eid, "zip": source["zip"], "files": 0, "skipped": 0, "bytes": 0}

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read(f"{root}/run_manifest.json"))
        engine = json.loads(archive.read(f"{root}/engine.json")) if f"{root}/engine.json" in names else {}
        roster = json.loads(archive.read(f"{root}/roster.json"))
        roster_ids = [item if isinstance(item, str) else item.get("agent_id") for item in roster]
        missing = [aid for aid in roster_ids if aid not in personas]
        if missing:
            raise SystemExit(f"{eid}: 명단 {len(missing)}명이 페르소나 원본에 없습니다 (예: {missing[:3]}). 다른 세대 파일일 수 있습니다.")

        arms_by_id = {arm["arm_id"]: arm for arm in definition["arms"]}
        for arm_id, folder in source["arms"].items():
            arm = arms_by_id[arm_id]
            arm_dir = target / "arms" / arm_id
            prefix = f"{root}/{folder}/"
            for name in sorted(n for n in names if n.startswith(prefix)):
                relative = name[len(prefix):]
                match = DAY_FILE.match(relative)
                if not match:
                    continue
                destination = arm_dir / relative
                info = archive.getinfo(name)
                if destination.is_file() and destination.stat().st_size == info.file_size:
                    stats["skipped"] += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_suffix(destination.suffix + ".part")
                # ZipExtFile 은 끝까지 읽을 때 CRC 를 대조한다. 어긋나면 BadZipFile 로 멈춘다
                with archive.open(info) as reader, partial.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, 1 << 20)
                partial.replace(destination)
                stats["files"] += 1
                stats["bytes"] += info.file_size

            _write_json(
                arm_dir / "experiment_run.json",
                {
                    "status": "complete",
                    "arm": arm_id,
                    "phase": arm["role"],
                    "start": arm["start"],
                    "days": arm["days"],
                    "model": manifest.get("served_model"),
                    "workers": manifest.get("workers_per_arm"),
                    "prompt_variant": manifest.get("prompt_variant"),
                    "interpretation": definition.get("interpretation"),
                    "cohort_ids": roster_ids,
                    "runtime_sha256": engine.get("execution_fingerprint"),
                    "source_zip": source["zip"],
                    "source_folder": f"{root}/{folder}",
                },
            )

    _write_json(target / "bundle" / "personas.json", [personas[aid] for aid in roster_ids])
    _write_json(
        target / "_sync.json",
        {
            "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "drive_zip",
            "zip": source["zip"],
            "zip_bytes": zip_path.stat().st_size,
            "zip_sha256": _sha256(zip_path),
            "files_written": stats["files"],
            "files_skipped": stats["skipped"],
        },
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--src", required=True, help="KW_시뮬결과_공유_20261011 폴더")
    parser.add_argument("--personas", required=True, help="agents_final.json.bak_orig (그래프 원본 14,881명)")
    parser.add_argument("--out", help="실험 미러 루트 (기본: 콘솔과 같은 규칙)")
    parser.add_argument("--only", nargs="*", help="이 실험만")
    args = parser.parse_args()

    out_root = Path(args.out) if args.out else experiments_root()
    personas_raw = json.loads(Path(args.personas).read_text(encoding="utf-8"))
    personas = {item["agent_id"]: item for item in personas_raw}

    definitions = []
    for path in sorted(REGISTRY_DIR.glob("*.json")):
        definition = json.loads(path.read_text(encoding="utf-8"))
        if (definition.get("source") or {}).get("kind") != "drive_zip":
            continue
        if args.only and definition["experiment_id"] not in args.only:
            continue
        definitions.append(definition)

    for definition in definitions:
        stats = import_one(definition, Path(args.src), personas, out_root)
        print(
            f"{stats['experiment']}: 새로 {stats['files']}개 ({stats['bytes'] / 1e6:.0f}MB), "
            f"이미 있음 {stats['skipped']}개 ← {stats['zip']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
