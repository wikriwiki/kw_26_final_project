"""정책 본런 zip 에서 '실행' 화면용 파일을 한 번만 꺼낸다 — 드라이브에서 zip 을 여러 번 읽지 않도록.

꺼내는 것 (실험 정의의 ``source`` 를 따른다):
* ``<root>/{pre,on}/`` 의 ``summary.json`` · ``day_*.json``(하루치 요약) · ``stage1_failures.jsonl`` · ``timing/*`` · ``checkpoints/*``
* ``<root>/on/dossier.jsonl`` — 정책 있음 쪽 사람별 계획·기억·상태(정책 전 기간 포함)
* ``<root>/{roster.json, run_manifest.json, engine.json}``

일자 기록(metrics)은 ``import_policy_runs.py`` 가 실험 미러에 이미 꺼내 두었으므로 여기서 꺼내지 않는다.
CRC 를 검사하며 읽고, 같은 크기의 파일이 있으면 건너뛴다. 원본 zip 은 읽기만 한다.

    python web/tools/extract_policy_runs.py --src "G:/내 드라이브/KW_시뮬결과_공유_20261011" --out <폴더>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = ("pre", "on")
ARM_FILES = ("summary.json", "stage1_failures.jsonl", "stage2.json")
ARM_DIRS = ("timing/", "checkpoints/")
TOP_FILES = ("roster.json", "run_manifest.json", "engine.json")
DAY_SUMMARY = re.compile(r"^day_\d{4}-\d{2}-\d{2}\.json$")  # 하루치 실행 요약


def wanted(name: str, root: str) -> bool:
    rel = name[len(root) + 1:] if name.startswith(root + "/") else None
    if not rel or rel.endswith("/"):
        return False
    if rel in TOP_FILES or rel == "on/dossier.jsonl":
        return True
    arm, _, rest = rel.partition("/")
    if arm not in ARMS:
        return False
    return rest in ARM_FILES or bool(DAY_SUMMARY.match(rest)) or any(rest.startswith(d) for d in ARM_DIRS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out_root = Path(args.out)
    for path in sorted((REPO / "web" / "experiments").glob("*.json")):
        definition = json.loads(path.read_text(encoding="utf-8"))
        source = definition.get("source")
        exp_id = definition["experiment_id"]
        if not source or (args.only and exp_id not in args.only):
            continue
        zpath = Path(args.src) / source["zip"]
        dest = out_root / exp_id
        with zipfile.ZipFile(zpath) as z:
            members = [i for i in z.infolist() if wanted(i.filename, source["root"])]
            copied = skipped = 0
            for info in members:
                target = dest / info.filename[len(source["root"]) + 1:]
                if target.exists() and target.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(target.suffix + ".part")
                with z.open(info) as src, tmp.open("wb") as dst:  # z.open 이 CRC 를 검사한다
                    shutil.copyfileobj(src, dst, 1 << 20)
                tmp.replace(target)
                copied += 1
        print(f"{exp_id}: {copied} copied, {skipped} skipped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
