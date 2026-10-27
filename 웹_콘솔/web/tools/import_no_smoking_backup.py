"""doinggyu 의 금연구역 실험 백업(구글 드라이브 No_SmokingZone_EXP_Backups)을 콘솔 미러로 옮긴다.

원격 서버(vast.ai)가 없어진 뒤에는 ``experiment_sync`` 로 받을 수 없다. 드라이브 백업에는 날짜마다
``checkpoints/<날짜>/<스냅샷>/run-day.tar.gz`` 가 있고, 그 안에 그날의 ``metrics/day_*.jsonl`` 이 있다.

* 날짜마다 committed.json 이 있는 스냅샷 중 가장 늦은 것을 쓴다. checkpoint.json 의 sha256 과 대조한다.
* 정책 전 첫날(11-19)은 v22 가 이어받은 v19 런의 체크포인트에 있다(checkpoint.json 의 parent_checkpoints).
* 인터뷰 근거(evidence/)는 하루 200MB 가 넘어 옮기지 않는다. 인터뷰는 하루 기록(상태·영수증·대상 시설)으로 답한다.
* 실험 번들(personas.json·runtime.json)은 따로 받은 experiment-bundles-3gu-1154-v1.tar.gz 에서 꺼낸다.

    python web/tools/import_no_smoking_backup.py --backup "G:/.shortcut-targets-by-id/<폴더id>/No_SmokingZone_EXP_Backups" \
        --bundle-tar C:/Users/<나>/Downloads/experiment-bundles-3gu-1154-v1.tar.gz \
        --dest C:/Users/<나>/kw_web_data/experiments/no_smoking_3gu_v22
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import date, timedelta
from pathlib import Path

ARMS = {
    # arm_id: (런 폴더, 시작일, 일수)
    "shared-pre": ("integration-main-v22-1154-shared-pre", "2017-11-19", 14),
    "post-off": ("integration-main-v22-1154-post-off", "2017-12-03", 14),
    "post-on": ("integration-main-v22-1154-post-on", "2017-12-03", 14),
}
# v22 정책 전 런은 v19 의 11-19 를 이어받았다
INHERITED = {("shared-pre", "2017-11-19"): "integration-main-v19-1154-shared-pre"}
BUNDLE = "full_3gu_1154_v1"


def days(start: str, count: int) -> list[str]:
    first = date.fromisoformat(start)
    return [(first + timedelta(days=i)).isoformat() for i in range(count)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(run_dir: Path, day: str) -> Path | None:
    """그날의 확정된 스냅샷 — checkpoints 를 먼저, 없으면 recoverable-progress."""
    for kind in ("checkpoints", "recoverable-progress"):
        root = run_dir / kind / day
        if not root.is_dir():
            continue
        done = sorted(p for p in root.iterdir() if (p / "committed.json").is_file() and (p / "run-day.tar.gz").is_file())
        if done:
            return done[-1]
    return None


def extract(archive: Path, wanted: dict[str, Path]) -> list[str]:
    """이름이 정확히 맞는 멤버만 정해 둔 자리에 쓴다(압축 안의 경로를 그대로 믿지 않는다)."""
    found = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            target = wanted.get(member.name)
            if target is None or not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as out:  # type: ignore[union-attr]
                shutil.copyfileobj(source, out)
            found.append(member.name)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backup", required=True, type=Path, help="No_SmokingZone_EXP_Backups 폴더")
    parser.add_argument("--bundle-tar", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path, help="<SIM_EXPERIMENTS_ROOT>/no_smoking_3gu_v22")
    args = parser.parse_args(argv)
    runs = args.backup / "runs"
    report: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as work:
        local = Path(work) / "run-day.tar.gz"
        for arm_id, (run_name, start, count) in ARMS.items():
            arm_dir = args.dest / "arms" / arm_id
            rows = report.setdefault(arm_id, {})
            for day in days(start, count):
                if (arm_dir / "metrics" / f"day_{day}.jsonl").is_file() and (arm_dir / f"night2_completed_{day}.json").is_file():
                    rows[day] = {"skipped": "이미 받음"}
                    continue
                run_dir = runs / INHERITED.get((arm_id, day), run_name)
                snap = snapshot(run_dir, day)
                if snap is None:
                    rows[day] = "스냅샷 없음"
                    print(arm_id, day, "스냅샷 없음", flush=True)
                    continue
                expected = json.loads((snap / "checkpoint.json").read_text(encoding="utf-8"))["files"]["run-day.tar.gz"]["sha256"]
                shutil.copyfile(snap / "run-day.tar.gz", local)
                actual = sha256(local)
                if actual != expected:
                    rows[day] = f"sha256 불일치 {actual[:12]} != {expected[:12]}"
                    print(arm_id, day, rows[day], flush=True)
                    continue
                found = extract(local, {
                    f"run/metrics/day_{day}.jsonl": arm_dir / "metrics" / f"day_{day}.jsonl",
                    f"run/night2_completed_{day}.json": arm_dir / f"night2_completed_{day}.json",
                })
                rows[day] = {"snapshot": f"{snap.parent.parent.name}/{day}/{snap.name}", "sha256": actual, "files": found}
                print(arm_id, day, snap.name, len(found), "개", flush=True)
            # 런 정의는 마지막 확정본(final/)에서
            # 드라이브 데스크톱은 폴더마다 desktop.ini 를 보여 준다 — 런 정의가 든 폴더만 고른다
            final_root = runs / run_name / "final"
            finals = sorted(p for p in final_root.iterdir() if (p / "experiment_run.json").is_file()) if final_root.is_dir() else []
            if finals:
                shutil.copyfile(finals[-1] / "experiment_run.json", arm_dir / "experiment_run.json")
    found = extract(args.bundle_tar, {
        f"{BUNDLE}/personas.json": args.dest / "bundle" / "personas.json",
        f"{BUNDLE}/runtime.json": args.dest / "bundle" / "runtime.json",
    })
    report["bundle"] = {"source": args.bundle_tar.name, "files": found, "sha256": sha256(args.bundle_tar)}
    (args.dest / "_import.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = [f"{a}:{d}" for a, rows in report.items() if a != "bundle" for d, v in rows.items() if not isinstance(v, dict)]
    print("빠진 날:", ", ".join(missing) if missing else "없음")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
