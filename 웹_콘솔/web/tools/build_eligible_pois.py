"""캐시백 실적으로 치는 가게 목록을 실행 결제 기록에서 뽑는다.

정책 있음 실행의 결제 기록(events.jsonl)에는 결제마다 실적 인정 여부(elig)가 있다. 인정 여부는 가게 성격(대형마트·
백화점·온라인 등 제외)으로 정해지므로 가게(poi_id) 단위로 모은다. 정책 없음 쪽 결제에도 같은 표를 써서
적립 업종 소비를 셀 수 있다.

    python web/tools/build_eligible_pois.py --events <runs>/out_P012/events.jsonl \
        --out <SIM_EXPERIMENTS_ROOT>/p012_sangsaeng_cashback/bundle/eligible_pois.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    seen: dict[str, set[bool]] = defaultdict(set)
    with args.events.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("poi_id") and row.get("elig") is not None:
                seen[str(row["poi_id"])].add(bool(row["elig"]))
    mixed = [poi for poi, values in seen.items() if len(values) > 1]
    # 한 가게가 두 값을 가지면 인정하지 않는 쪽으로 둔다(실적을 부풀리지 않게)
    table = {poi: (values == {True}) for poi, values in sorted(seen.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    print(f"가게 {len(table)}곳 · 인정 {sum(table.values())}곳 · 값이 섞인 곳 {len(mixed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
