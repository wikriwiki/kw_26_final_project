"""Post-result read-only diagnosis of an unchanged P014 KSIC join gate.

Separates missing merchant IDs from missing or malformed classification codes.
It never changes the KSIC proxy, thresholds, groups, or published point values.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from score_p014_ksic2026_receipts_20260928 import has_resolved_industry, sha256


REASONS = ("source_merchant_id_absent", "source_industry_code_blank",
           "source_industry_code_malformed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    proxy = json.loads(args.proxy.read_text(encoding="utf-8"))
    if proxy.get("schema") != "p014_ksic2026_posthoc_receipt_proxy_v1":
        raise ValueError("unexpected KSIC proxy schema")
    if proxy.get("technical_gate_pass") is not False:
        raise ValueError("expected unchanged failed technical join gate")
    source = Path(proxy["source"]["path"])
    if sha256(source) != proxy["source"]["sha256"]:
        raise ValueError("catalog SHA differs from the scored proxy")
    numeric_path = Path(proxy["numeric_path"])
    if sha256(numeric_path) != proxy["numeric_sha256"]:
        raise ValueError("frozen numeric SHA changed")
    wanted = set(proxy["on"]["unmatched_pois"] + proxy["off"]["unmatched_pois"])
    raw_code_by_poi = {}
    with source.open(encoding="utf-8-sig", errors="strict", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            poi = "C_" + row["상가업소번호"]
            if poi in wanted:
                if poi in raw_code_by_poi:
                    raise ValueError("ambiguous unmatched merchant-ID join")
                raw_code_by_poi[poi] = row["표준산업분류코드"]
    arms = {}
    for arm in ("on", "off"):
        expected = proxy[arm]
        by_reason = {reason: [] for reason in REASONS}
        positive_total = 0
        for file in expected["metrics_evidence"]:
            path = Path(file["path"])
            if sha256(path) != file["sha256"]:
                raise ValueError("canonical metrics SHA changed")
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for receipt in row.get("execution_receipts") or []:
                    if receipt.get("kind") != "purchase_receipt" or receipt["amount"] <= 0:
                        continue
                    positive_total += 1
                    poi = receipt["poi_id"]
                    if poi not in expected["unmatched_pois"]:
                        continue
                    if poi not in raw_code_by_poi:
                        reason = "source_merchant_id_absent"
                    elif not raw_code_by_poi[poi]:
                        reason = "source_industry_code_blank"
                    elif not has_resolved_industry(raw_code_by_poi[poi]):
                        reason = "source_industry_code_malformed"
                    else:
                        raise ValueError("supposed unmatched receipt has resolved source classification")
                    by_reason[reason].append(receipt)
        if positive_total != expected["positive_receipts"]:
            raise ValueError("receipt denominator differs from proxy")
        reasons = {}
        for reason, receipts in by_reason.items():
            reasons[reason] = {"positive_receipts": len(receipts),
                               "won": sum(row["amount"] for row in receipts),
                               "unique_citizens": len({row["agent_id"] for row in receipts}),
                               "unique_pois": len({row["poi_id"] for row in receipts}),
                               "poi_ids": sorted({row["poi_id"] for row in receipts})}
        if (sum(stats["positive_receipts"] for stats in reasons.values()) != expected["unmatched_receipts"]
                or sum(stats["won"] for stats in reasons.values()) != expected["unmatched_receipt_won"]):
            raise ValueError("unmatched count/won partition differs from proxy")
        arms[arm] = {"unmatched_receipts": expected["unmatched_receipts"],
                     "unmatched_won": expected["unmatched_receipt_won"],
                     "by_reason": reasons}
    payload = {"schema": "p014_ksic2026_join_failure_diagnostic_v1",
               "purpose": "Post-result missingness diagnosis; unchanged failed 99% gate and null growth values",
               "proxy_path": args.proxy.as_posix(), "proxy_sha256": sha256(args.proxy),
               "numeric_path": proxy["numeric_path"], "numeric_sha256": proxy["numeric_sha256"],
               "source_path": source.as_posix(), "source_sha256": proxy["source"]["sha256"],
               "tool_path": Path(__file__).as_posix(), "tool_sha256": sha256(Path(__file__)),
               "on": arms["on"], "off": arms["off"],
               "original_graph_ingest_byte_identity_confirmed": False,
               "conclusion": "Recovered catalog lacks some receipt merchant IDs; more citizens alone cannot guarantee classification coverage. Recover the exact graph-ingestion source or a verified merchant-ID crosswalk. Blank-code receipts also remain unresolved; no name-based relabeling.",
               "score_or_threshold_changes": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        temp.replace(args.out)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"path": args.out.as_posix(), "on": arms["on"], "off": arms["off"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
