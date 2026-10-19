# -*- coding: utf-8 -*-
"""Does the on arm's input actually differ from the off arm's, cell by cell?

Every indicator is an on/off contrast. If a pair's input were identical the contrast
would be pure resampling noise by construction, and no sample size would fix it. This
is cheap and worth running on any new cohort before reading a single number from it.

    python scripts/report/verify_arm_contrast.py --source action_source.json
"""
import json, difflib, collections
import argparse, sys
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
args = ap.parse_args()
src = json.load(open(args.source, encoding="utf-8"))
cells = {(c["aid"], c["case"], c["arm"]): c for c in src["cells"]}
aids = sorted({a for a, _, _ in cells})
diff_lines = collections.Counter(); samples = {}
for case in ("cashback", "grant", "local_voucher", "distancing"):
    n_same = 0
    for aid in aids:
        off = cells.get((aid, case, "off")); on = cells.get((aid, case, "on"))
        if not off or not on: continue
        a, b = off["user"], on["user"]
        if a == b:
            n_same += 1; continue
        dl = [l for l in difflib.unified_diff(a.split("\n"), b.split("\n"), lineterm="", n=0)
              if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
        diff_lines[case] += len(dl)
        if case not in samples and dl:
            samples[case] = [l[:95] for l in dl[:3]]
    print("%-16s 쌍 %d · 입력이 같은 쌍 %d · 평균 다른 줄 %.1f"
          % (case, len(aids), n_same, diff_lines[case] / max(1, len(aids) - n_same)))
print()
for case, sample in samples.items():
    print("===", case)
    for l in sample:
        print("   ", l)
identical = sum(1 for case in samples if False)
broken = [c for c in ("cashback", "grant", "local_voucher", "distancing") if diff_lines[c] == 0]
if broken:
    print()
    print("*** 대조가 없는 기전:", broken, "— 이 기전의 지표는 구조적으로 잡음이다 ***")
    sys.exit(1)
