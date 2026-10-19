"""P012 런이 남겨야 할 것 세 가지를 **검사로** 증명한다. 하나라도 어긋나면 실패한다.

    python scripts/report/verify_p012_preservation.py \
        --base /data/multipolicy_v53_20260928/p012m --days 7 \
        --json-out /data/multipolicy_v53_20260928/p012m/preservation_check.json

## 무엇을 확인하는가 (두 팔 각각)

1. **그래프 보존** — `graph_backup/` 에 neo4j·system 덤프와 설정이 있고,
   SHA256SUMS 의 모든 파일 해시가 지금 다시 계산한 값과 같다.
2. **메모리 보존 · 1대1 인터뷰 가능** — dossier 에 명부 **전원**이 한 줄씩 있고,
   상태가 명부 × 일수만큼 있으며(빠진 날 0), 모든 사람이 프로필·상태·계획을 갖는다.
3. **결제원장 보존** — 업종 원장과 캐시백 원장이 명부 × 일수 행을 갖고, 사람과 날이
   명부·창과 정확히 같다. manifest 가 있다.

덧붙여 날마다 내린 증분(보험)의 수도 센다 — 이것은 실패 사유가 아니라 기록이다.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(p: Path):
    with io.open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_graph(arm_dir: Path) -> tuple[bool, list[str]]:
    gb = arm_dir / "graph_backup"
    notes = []
    sums = gb / "SHA256SUMS"
    if not sums.is_file():
        return False, ["SHA256SUMS 없음"]
    want = {}
    for line in io.open(sums, encoding="utf-8"):
        parts = line.split()
        if len(parts) >= 2:
            want[parts[-1].lstrip("*")] = parts[0]
    for name in ("neo4j.dump", "system.dump"):
        if name not in want:
            notes.append("%s 가 체크섬 목록에 없다" % name)
    ok = not notes
    for name, h in want.items():
        f = gb / name
        if not f.is_file():
            ok = False
            notes.append("%s 파일 없음" % name)
            continue
        got = sha256_file(f)
        if got != h:
            ok = False
            notes.append("%s 해시 불일치" % name)
        elif name == "neo4j.dump":
            notes.append("neo4j.dump %.0fMB 해시 일치" % (f.stat().st_size / 1e6))
    return ok, notes


def check_dossier(arm_dir: Path, roster: list[str], days: list[str]) -> tuple[bool, list[str]]:
    p = arm_dir / "dossier.jsonl"
    m = arm_dir / "dossier.jsonl.manifest.json"
    if not p.is_file() or not m.is_file():
        return False, ["dossier 또는 manifest 없음"]
    man = json.loads(m.read_text(encoding="utf-8"))
    notes, ok = [], True
    seen, states, mem, plans, bad = set(), 0, 0, 0, []
    for r in jsonl(p):
        a = r["aid"]
        seen.add(a)
        st = r.get("states") or []
        states += len(st)
        mem += len(r.get("memories") or [])
        plans += sum(len(x.get("items") or []) for x in r.get("plans") or [])
        got_days = {x["day"] for x in st}
        if not r.get("profile") or not r.get("plans") or got_days != set(days):
            bad.append(a)
    miss = sorted(set(roster) - seen)
    extra = sorted(seen - set(roster))
    if miss:
        ok = False; notes.append("dossier 에 없는 명부 인원 %d" % len(miss))
    if extra:
        ok = False; notes.append("명부 밖 인원 %d" % len(extra))
    if states != len(roster) * len(days):
        ok = False; notes.append("상태 %d ≠ 명부x일수 %d" % (states, len(roster) * len(days)))
    if bad:
        ok = False; notes.append("인터뷰 불가(프로필·계획·날짜 결손) %d명 예 %s" % (len(bad), bad[:2]))
    if man.get("state_day_gaps"):
        ok = False; notes.append("manifest 에 빠진 날 %d명" % len(man["state_day_gaps"]))
    if hashlib.sha256(p.read_bytes()).hexdigest() != man.get("sha256"):
        ok = False; notes.append("dossier 파일 해시가 manifest 와 다르다")
    notes.insert(0, "명부 %d명 전원 · 상태 %d · 기억 %d · 계획항목 %d"
                 % (len(seen), states, mem, plans))
    return ok, notes


def check_ledger(p: Path, roster: list[str], days: list[str], label: str) -> tuple[bool, list[str]]:
    if not p.is_file():
        return False, ["%s 없음" % label]
    if not Path(str(p) + ".manifest.json").is_file():
        return False, ["%s manifest 없음" % label]
    rows, pairs = 0, set()
    for r in jsonl(p):
        rows += 1
        pairs.add((r["aid"], r["day"]))
    want = {(a, d) for a in roster for d in days}
    ok, notes = True, ["%s %d행" % (label, rows)]
    if rows != len(want):
        ok = False; notes.append("행 수 %d ≠ 명부x일수 %d" % (rows, len(want)))
    if pairs != want:
        ok = False
        notes.append("사람·날 조합 불일치 — 빠짐 %d · 남음 %d"
                     % (len(want - pairs), len(pairs - want)))
    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--days", type=int, required=True)
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    base = Path(a.base)
    roster = json.loads((base / "roster.json").read_text(encoding="utf-8"))
    d0 = date.fromisoformat(a.start)
    days = [(d0 + timedelta(days=i)).isoformat() for i in range(a.days)]

    print("# P012 보존 검사 — %s" % base)
    print()
    print("  명부 %d명 · 창 %s ~ %s (%d일)" % (len(roster), days[0], days[-1], len(days)))
    print()
    out, all_ok = {}, True
    for arm in ("on", "off"):
        ad = base / arm
        res = {
            "그래프": check_graph(ad),
            "메모리·인터뷰": check_dossier(ad, roster, days),
            "업종 원장": check_ledger(ad / "sector.ledger.jsonl", roster, days, "업종 원장"),
            "캐시백 원장": check_ledger(ad / "cashback.ledger.jsonl", roster, days, "캐시백 원장"),
        }
        inc = sorted((ad / "daily").glob("dossier_*.jsonl")) if (ad / "daily").is_dir() else []
        print("## %s 팔" % arm)
        for k, (ok, notes) in res.items():
            all_ok &= ok
            print("  %-12s %s  %s" % (k, "통과" if ok else "**실패**", " · ".join(notes)))
        print("  %-12s %s" % ("일별 증분", "%d/%d일 (보험 — 실패 사유 아님)" % (len(inc), len(days))))
        print()
        out[arm] = {k: {"ok": ok, "notes": notes} for k, (ok, notes) in res.items()}
        out[arm]["daily_increments"] = len(inc)
    print("## 결론 — %s" % ("**세 가지 모두 보존됐다**" if all_ok else "**보존 실패 — 검증지표 비교를 믿을 수 없다**"))
    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(
            json.dumps({"ok": all_ok, "roster": len(roster), "days": days, "arms": out},
                       ensure_ascii=False, indent=1))
        print("→ %s" % a.json_out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
