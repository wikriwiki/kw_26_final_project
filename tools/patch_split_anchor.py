"""2층 가르기를 **어느 저장소에든 같은 모양으로** 심는다 — 파일을 덮지 않는다.

    python tools/patch_split_anchor.py <repo_root> [--check]

## 왜 복사가 아니라 패치인가

서버에 저장소가 둘이고 **세대가 다르다.**

    /data/repo               거시(정답지) 런이 도는 곳 · consumption.py 1,114줄
    /data/validation_v3/repo 검증 런이 도는 곳 · 1,154줄 (로컬과 같음)

거시 쪽에 없는 40여 줄(affordability 보정·choice_shares 처리)이 검증 쪽에 있다.
**로컬 파일을 그대로 복사하면 그 차이를 지운다** — 그러면 비교가 2층 가르기의
효과가 아니라 40줄의 효과와 섞인다. 그래서 닻 문자열을 찾아 그 자리만 고친다.

정답지 비교는 기존 P012 읽기(result_r2_v5)와 맞대야 하므로 **거시 저장소에서**
돌려야 한다. 그 저장소를 최소로 건드리는 것이 이 스크립트의 목적이다.

## 심는 것 넷

    consumption.py      상수 묶음 + 회계 분기 (EXP_SPLIT_ANCHOR 로만 켜진다)
    score_policy.py     online_spend_paired 지표와 State 조회
    prompts/__init__.py v5offsite 등록 (레지스트리 모양이 달라 줄 구조로 찾는다)
    scoring_table.json  P012-2 의 눈금을 online_spend_paired 로

## 멱등이다

이미 심겨 있으면 아무것도 안 한다. `--check` 는 심지 않고 상태만 알린다.
닻이 하나라도 없으면 **아무것도 심지 않고 멈춘다** — 반쯤 심긴 저장소가
제일 나쁘다.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

B = json.loads(io.open(Path(__file__).with_name("_split_patch_blocks.json"),
                       encoding="utf-8").read())


def patch_consumption(root: Path, check: bool) -> str:
    p = root / "scripts/sim/consumption.py"
    s = io.open(p, encoding="utf-8").read()
    if "EXP_SPLIT_ANCHOR" in s:
        return "consumption.py      이미 심겨 있다"
    if B["CONST_ANCHOR"] not in s or B["BRANCH_OLD"] not in s:
        raise SystemExit(f"[실패] {p}: 닻을 못 찾았다 — 세대가 다르다")
    if check:
        return "consumption.py      닻 둘 다 있다 (심으면 됨)"
    s = s.replace(B["CONST_ANCHOR"], B["CONST_ANCHOR"] + B["CONST_ADD"], 1)
    s = s.replace(B["BRANCH_OLD"], B["BRANCH_NEW"], 1)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    return "consumption.py      심었다"


def patch_score(root: Path, check: bool) -> str:
    p = root / "scripts/sim/score_policy.py"
    s = io.open(p, encoding="utf-8").read()
    if "online_spend_paired" in s:
        return "score_policy.py     이미 심겨 있다"
    if B["SCORE_FETCH_ANCHOR"] not in s or B["SCORE_METRIC_OLD"] not in s:
        raise SystemExit(f"[실패] {p}: 닻을 못 찾았다 — 세대가 다르다")
    if check:
        return "score_policy.py     닻 둘 다 있다 (심으면 됨)"
    s = s.replace(B["SCORE_FETCH_ANCHOR"], B["SCORE_FETCH_ADD"] + B["SCORE_FETCH_ANCHOR"], 1)
    s = s.replace(B["SCORE_METRIC_OLD"], B["SCORE_METRIC_NEW"], 1)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    return "score_policy.py     심었다"


def patch_prompts(root: Path, check: bool) -> str:
    """레지스트리 모양이 저장소마다 다르므로 **줄 구조로** 찾는다."""
    p = root / "scripts/sim/prompts/__init__.py"
    s = io.open(p, encoding="utf-8").read()
    if "v5offsite" in s:
        return "prompts/__init__.py 이미 심겨 있다"
    lines = s.split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith("_VARIANTS")), None)
    if start is None:
        raise SystemExit(f"[실패] {p}: _VARIANTS 를 못 찾았다")
    close = next((i for i in range(start, len(lines)) if lines[i].strip() == "}"), None)
    last_imp = max((i for i, l in enumerate(lines[:start])
                    if l.startswith("from . import v")), default=None)
    if close is None or last_imp is None:
        raise SystemExit(f"[실패] {p}: 삽입 자리를 못 찾았다")
    if check:
        return "prompts/__init__.py 삽입 자리 있다 (심으면 됨)"
    lines.insert(close, '    "v5offsite": v5offsite,')
    lines.insert(last_imp + 1,
                 "from . import v5offsite  # noqa: E402  (질문 범위를 회계와 맞춘 판)")
    io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    return "prompts/__init__.py 심었다"


def patch_table(root: Path, check: bool) -> str:
    p = root / "data/experiments/scoring_table.json"
    if not p.exists():
        return "scoring_table.json  없다 — 건너뜀"
    d = json.loads(io.open(p, encoding="utf-8").read())
    ind = [i for i in (d.get("P012") or {}).get("indicators", []) if i.get("id") == "P012-2"]
    if not ind:
        return "scoring_table.json  P012-2 없다 — 건너뜀"
    if ind[0].get("metric") == "online_spend_paired":
        return "scoring_table.json  이미 옮겼다"
    if check:
        return "scoring_table.json  P012-2 = %s (옮기면 됨)" % ind[0].get("metric")
    ind[0]["metric"] = "online_spend_paired"
    ind[0]["ruler_note"] = (
        "제외업종(백화점·대형마트·온라인)은 소비 모델이 POI 원장에 넣기 전에 "
        "st.online_spent 로 걷어낸다. 원장의 비적격으로 재면 새어 나온 잔여물"
        "(전체 지출의 1.35%, 84행)만 잡힌다 — 2026-09-24 눈금 이동")
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(d, ensure_ascii=False, indent=1))
    return "scoring_table.json  P012-2 눈금 옮겼다"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    if not (root / "scripts/sim/consumption.py").exists():
        raise SystemExit(f"[실패] 저장소가 아니다: {root}")
    fns = (patch_consumption, patch_score, patch_prompts, patch_table)
    # 닻 확인을 **먼저 전부** 돌린다 — 반쯤 심긴 저장소를 만들지 않기 위해서.
    if not a.check:
        for fn in fns:
            fn(root, True)
    for fn in fns:
        print(" ", fn(root, a.check))
    return 0


if __name__ == "__main__":
    sys.exit(main())
