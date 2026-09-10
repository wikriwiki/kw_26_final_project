# -*- coding: utf-8 -*-
"""프롬프트를 렌더링해 골든 스냅샷을 만들거나 대조한다.

    python tests/golden/render_prompts.py --write     # 기준선 생성
    python tests/golden/render_prompts.py --check     # 기준선과 대조

LLM·DB를 호출하지 않는다. 문자열 조립 경로만 실행한다.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.sim.dawn_context import DawnContext          # noqa: E402
from scripts.sim import stage1_intent as S1                # noqa: E402
import tests.golden.render_fixtures as FX                  # noqa: E402

OUT = Path(__file__).parent / "snapshots"


def render_all() -> dict[str, str]:
    """케이스별 프롬프트 전문을 dict로."""
    rendered: dict[str, str] = {}
    rendered["SYSTEM/stage1"] = S1.SYSTEM_PROMPT

    # 환경 ON — 코로나 채널이 실제로 렌더되는지 회귀 감시
    from datetime import date as _date
    from scripts.sim.environments import build_environment
    _p, _s = FX.PERSONAS[0], FX.STATES[0]
    for _d in (_date(2021, 8, 25), _date(2021, 9, 6), _date(2021, 9, 20)):
        _ctx = DawnContext(
            persona=_p, state=_s, memory=FX.MEMORY, appointment=FX.APPOINTMENT,
            policy=[], social=FX.SOCIAL, knows_poi_summary=FX.KNOWS_POI,
            zone_candidates=FX.ZONES,
            environment=build_environment("covid_2021", _d),
        )
        rendered[f"USER/stage1/covid_env/{_d.isoformat()}"] = S1._format_dawn_blocks(
            _ctx, _d, S1._day_type(_d))

    for name, persona, state, policy in FX.cases():
        for d in FX.DATES:
            ctx = DawnContext(
                persona=persona, state=state, memory=FX.MEMORY,
                appointment=FX.APPOINTMENT, policy=policy, social=FX.SOCIAL,
                knows_poi_summary=FX.KNOWS_POI, zone_candidates=FX.ZONES,
            )
            day_type = S1._day_type(d)
            key = f"USER/stage1/{name}/{d.isoformat()}"
            rendered[key] = S1._format_dawn_blocks(ctx, d, day_type)
    return rendered


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    rendered = render_all()
    manifest = {k: digest(v) for k, v in sorted(rendered.items())}

    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        for k, v in rendered.items():
            p = OUT / (k.replace("/", "__") + ".txt")
            p.write_text(v, encoding="utf-8", newline="\n")
        (OUT / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        print(f"기준선 생성: {len(rendered)}건 → {OUT}")
        for k in sorted(manifest):
            print(f"  {manifest[k][:12]}  {k}")
        return 0

    if a.check:
        mp = OUT / "MANIFEST.json"
        if not mp.exists():
            print("기준선 없음. 먼저 --write 실행", file=sys.stderr)
            return 2
        base = json.loads(mp.read_text(encoding="utf-8"))
        bad, missing, extra = [], [], []
        for k, h in base.items():
            if k not in manifest:
                missing.append(k)
            elif manifest[k] != h:
                bad.append(k)
        for k in manifest:
            if k not in base:
                extra.append(k)
        for k in bad:
            cur = rendered[k]
            old = (OUT / (k.replace("/", "__") + ".txt")).read_text(encoding="utf-8")
            print(f"\n=== 불일치: {k} ===")
            import difflib
            for line in list(difflib.unified_diff(
                    old.splitlines(), cur.splitlines(),
                    "기준선", "현재", lineterm=""))[:40]:
                print(line)
        if bad or missing:
            print(f"\n[실패] 불일치 {len(bad)} · 누락 {len(missing)} · 신규 {len(extra)}")
            return 1
        note = f" (신규 케이스 {len(extra)}건은 대조 대상 아님)" if extra else ""
        print(f"[통과] {len(base)}건 전부 바이트 동일{note}")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
