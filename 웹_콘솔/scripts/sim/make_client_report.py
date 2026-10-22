"""sim_standalone 대시보드를 '클라이언트 친화' 보고서로 재작성.

개발자 중심(모델명·내부 메커니즘·ID 검색·AI틱 네온 UI) → 클라이언트가 한눈에
'무엇을/과정/결과'를 이해하도록 카피·라벨·UI를 정제한다. 데이터·지도 로직은 그대로.

- 거대한 내장 JSON/JS 는 건드리지 않고, head(CSS override)·body(카피/라벨)만 교체
- getElementById 가 참조하는 ID 는 전부 보존(숨김 포함) → JS 무손상
- 템플릿이 둘이라 --profile 로 구분: dasol(사이드바형) / full(패널형 index.html)

사용:
  python scripts/sim/make_client_report.py --in dasol.html --profile dasol
  python scripts/sim/make_client_report.py --in full.html  --profile full \\
      --after output/sim/visualization/sim_standalone_full_client.html --zip
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VIZ_DIR = Path(__file__).resolve().parents[2] / "output" / "sim" / "visualization"


# ═══════════════════════════════════════════════════════════════════════════
# 공통: 차분한 톤 + 카드 최소화 CSS (각 프로필이 자기 selector에 맞춰 확장)
# ═══════════════════════════════════════════════════════════════════════════
_BASE_TONE = """
  --c-bg:#0b0e13; --c-panel:#0f141b; --c-border:rgba(255,255,255,0.10);
  --c-accent:#9fb4cc; --c-text:#e6e9ee; --c-dim:#aab3c0; --c-muted:#828c99; --c-bright:#f4f6f9;
  --c-font:'Pretendard','Apple SD Gothic Neo','Segoe UI',system-ui,-apple-system,sans-serif;
"""


# ───────────────────────────────────────────────────────────────────────────
# PROFILE: dasol (사이드바형 대시보드)
# ───────────────────────────────────────────────────────────────────────────
_DASOL_INTRO = """    <!-- Client intro -->
    <div class="intro">
      <div class="lead">정책이 동네 상권 소비를 어떻게 바꿀까?</div>
      <p>종로구·중구에 <b>외식·카페·디저트 소비 지원 정책</b>을 시행했을 때 시민들의 소비가
      어떻게 달라지는지, 가상 시뮬레이션으로 미리 예측한 결과입니다.</p>
      <ol>
        <li>서울 25개 자치구의 다양한 시민을 가상으로 재현해 이틀간 생활하게 합니다.</li>
        <li>둘째 날 정오, 지원 정책이 발효됩니다.</li>
        <li>시민들의 방문·소비 변화가 지도와 그래프에 실시간으로 나타납니다.</li>
      </ol>
      <div class="howto">
        <div>위 그래프 — 정책 대상 업종·지역의 소비액 추이 (정책 시행 시점 표시)</div>
        <div>지도의 점 — 시민 한 명. 클릭하면 그 사람의 하루를 볼 수 있습니다.</div>
        <div>아래 ▶ 재생 — 시간대별로 도시가 어떻게 움직이는지 확인</div>
      </div>
    </div>
    <!-- KPI -->"""

_DASOL_CSS = """
/* ═══════════ Client-friendly override (정제: 카드 최소화 · 차분한 톤) ═══════════ */
:root{""" + _BASE_TONE + """
  --bg-base:var(--c-bg); --bg-panel:var(--c-panel); --bg-card:transparent;
  --border:var(--c-border); --accent:var(--c-accent); --accent-glow:transparent;
  --accent2:var(--c-accent); --accent2-glow:transparent;
  --text:var(--c-text); --text-dim:var(--c-dim); --text-muted:var(--c-muted); --text-bright:var(--c-bright);
  --radius:10px; --font:var(--c-font); --mono:var(--c-font);
}
.panel{ backdrop-filter:none !important; -webkit-backdrop-filter:none !important;
  background:var(--c-panel) !important; box-shadow:0 6px 22px rgba(0,0,0,0.32) !important;
  border-color:var(--c-border) !important; }
.sb-header h1{ font-size:16px; font-weight:700; letter-spacing:-.01em; }
.sb-header .subtitle{ font-size:11px; color:var(--c-dim); margin-top:3px; line-height:1.45; }
.sb-header .frame-now{ color:var(--c-dim); }
.sb-section-title{ text-transform:none !important; letter-spacing:0 !important;
  color:var(--c-muted) !important; font-size:11px !important; font-weight:700 !important; }
.kpi-grid{ gap:0; border-top:1px solid var(--c-border); }
.kpi{ background:transparent !important; border:none !important;
  border-bottom:1px solid var(--c-border) !important; border-radius:0 !important; padding:9px 2px; }
.kpi:hover{ background:transparent !important; }
.kpi-val{ font-family:var(--c-font) !important; font-size:18px; font-weight:700; letter-spacing:-.01em; }
.kpi-lbl{ font-size:10px; color:var(--c-muted); }
.macro-card{ background:var(--c-panel) !important; border:1px solid var(--c-border) !important;
  box-shadow:none !important; border-radius:10px !important; }
.macro-title{ color:var(--c-dim) !important; font-weight:600 !important; letter-spacing:0 !important; }
.info-card{ background:transparent !important; border:none !important;
  border-top:1px solid var(--c-border) !important; border-radius:0 !important; }
input:focus, select:focus, button:focus{ box-shadow:none !important; }
.search-row{ display:none !important; }
.intro{ border:1px solid var(--c-border); border-radius:10px; padding:14px 14px 12px;
  background:rgba(255,255,255,0.015); }
.intro .lead{ color:var(--c-bright); font-weight:700; font-size:12.5px; margin-bottom:7px; letter-spacing:-.01em; }
.intro p{ font-size:11.5px; color:var(--c-dim); line-height:1.62; }
.intro ol{ margin:9px 0 0; padding-left:17px; }
.intro ol li{ font-size:11px; color:var(--c-dim); line-height:1.5; margin:3px 0; }
.intro .howto{ margin-top:11px; padding-top:10px; border-top:1px solid var(--c-border); }
.intro .howto div{ font-size:10.5px; color:var(--c-muted); line-height:1.5; margin:3px 0; padding-left:2px; }
</style>"""

_DASOL = [
    ("<title>서울 상권정책 시뮬레이션 — 인터랙티브 대시보드</title>",
     "<title>서울 상권정책 시뮬레이션 — 정책 효과 예측 리포트</title>", True),
    ("<h1>🏙️ Seoul Policy Sim</h1>", "<h1>서울 상권정책 시뮬레이션</h1>", True),
    ('<div class="subtitle">풀런 2일 · Qwen3-14B · 25개 자치구</div>',
     '<div class="subtitle">정책이 동네 상권 소비에 미치는 영향을 미리 예측합니다</div>', True),
    ("    <!-- KPI -->", _DASOL_INTRO, True),
    (">활동 에이전트<", ">지금 활동 중<", True),
    (">총 인구<", ">가상 시민<", True),
    (">방문 기억<", ">누적 방문<", True),
    (">약속<", ">만남 약속<", True),
    (">들은 소문<", ">입소문<", True),
    (">📊 실시간 통계<", ">현황<", True),
    (">⚙️ 필터<", ">보기 설정<", True),
    (">🎨 범례<", ">색상 안내<", True),
    (">🔍 에이전트 검색<", ">시민 살펴보기<", True),
    ('<input type="text" id="search-input" placeholder="ID 입력 (예: agent_0042)"/>',
     '<input type="text" id="search-input" placeholder="" />', True),
    ('data-tab="schedule">스케줄<', 'data-tab="schedule">오늘 동선<', True),
    ('data-tab="memory">기억<', 'data-tab="memory">자주 가는 곳<', True),
    ('data-tab="social">소셜<', 'data-tab="social">관계<', True),
    ('data-tab="state">상태<', 'data-tab="state">컨디션<', True),
    (">⭐ 단골 POI<", ">자주 찾는 장소<", False),
    ("🧠 방문 기억 <span", "방문 기록 <span", False),
    (">🤝 약속<", ">약속<", False),
    (">💬 소문<", ">들은 이야기<", False),
    ("📊 카테고리별 소비 포트폴리오", "소비 구성", False),
    (">📈 정책 대상 카테고리 누적 소비액<", ">정책 대상 업종 소비액 추이 — 식사·카페·디저트<", False),
    (">📍 정책 타겟 자치구 누적 소비액<", ">정책 대상 지역 소비액 추이 — 종로구·중구<", False),
    ("</style>", _DASOL_CSS, True),
]


# ───────────────────────────────────────────────────────────────────────────
# PROFILE: full (패널형 index.html 템플릿 — P008 강남 보행친화거리)
# ───────────────────────────────────────────────────────────────────────────
_FULL_INTRO = """  <h3>서울 상권정책 시뮬레이션</h3>
  <div class="intro-sub">강남역–역삼역 일대에 <b>보행친화거리</b>(보도 확장·조명 개선·휴식 벤치)를
  조성했을 때, 시민들의 외출·소비가 어떻게 달라지는지 가상으로 예측한 결과입니다.</div>
  <div class="intro-steps">
    <span>① 서울 25개구 시민 약 1.5만 명을 5일간 생활하게 합니다</span>
    <span>② 셋째 날, 강남역–역삼역 보행친화거리가 조성됩니다</span>
    <span>③ 시민들의 보행·방문·소비 변화가 지도에 나타납니다</span>
  </div>
  <div class="intro-tip">지도의 점은 시민 한 명입니다. 점을 클릭하면 그 사람의 하루를, 아래 ▶로 시간대별 도시의 움직임을 볼 수 있어요.</div>"""

_FULL_CSS = """
/* ═══════════ Client-friendly override (정제: 카드 최소화 · 차분한 톤) ═══════════ */
:root{""" + _BASE_TONE + """}
body{ background:var(--c-bg) !important; font-family:var(--c-font) !important; }
.panel{ background:var(--c-panel) !important; backdrop-filter:none !important;
  -webkit-backdrop-filter:none !important; border:1px solid var(--c-border) !important;
  box-shadow:0 6px 22px rgba(0,0,0,0.32) !important; }
.panel h3{ color:var(--c-bright) !important; font-weight:700 !important; font-size:14px !important; }
.panel h4{ color:var(--c-dim) !important; font-weight:600 !important; }
#info{ max-width:320px; }
#info .row .label{ color:var(--c-muted) !important; }
#info .val{ color:var(--c-bright) !important; }
#ctrl button{ background:var(--c-accent) !important; color:var(--c-bg) !important; }
#ctrl .frame-label{ color:var(--c-dim) !important; }
select, .ctrl-group select{ color:var(--c-text) !important; }
.intro-sub{ font-size:11px; color:var(--c-dim); line-height:1.6; margin:6px 0 8px; }
.intro-steps{ display:flex; flex-direction:column; gap:3px; padding:8px 0;
  border-top:1px solid var(--c-border); border-bottom:1px solid var(--c-border); margin-bottom:8px; }
.intro-steps span{ font-size:10.5px; color:var(--c-dim); line-height:1.45; }
.intro-tip{ font-size:10px; color:var(--c-muted); line-height:1.5; margin-bottom:4px; }
</style>"""

_FULL = [
    ("<title>서울 상권정책 시뮬 — 풀런 5일 (Qwen3-14B + 강남역–역삼역 보행친화거리 P008) · 25개 자치구 표본</title>",
     "<title>서울 상권정책 시뮬레이션 — 강남 보행친화거리 효과 예측</title>", True),
    ('<h3>🏙️ Seoul Sim — 풀런 5일 (Qwen3-14B)</h3>', _FULL_INTRO, True),
    ('<span class="label">활동 agent</span>', '<span class="label">지금 활동 중</span>', True),
    ('<span class="label">표본 자치구</span><span class="val">25개구 균등</span>',
     '<span class="label">대상 지역</span><span class="val">서울 25개구</span>', True),
    ('<span class="label">총 agent</span>', '<span class="label">가상 시민</span>', True),
    ('<span class="label">총 visited memory</span>', '<span class="label">누적 방문</span>', True),
    ('<span class="label">총 약속</span>', '<span class="label">만남 약속</span>', True),
    ('<span class="label">총 들은 소문</span>', '<span class="label">입소문</span>', True),
    ('<h3>🎨 범례 / 필터</h3>', '<h3>보기 설정</h3>', True),
    ('>약속 있는 agent<', '>약속 있는 시민<', False),
    ('<h4>🗓️ 오늘 스케줄</h4>', '<h4>오늘 동선</h4>', False),
    ('🧠 기억 메모리 (Memory Stream) <span', '방문 기록 <span', False),
    ('<h4>💬 들은 소문 (rumor)</h4>', '<h4>들은 이야기</h4>', False),
    ('<h4>🤝 잡은 약속 (appointment)</h4>', '<h4>약속</h4>', False),
    ('<h4>⭐ 단골 POI (KNOWS_POI Top)</h4>', '<h4>자주 찾는 장소</h4>', False),
    ('<h4>📊 어제 State</h4>', '<h4>어제 컨디션</h4>', False),
    ("</style>", _FULL_CSS, True),
]

PROFILES = {"dasol": _DASOL, "full": _FULL}


def detect_profile(html: str) -> str:
    if "Seoul Policy Sim" in html:
        return "dasol"
    if "Seoul Sim — 풀런" in html or 'id="info"' in html:
        return "full"
    return "dasol"


def transform(html: str, replacements) -> str:
    for old, new, required in replacements:
        cnt = html.count(old)
        if cnt == 0:
            print(f"  [{'MISS' if required else 'skip'}] 못 찾음: {old[:48]!r}",
                  file=sys.stderr if required else sys.stdout)
            continue
        html = html.replace(old, new, 1 if old == "</style>" else cnt)
        shown = new[:40].replace("\n", " ")
        print(f"  [ok x{cnt if old != '</style>' else 1}] {old[:40]!r} → {shown!r}")
    return html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--profile", choices=["auto", "dasol", "full"], default="auto")
    ap.add_argument("--before", type=Path, default=None)
    ap.add_argument("--after", type=Path, default=None)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    html = args.in_path.read_text(encoding="utf-8")
    profile = detect_profile(html) if args.profile == "auto" else args.profile
    print(f"[read] {args.in_path}  ({len(html)/1024/1024:.1f} MB) · profile={profile}")

    stem = args.in_path.stem
    before = args.before or (VIZ_DIR / f"{stem}_before.html")
    after = args.after or (VIZ_DIR / f"{stem}_after.html")

    before.write_text(html, encoding="utf-8")
    print(f"[before] {before}  ({before.stat().st_size/1024/1024:.1f} MB)")

    out = transform(html, PROFILES[profile])
    after.write_text(out, encoding="utf-8")
    print(f"[after]  {after}  ({after.stat().st_size/1024/1024:.1f} MB)")

    if args.zip:
        zp = after.with_suffix(".zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(after, arcname=after.name)
        print(f"[after.zip] {zp}  ({zp.stat().st_size/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
