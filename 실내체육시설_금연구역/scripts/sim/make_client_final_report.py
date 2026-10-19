"""FINAL_REPORT_5D_FULL.html(개발자 중심)을 클라이언트 친화 보고서로 재저술.

원본의 7개 그림(base64)을 추출해 그대로 임베드하되, 서사·정보 흐름·UI는 처음
읽는 클라이언트가 '무엇을→왜→어떻게→결과→유의'를 순서대로 이해하도록 새로 쓴다.

- 전(before): 원본 그대로 저장
- 후(after):  새 보고서 (라이트 톤 · 카드 최소화 · 이모지 배제 · 자체완결)

사용:
  python scripts/sim/make_client_final_report.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPORT_DIR = Path(__file__).resolve().parents[2] / "output" / "sim" / "report"

# ── 새 보고서 본문 (정보 흐름 설계) ──────────────────────────────────────────
# 그림 자리표시자: {IMG0}=1인당매출 {IMG1}=변화율 {IMG2}=DID {IMG3}=spillover
#                  {IMG4}=trigger {IMG5}=단골 {IMG6}=만족도
PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>강남역–역삼역 보행친화거리 — 상권 효과 예측 리포트</title>
<style>
  :root{
    --ink:#1f2430; --ink-soft:#4b5563; --muted:#7b8493; --line:#e6e8ec;
    --accent:#1f4e79; --accent-soft:#eef3f9; --good:#1f7a4d; --bg:#ffffff;
    --font:'Pretendard','Apple SD Gothic Neo','Segoe UI',system-ui,-apple-system,sans-serif;
  }
  *{ box-sizing:border-box; }
  html,body{ margin:0; padding:0; background:#f6f7f9; color:var(--ink); font-family:var(--font);
    line-height:1.75; -webkit-font-smoothing:antialiased; }
  .wrap{ max-width:860px; margin:0 auto; background:var(--bg); padding:64px 56px 80px;
    box-shadow:0 1px 3px rgba(0,0,0,0.04); }
  .eyebrow{ font-size:12px; font-weight:600; letter-spacing:.12em; color:var(--accent);
    text-transform:uppercase; margin-bottom:14px; }
  h1{ font-size:30px; line-height:1.3; font-weight:750; letter-spacing:-.02em; margin:0 0 12px; color:#15191f; }
  .dek{ font-size:16px; color:var(--ink-soft); margin:0 0 20px; }
  .meta{ font-size:13px; color:var(--muted); padding:14px 0; border-top:1px solid var(--line);
    border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:6px 18px; }
  .meta b{ color:var(--ink); font-weight:600; }

  .summary{ background:var(--accent-soft); border-left:3px solid var(--accent);
    padding:20px 22px; margin:30px 0; border-radius:0 6px 6px 0; }
  .summary .label{ font-size:12px; font-weight:700; color:var(--accent); letter-spacing:.04em; margin-bottom:6px; }
  .summary p{ margin:0; font-size:16px; color:var(--ink); line-height:1.65; }
  .stats{ display:flex; flex-wrap:wrap; gap:30px; margin-top:18px; }
  .stat .n{ font-size:24px; font-weight:750; color:var(--accent); letter-spacing:-.01em; }
  .stat .k{ font-size:12px; color:var(--muted); margin-top:1px; }

  h2{ font-size:20px; font-weight:700; letter-spacing:-.01em; margin:48px 0 4px; padding-top:26px;
    border-top:1px solid var(--line); color:#15191f; }
  h2 .step{ color:var(--muted); font-weight:600; font-size:15px; margin-right:8px; }
  h3{ font-size:15px; font-weight:700; color:var(--ink); margin:26px 0 6px; }
  p{ margin:12px 0; color:var(--ink-soft); font-size:15.5px; }
  p strong, li strong{ color:var(--ink); font-weight:650; }
  ul,ol{ margin:12px 0; padding-left:22px; color:var(--ink-soft); font-size:15.5px; }
  li{ margin:6px 0; }

  figure{ margin:24px 0; }
  figure img{ width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }
  figcaption{ font-size:13px; color:var(--muted); margin-top:8px; text-align:center; }
  .two{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  @media(max-width:680px){ .two{ grid-template-columns:1fr; } .wrap{ padding:40px 22px 60px; } }

  table{ width:100%; border-collapse:collapse; margin:18px 0; font-size:14.5px; }
  th,td{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); }
  th{ color:var(--muted); font-weight:600; font-size:13px; }
  td.num{ text-align:right; font-variant-numeric:tabular-nums; }
  .pos{ color:var(--good); font-weight:700; }

  .note{ background:#fbfbfc; border:1px solid var(--line); border-radius:6px; padding:18px 20px; margin:24px 0; }
  .note h3{ margin-top:0; }
  .note p, .note li{ font-size:14px; color:var(--ink-soft); }

  .footer{ margin-top:48px; padding-top:22px; border-top:1px solid var(--line); font-size:13.5px; color:var(--muted); }
  .footer b{ color:var(--ink); }
</style>
</head>
<body>
<div class="wrap">

  <div class="eyebrow">정책 시뮬레이션 리포트</div>
  <h1>강남역–역삼역 보행친화거리,<br/>동네 상권에 도움이 될까?</h1>
  <p class="dek">보행 환경을 개선하면 사람들이 더 나오고 더 소비할지를,
  가상의 서울 시민으로 미리 실험해 본 결과입니다.</p>
  <div class="meta">
    <span>대상 <b>서울 25개 자치구</b></span>
    <span>가상 시민 <b>14,881명</b></span>
    <span>기간 <b>2026.5.18–5.22 (5일)</b></span>
    <span>정책 시행 <b>5월 20일</b></span>
  </div>

  <div class="summary">
    <div class="label">한눈에</div>
    <p>분석 기간 동안 서울 전체 소비가 가라앉는 흐름이었지만, <strong>정책 대상 지역(강남)은
    비대상 지역보다 소비가 7.2%p 덜 줄었습니다.</strong> 걷기 좋은 거리가 상권 소비를
    상대적으로 떠받친 것으로 해석됩니다.</p>
    <div class="stats">
      <div class="stat"><div class="n">14,881명</div><div class="k">하루를 살아간 가상 시민</div></div>
      <div class="stat"><div class="n">64.6만 건</div><div class="k">분석된 방문·활동</div></div>
      <div class="stat"><div class="n">+7.2%p</div><div class="k">정책의 순효과(상대 방어)</div></div>
    </div>
  </div>

  <h2><span class="step">01</span>무엇을 실험했나</h2>
  <p>서울시는 강남역에서 역삼역으로 이어지는 테헤란로 이면 골목을 <strong>보행친화거리</strong>로
  새단장했습니다. 보도를 넓히고 가로 조명을 밝히고 곳곳에 휴식 벤치를 두는 사업입니다.
  할인이나 쿠폰 같은 <strong>금전 혜택이 아니라, ‘걷기 좋은 환경’을 만드는 정책</strong>입니다.</p>
  <p>그래서 던진 질문은 단순합니다 — <strong>거리가 걷기 좋아지면, 사람들이 더 자주 나오고
  더 소비할까?</strong> 이걸 실제 도시에 적용하기 전에, 가상으로 먼저 확인해 본 것입니다.</p>

  <h2><span class="step">02</span>어떻게 알아봤나</h2>
  <p>실제 소비·이동 통계로 빚어낸 <strong>가상 시민 약 1만 5천 명</strong>이, 매일 스스로 하루
  계획을 세우고 도시를 돌아다닙니다. 점심을 먹고, 카페에 들르고, 쇼핑을 하고, 약속을 잡는
  — 진짜 사람처럼 행동합니다.</p>
  <ul>
    <li>5일 중 <strong>셋째 날(5월 20일)</strong>에 보행친화거리가 조성됩니다.</li>
    <li><strong>시행 전 2일</strong>과 <strong>시행 후 3일</strong>의 소비를 비교합니다.</li>
    <li>비교할 때는 정책 지역의 변화에서 <strong>같은 기간 서울 전체의 변화를 빼서</strong>,
    계절·경기 같은 공통 요인을 걷어내고 <strong>순수한 정책 효과만</strong> 남깁니다.</li>
  </ul>

  <h2><span class="step">03</span>결과 ① — 정책이 소비를 떠받쳤다</h2>
  <p>분석 기간 동안 서울 전체의 소비는 줄어드는 흐름이었습니다. 그런데 <strong>정책 지역(강남)은
  훨씬 덜 줄었습니다.</strong> 아래 두 그림은 시행 전후의 1인당 소비와 그 변화율을 보여줍니다.</p>
  <div class="two">
    <figure><img src="{IMG0}" alt="1인당 소비 추이"/><figcaption>1인당 소비 추이 (시행 전 → 후)</figcaption></figure>
    <figure><img src="{IMG1}" alt="소비 변화율"/><figcaption>지역별 소비 변화율</figcaption></figure>
  </div>
  <table>
    <tr><th>지역</th><th class="num">시행 전 변화 흐름</th><th class="num">정책 시행 후</th></tr>
    <tr><td>강남 (정책 대상)</td><td class="num">기준</td><td class="num">−1.8%</td></tr>
    <tr><td>그 외 지역 (비교군)</td><td class="num">기준</td><td class="num">−9.0%</td></tr>
    <tr><td><strong>차이 = 정책 순효과</strong></td><td class="num"></td><td class="num pos">+7.2%p</td></tr>
  </table>
  <p>두 지역의 차이 <strong>+7.2%p</strong>가 곧 정책의 순효과입니다. 전반적으로 소비가 식는
  시기였음에도, 걷기 좋아진 거리가 강남 상권의 소비를 상대적으로 지켜낸 셈입니다.</p>
  <figure><img src="{IMG2}" alt="정책 순효과(이중차분)"/><figcaption>정책 순효과 — 두 지역의 변화 차이(이중차분)</figcaption></figure>

  <h2><span class="step">04</span>결과 ② — 주변 지역으로의 파급</h2>
  <p>효과가 강남 안에만 머물렀을까요, 아니면 옆 동네로도 번졌을까요? 강남과 맞닿은
  <strong>서초·송파</strong>를, 멀리 떨어진 지역과 비교해 살펴봤습니다. 인접 지역과의 소비
  격차가 좁혀졌다면 효과가 주변으로 퍼진(spillover) 신호로 볼 수 있습니다.</p>
  <figure><img src="{IMG3}" alt="주변 지역 파급효과"/><figcaption>강남 vs 인접·원거리 자치구 소비 격차 변화</figcaption></figure>

  <h2><span class="step">05</span>결과 ③ — 시민은 어떻게 움직였나</h2>
  <p>사람들이 굳이 ‘정책 때문에’ 나온 건 아닙니다. 대부분의 외출은 <strong>평소의 취향과
  생활 리듬</strong>에서 비롯됐습니다. 다만 걷기 좋아진 환경이 그 일상의 배경으로 작용했습니다.</p>
  <div class="two">
    <figure><img src="{IMG4}" alt="외출 동기"/><figcaption>사람들이 외출한 이유</figcaption></figure>
    <figure><img src="{IMG5}" alt="단골 형성"/><figcaption>단골이 만들어지는 과정 (신규 → 재방문 → 단골)</figcaption></figure>
  </div>
  <p>주목할 점은 단 5일 만에도 <strong>‘단골 가게’가 생겨났다는 것</strong>입니다. 한 번 들른 곳을
  다시 찾고, 그중 일부가 단골이 되는 자연스러운 흐름이 관찰됐습니다. 아래는 어떤 이유로 나온
  외출이 더 만족스러웠는지를 보여줍니다.</p>
  <figure><img src="{IMG6}" alt="외출 동기별 만족도"/><figcaption>외출 이유별 평균 만족도</figcaption></figure>

  <div class="note">
    <h3>이 리포트를 읽으실 때</h3>
    <ul>
      <li>실제 측정값이 아니라 <strong>가상 시민으로 미리 돌려본 ‘예측’</strong>입니다.
      절대 금액보다 <strong>방향성과 지역 간 상대 비교</strong>로 봐주세요.</li>
      <li><strong>5일간의 짧은 실험</strong>입니다. 장기적인 효과는 별도의 검증이 필요합니다.</li>
    </ul>
  </div>

  <div class="footer">
    <p><b>함께 보기 —</b> 같은 폴더의 인터랙티브 지도 대시보드에서, 가상 시민 한 명 한 명이
    하루를 어떻게 보내는지 직접 따라가 볼 수 있습니다.</p>
  </div>

</div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=REPORT_DIR / "FINAL_REPORT_5D_FULL.html")
    ap.add_argument("--before", type=Path, default=REPORT_DIR / "FINAL_REPORT_5D_FULL_before.html")
    ap.add_argument("--after", type=Path, default=REPORT_DIR / "FINAL_REPORT_5D_FULL_client.html")
    args = ap.parse_args()

    src = args.in_path.read_text(encoding="utf-8")
    imgs = re.findall(r"data:image/png;base64,[A-Za-z0-9+/=]+", src)
    print(f"[read] {args.in_path}  · 그림 {len(imgs)}개 추출")
    if len(imgs) < 7:
        print(f"[WARN] 그림이 7개 미만({len(imgs)}) — 빈 자리는 placeholder 유지", file=sys.stderr)

    # before = 원본 보존
    args.before.write_text(src, encoding="utf-8")
    print(f"[before] {args.before}  ({args.before.stat().st_size/1024:.0f} KB)")

    out = PAGE
    for i in range(7):
        uri = imgs[i] if i < len(imgs) else ""
        out = out.replace("{IMG%d}" % i, uri)
    args.after.write_text(out, encoding="utf-8")
    print(f"[after]  {args.after}  ({args.after.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
