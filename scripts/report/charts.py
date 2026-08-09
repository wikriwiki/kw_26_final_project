"""보고서 v2 — 의존성 없는 인라인 SVG 차트.

matplotlib PNG 대신 SVG 를 쓰는 이유
------------------------------------
* 보고서가 **단일 HTML** 이어야 한다. SVG 는 base64 없이 그대로 들어간다.
* 색을 CSS 변수로 두면 라이트/다크 테마가 **같은 그림 하나로** 동작한다.
* 확대해도 눈금과 라벨이 깨지지 않는다. 표의 숫자와 그림이 같은 폰트를 쓴다.

모든 함수는 ``<figure>`` 없이 ``<svg>`` 문자열만 돌려준다. 캡션·출처는
``render_v2.py`` 가 붙인다. 값이 없으면 빈 문자열이 아니라 "데이터 없음"
플레이스홀더를 돌려준다 — 그림이 조용히 사라지지 않게 한다.
"""
from __future__ import annotations

import math
from html import escape
from typing import Any, Sequence

# 계열 색. 실제 값은 render_v2.py 의 CSS 토큰이 라이트/다크 각각으로 준다.
SERIES_VARS = [f"var(--k{i})" for i in range(1, 9)]
POS = "var(--k-pos)"
NEG = "var(--k-neg)"
GRID = "var(--k-grid)"
AXIS = "var(--k-axis)"
INK = "var(--k-ink)"
MUTED = "var(--k-muted)"
BAND = "var(--k-band)"


# --------------------------------------------------------------------------- #
# 숫자 포맷
# --------------------------------------------------------------------------- #


def krw(value: float | None, *, digits: int = 1) -> str:
    """원 단위 값을 사람이 읽는 축약형으로. 반올림 자리를 라벨에 숨기지 않는다."""
    if value is None:
        return "—"
    sign = "−" if value < 0 else ""
    v = abs(float(value))
    if v >= 1e8:
        return f"{sign}{v / 1e8:.{digits}f}억"
    if v >= 1e4:
        return f"{sign}{v / 1e4:.{digits}f}만"
    return f"{sign}{v:,.0f}"


def pct(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}%"


def num(value: float | None, *, digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """읽기 좋은 눈금값. 축 범위를 데이터에 맞춰 조금 넓힌다."""
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0, 1.0]
    if low == high:
        if low == 0:
            return [0.0, 1.0]
        low, high = min(0.0, low * 1.2), max(0.0, high * 1.2)
    span = high - low
    raw = span / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if raw <= step:
            break
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 0.5:
        ticks.append(round(value, 10))
        value += step
    return ticks or [low, high]


def _empty(message: str, width: int = 720, height: int = 240) -> str:
    return (
        f'<svg class="chart chart--empty" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(message)}" preserveAspectRatio="xMidYMid meet">'
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" fill="none" '
        f'stroke="{GRID}" stroke-dasharray="4 4"/>'
        f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" fill="{MUTED}" '
        f'font-size="13">{escape(message)}</text></svg>'
    )


def _open(width: int, height: int, label: str, cls: str = "chart") -> str:
    return (
        f'<svg class="{cls}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(label)}" preserveAspectRatio="xMidYMid meet">'
    )


def _y_axis(
    ticks: Sequence[float],
    *,
    plot_left: float,
    plot_right: float,
    scale,
    formatter=krw,
) -> str:
    parts = []
    for tick in ticks:
        y = scale(tick)
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}" '
            f'font-size="11" class="tick">{escape(formatter(tick))}</text>'
        )
    return "".join(parts)


def _x_labels(labels: Sequence[str], xs: Sequence[float], baseline: float, *, every: int = 1) -> str:
    parts = []
    for index, (label, x) in enumerate(zip(labels, xs)):
        if index % every:
            continue
        parts.append(
            f'<text x="{x:.1f}" y="{baseline + 18:.1f}" text-anchor="middle" fill="{MUTED}" '
            f'font-size="11" class="tick">{escape(str(label))}</text>'
        )
    return "".join(parts)


def _legend(items: Sequence[tuple[str, str]], x: float, y: float) -> str:
    parts = []
    offset = 0.0
    for name, color in items:
        parts.append(
            f'<rect x="{x + offset:.1f}" y="{y - 9:.1f}" width="10" height="10" fill="{color}" rx="1"/>'
            f'<text x="{x + offset + 15:.1f}" y="{y:.1f}" fill="{INK}" font-size="12">{escape(name)}</text>'
        )
        offset += 22 + len(name) * 8.2
    return "".join(parts)


def _path(points: Sequence[tuple[float, float]]) -> str:
    if not points:
        return ""
    head = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
    tail = " ".join(f"L {x:.1f} {y:.1f}" for x, y in points[1:])
    return f"{head} {tail}".strip()


# --------------------------------------------------------------------------- #
# 1. 선 그래프 (시계열)
# --------------------------------------------------------------------------- #


def line_chart(
    labels: Sequence[str],
    series: Sequence[dict[str, Any]],
    *,
    title: str = "",
    marker_index: int | None = None,
    marker_label: str = "",
    formatter=krw,
    height: int = 300,
    width: int = 860,
    show_points: bool = True,
) -> str:
    """여러 계열을 겹쳐 그린다. ``marker_index`` 에 정책 시행일 세로선을 세운다."""
    usable = [s for s in series if s.get("values")]
    if not labels or not usable:
        return _empty("표시할 시계열이 없습니다")
    left, right, top, bottom = 76, 20, 28, 44
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for s in usable for v in s["values"] if v is not None]
    if not values:
        return _empty("시계열 값이 모두 비어 있습니다")
    ticks = _nice_ticks(min(0.0, min(values)), max(values))
    low, high = ticks[0], ticks[-1]

    def sy(value: float) -> float:
        if high == low:
            return top + plot_h / 2
        return top + plot_h - (value - low) / (high - low) * plot_h

    n = len(labels)
    xs = [left + (plot_w * i / max(n - 1, 1)) for i in range(n)] if n > 1 else [left + plot_w / 2]

    out = [_open(width, height, title or "시계열 그래프")]
    out.append(_y_axis(ticks, plot_left=left, plot_right=width - right, scale=sy, formatter=formatter))
    out.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    if marker_index is not None and 0 <= marker_index < n:
        mx = xs[marker_index]
        out.append(
            f'<line x1="{mx:.1f}" y1="{top}" x2="{mx:.1f}" y2="{top + plot_h}" stroke="{NEG}" '
            f'stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        if marker_label:
            out.append(
                f'<text x="{mx + 6:.1f}" y="{top + 12}" fill="{NEG}" font-size="11">{escape(marker_label)}</text>'
            )
    for index, s in enumerate(usable):
        color = s.get("color") or SERIES_VARS[index % len(SERIES_VARS)]
        points = [(x, sy(v)) for x, v in zip(xs, s["values"]) if v is not None]
        dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
        out.append(
            f'<path d="{_path(points)}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash}/>'
        )
        if show_points and len(points) <= 40:
            for x, y in points:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="{color}"/>')
    every = max(1, math.ceil(n / 12))
    out.append(_x_labels(labels, xs, top + plot_h, every=every))
    out.append(
        _legend(
            [(s.get("name", f"계열{i + 1}"), s.get("color") or SERIES_VARS[i % len(SERIES_VARS)]) for i, s in enumerate(usable)],
            left,
            height - 8,
        )
    )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 2. 시행 전/후 겹쳐보기
# --------------------------------------------------------------------------- #


def overlay_chart(
    labels: Sequence[str],
    pre: Sequence[float],
    post: Sequence[float],
    *,
    pre_name: str = "시행 전",
    post_name: str = "시행 후",
    title: str = "",
    formatter=krw,
    height: int = 320,
    width: int = 860,
) -> str:
    """같은 길이의 두 구간을 한 축에 겹치고 **차이를 면으로 칠한다**.

    선 두 개만 그리면 '어디서 얼마나' 벌어졌는지 눈으로 못 읽는다.
    사후가 위면 증가색, 아래면 감소색으로 면을 채워 방향까지 같이 보여준다.
    """
    if not labels or not pre or not post or len(pre) != len(post):
        return _empty("전/후 길이가 같은 구간이 없어 겹쳐 그릴 수 없습니다")
    left, right, top, bottom = 76, 20, 28, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for v in list(pre) + list(post) if v is not None]
    ticks = _nice_ticks(min(0.0, min(values)), max(values))
    low, high = ticks[0], ticks[-1]

    def sy(value: float) -> float:
        if high == low:
            return top + plot_h / 2
        return top + plot_h - (value - low) / (high - low) * plot_h

    n = len(labels)
    xs = [left + (plot_w * i / max(n - 1, 1)) for i in range(n)] if n > 1 else [left + plot_w / 2]
    pre_pts = [(x, sy(v)) for x, v in zip(xs, pre)]
    post_pts = [(x, sy(v)) for x, v in zip(xs, post)]

    out = [_open(width, height, title or "시행 전후 겹쳐보기")]
    out.append(_y_axis(ticks, plot_left=left, plot_right=width - right, scale=sy, formatter=formatter))

    # 증가 구간과 감소 구간을 나눠 칠한다.
    for index in range(n - 1):
        quad = [pre_pts[index], pre_pts[index + 1], post_pts[index + 1], post_pts[index]]
        rising = (post[index] + post[index + 1]) >= (pre[index] + pre[index + 1])
        fill = POS if rising else NEG
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in quad)
        out.append(f'<polygon points="{pts}" fill="{fill}" fill-opacity="0.16" stroke="none"/>')

    out.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    out.append(
        f'<path d="{_path(pre_pts)}" fill="none" stroke="{SERIES_VARS[1]}" stroke-width="2.2" '
        f'stroke-dasharray="6 4" stroke-linejoin="round"/>'
    )
    out.append(
        f'<path d="{_path(post_pts)}" fill="none" stroke="{SERIES_VARS[0]}" stroke-width="2.6" '
        f'stroke-linejoin="round"/>'
    )
    for x, y in pre_pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{SERIES_VARS[1]}"/>')
    for x, y in post_pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{SERIES_VARS[0]}"/>')
    out.append(_x_labels(labels, xs, top + plot_h, every=max(1, math.ceil(n / 12))))
    out.append(
        _legend(
            [(pre_name, SERIES_VARS[1]), (post_name, SERIES_VARS[0]), ("증가 구간", POS), ("감소 구간", NEG)],
            left,
            height - 10,
        )
    )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 3. 그룹 막대 (업종별 전/후)
# --------------------------------------------------------------------------- #


def grouped_bar(
    labels: Sequence[str],
    groups: Sequence[dict[str, Any]],
    *,
    title: str = "",
    formatter=krw,
    height: int = 340,
    width: int = 860,
    value_labels: bool = False,
) -> str:
    usable = [g for g in groups if g.get("values")]
    if not labels or not usable:
        return _empty("비교할 막대 데이터가 없습니다")
    left, right, top, bottom = 76, 20, 28, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for g in usable for v in g["values"] if v is not None]
    ticks = _nice_ticks(min(0.0, min(values)), max(values))
    low, high = ticks[0], ticks[-1]

    def sy(value: float) -> float:
        if high == low:
            return top + plot_h
        return top + plot_h - (value - low) / (high - low) * plot_h

    n = len(labels)
    slot = plot_w / max(n, 1)
    bar_w = min(28.0, slot / (len(usable) + 0.8))
    zero = sy(0)

    out = [_open(width, height, title or "그룹 막대")]
    out.append(_y_axis(ticks, plot_left=left, plot_right=width - right, scale=sy, formatter=formatter))
    for index in range(n):
        center = left + slot * (index + 0.5)
        for gi, group in enumerate(usable):
            value = group["values"][index] if index < len(group["values"]) else None
            if value is None:
                continue
            color = group.get("color") or SERIES_VARS[gi % len(SERIES_VARS)]
            x = center - (len(usable) * bar_w) / 2 + gi * bar_w
            y = sy(value)
            out.append(
                f'<rect x="{x:.1f}" y="{min(y, zero):.1f}" width="{bar_w - 2:.1f}" '
                f'height="{abs(zero - y):.1f}" fill="{color}" rx="1"><title>'
                f"{escape(str(labels[index]))} · {escape(group.get('name', ''))} · "
                f"{escape(formatter(value))}</title></rect>"
            )
            if value_labels:
                out.append(
                    f'<text x="{x + bar_w / 2 - 1:.1f}" y="{min(y, zero) - 4:.1f}" text-anchor="middle" '
                    f'fill="{MUTED}" font-size="10">{escape(formatter(value))}</text>'
                )
    out.append(
        f'<line x1="{left}" y1="{zero:.1f}" x2="{width - right}" y2="{zero:.1f}" stroke="{AXIS}" stroke-width="1"/>'
    )
    for index, label in enumerate(labels):
        center = left + slot * (index + 0.5)
        out.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" fill="{MUTED}" '
            f'font-size="11" class="tick">{escape(str(label))}</text>'
        )
    out.append(
        _legend(
            [(g.get("name", f"계열{i + 1}"), g.get("color") or SERIES_VARS[i % len(SERIES_VARS)]) for i, g in enumerate(usable)],
            left,
            height - 10,
        )
    )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 4. 발산 막대 (증감·DID)
# --------------------------------------------------------------------------- #


def diverging_bar(
    items: Sequence[dict[str, Any]],
    *,
    title: str = "",
    formatter=krw,
    width: int = 860,
    row_height: int = 26,
    highlight_key: str = "targeted",
) -> str:
    """0 기준 좌우 막대. 대상 업종은 진하게, 비대상은 흐리게 칠해 구분한다."""
    rows = [item for item in items if item.get("value") is not None]
    if not rows:
        return _empty("표시할 증감 값이 없습니다")
    label_w = 108
    left, right, top = label_w + 12, 76, 16
    height = top + row_height * len(rows) + 30
    plot_w = width - left - right
    span = max(abs(float(item["value"])) for item in rows) or 1.0
    zero = left + plot_w / 2

    out = [_open(width, height, title or "증감 막대")]
    out.append(f'<line x1="{zero:.1f}" y1="{top - 6}" x2="{zero:.1f}" y2="{height - 24}" stroke="{AXIS}" stroke-width="1"/>')
    for index, item in enumerate(rows):
        value = float(item["value"])
        y = top + index * row_height
        length = abs(value) / span * (plot_w / 2 - 8)
        x = zero if value >= 0 else zero - length
        color = POS if value >= 0 else NEG
        opacity = "1" if item.get(highlight_key) else "0.45"
        out.append(
            f'<rect x="{x:.1f}" y="{y + 5:.1f}" width="{length:.1f}" height="{row_height - 12}" '
            f'fill="{color}" fill-opacity="{opacity}" rx="1"><title>'
            f"{escape(str(item.get('label', '')))} · {escape(formatter(value))}</title></rect>"
        )
        out.append(
            f'<text x="{label_w}" y="{y + row_height / 2 + 4:.1f}" text-anchor="end" fill="{INK}" '
            f'font-size="12">{escape(str(item.get("label", "")))}</text>'
        )
        text_x = zero + length + 6 if value >= 0 else zero - length - 6
        anchor = "start" if value >= 0 else "end"
        out.append(
            f'<text x="{text_x:.1f}" y="{y + row_height / 2 + 4:.1f}" text-anchor="{anchor}" '
            f'fill="{MUTED}" font-size="11" class="tick">{escape(formatter(value))}</text>'
        )
    out.append(
        _legend([("정책 대상 업종", POS), ("비대상 업종(대조군)", MUTED)], label_w + 12, height - 6)
    )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 5. DID 슬로프 차트
# --------------------------------------------------------------------------- #


def slope_chart(
    *,
    treat_pre: float,
    treat_post: float,
    control_pre: float,
    control_post: float,
    counterfactual: float | None,
    treat_name: str = "정책 대상 업종",
    control_name: str = "대조군 업종",
    formatter=krw,
    width: int = 720,
    height: int = 340,
) -> str:
    """이중차분의 핵심 그림. 반사실선을 점선으로 넣어 DID 가 어느 간격인지 보여준다."""
    values = [treat_pre, treat_post, control_pre, control_post]
    if counterfactual is not None:
        values.append(counterfactual)
    values = [v for v in values if v is not None]
    if not values:
        return _empty("이중차분을 그릴 값이 없습니다")
    left, right, top, bottom = 84, 150, 30, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    ticks = _nice_ticks(min(0.0, min(values)), max(values))
    low, high = ticks[0], ticks[-1]

    def sy(value: float) -> float:
        if high == low:
            return top + plot_h / 2
        return top + plot_h - (value - low) / (high - low) * plot_h

    x0, x1 = left, left + plot_w
    out = [_open(width, height, "이중차분 슬로프 차트")]
    out.append(_y_axis(ticks, plot_left=left, plot_right=x1, scale=sy, formatter=formatter))
    out.append(f'<line x1="{x0}" y1="{top + plot_h}" x2="{x1}" y2="{top + plot_h}" stroke="{AXIS}"/>')
    out.append(
        f'<text x="{x0}" y="{top + plot_h + 20}" text-anchor="middle" fill="{MUTED}" font-size="12">시행 전</text>'
        f'<text x="{x1}" y="{top + plot_h + 20}" text-anchor="middle" fill="{MUTED}" font-size="12">시행 후</text>'
    )

    if counterfactual is not None:
        out.append(
            f'<polygon points="{x1},{sy(counterfactual):.1f} {x1},{sy(treat_post):.1f} '
            f'{x1 - 26},{sy(treat_post):.1f} {x1 - 26},{sy(counterfactual):.1f}" '
            f'fill="{POS if treat_post >= counterfactual else NEG}" fill-opacity="0.18"/>'
        )
        out.append(
            f'<line x1="{x0}" y1="{sy(treat_pre):.1f}" x2="{x1}" y2="{sy(counterfactual):.1f}" '
            f'stroke="{MUTED}" stroke-width="2" stroke-dasharray="6 5"/>'
        )
        out.append(
            f'<circle cx="{x1}" cy="{sy(counterfactual):.1f}" r="4" fill="none" stroke="{MUTED}" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{x1 + 10}" y="{sy(counterfactual) + 4:.1f}" fill="{MUTED}" font-size="11">'
            f'반사실 {escape(formatter(counterfactual))}</text>'
        )

    for name, pre, post, color in (
        (treat_name, treat_pre, treat_post, SERIES_VARS[0]),
        (control_name, control_pre, control_post, SERIES_VARS[1]),
    ):
        out.append(
            f'<line x1="{x0}" y1="{sy(pre):.1f}" x2="{x1}" y2="{sy(post):.1f}" stroke="{color}" stroke-width="2.6"/>'
        )
        out.append(f'<circle cx="{x0}" cy="{sy(pre):.1f}" r="4.5" fill="{color}"/>')
        out.append(f'<circle cx="{x1}" cy="{sy(post):.1f}" r="4.5" fill="{color}"/>')
        out.append(
            f'<text x="{x1 + 10}" y="{sy(post) + 4:.1f}" fill="{color}" font-size="12">{escape(name)}</text>'
        )
        out.append(
            f'<text x="{x0 - 10}" y="{sy(pre) + 4:.1f}" text-anchor="end" fill="{color}" font-size="11" '
            f'class="tick">{escape(formatter(pre))}</text>'
        )

    if counterfactual is not None:
        gap = treat_post - counterfactual
        mid_y = (sy(counterfactual) + sy(treat_post)) / 2
        out.append(
            f'<text x="{x1 - 34}" y="{mid_y + 4:.1f}" text-anchor="end" fill="{INK}" font-size="12" '
            f'font-weight="600">DID {escape(formatter(gap))}</text>'
        )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 6. 히트맵 (업종 × 일자)
# --------------------------------------------------------------------------- #


def heatmap(
    rows: Sequence[str],
    cols: Sequence[str],
    matrix: Sequence[Sequence[float | None]],
    *,
    title: str = "",
    formatter=pct,
    width: int = 860,
    cell_h: int = 22,
) -> str:
    if not rows or not cols:
        return _empty("히트맵을 만들 교차표가 없습니다")
    label_w = 104
    top = 34
    height = top + cell_h * len(rows) + 34
    cell_w = (width - label_w - 16) / len(cols)
    flat = [v for row in matrix for v in row if v is not None]
    span = max((abs(v) for v in flat), default=1.0) or 1.0

    out = [_open(width, height, title or "히트맵")]
    every = max(1, math.ceil(len(cols) / 16))
    for ci, col in enumerate(cols):
        if ci % every:
            continue
        out.append(
            f'<text x="{label_w + cell_w * (ci + 0.5):.1f}" y="{top - 10}" text-anchor="middle" '
            f'fill="{MUTED}" font-size="10" class="tick">{escape(str(col)[5:] or str(col))}</text>'
        )
    for ri, row_name in enumerate(rows):
        y = top + ri * cell_h
        out.append(
            f'<text x="{label_w - 8}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end" fill="{INK}" '
            f'font-size="11">{escape(str(row_name))}</text>'
        )
        for ci in range(len(cols)):
            value = matrix[ri][ci] if ci < len(matrix[ri]) else None
            x = label_w + cell_w * ci
            if value is None:
                out.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 1:.1f}" height="{cell_h - 1}" '
                    f'fill="none" stroke="{GRID}" stroke-dasharray="2 2"/>'
                )
                continue
            intensity = min(abs(value) / span, 1.0) ** 0.6
            color = POS if value >= 0 else NEG
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 1:.1f}" height="{cell_h - 1}" '
                f'fill="{color}" fill-opacity="{intensity:.3f}"><title>'
                f"{escape(str(row_name))} · {escape(str(cols[ci]))} · {escape(formatter(value))}</title></rect>"
            )
    out.append(_legend([("기준 대비 증가", POS), ("기준 대비 감소", NEG)], label_w, height - 8))
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 7. 100% 누적 막대 (구성비)
# --------------------------------------------------------------------------- #


def stacked_bar(
    labels: Sequence[str],
    series: Sequence[dict[str, Any]],
    *,
    title: str = "",
    normalize: bool = False,
    formatter=krw,
    width: int = 860,
    height: int = 320,
) -> str:
    usable = [s for s in series if s.get("values")]
    if not labels or not usable:
        return _empty("누적 막대 데이터가 없습니다")
    left, right, top, bottom = 76, 20, 24, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    totals = [sum((s["values"][i] or 0) for s in usable) for i in range(len(labels))]
    high = 100.0 if normalize else (max(totals) if totals else 1.0)
    ticks = _nice_ticks(0.0, high)
    high = ticks[-1]

    def sy(value: float) -> float:
        return top + plot_h - (value / high) * plot_h if high else top + plot_h

    slot = plot_w / max(len(labels), 1)
    bar_w = min(46.0, slot * 0.62)
    out = [_open(width, height, title or "구성비 막대")]
    out.append(
        _y_axis(
            ticks,
            plot_left=left,
            plot_right=width - right,
            scale=sy,
            formatter=(lambda v: f"{v:.0f}%") if normalize else formatter,
        )
    )
    for index, label in enumerate(labels):
        total = totals[index] or 1.0
        base = 0.0
        x = left + slot * (index + 0.5) - bar_w / 2
        for si, s in enumerate(usable):
            raw = s["values"][index] or 0
            value = (raw / total * 100.0) if normalize else raw
            if value <= 0:
                continue
            color = s.get("color") or SERIES_VARS[si % len(SERIES_VARS)]
            y0, y1 = sy(base + value), sy(base)
            out.append(
                f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{max(y1 - y0, 0.5):.1f}" '
                f'fill="{color}"><title>{escape(str(label))} · {escape(s.get("name", ""))} · '
                f'{escape(formatter(raw))}</title></rect>'
            )
            base += value
        out.append(
            f'<text x="{left + slot * (index + 0.5):.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" '
            f'fill="{MUTED}" font-size="11" class="tick">{escape(str(label))}</text>'
        )
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="{AXIS}"/>')
    out.append(
        _legend(
            [(s.get("name", f"계열{i + 1}"), s.get("color") or SERIES_VARS[i % len(SERIES_VARS)]) for i, s in enumerate(usable)],
            left,
            height - 10,
        )
    )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 8. 폭포 차트 (DID 분해)
# --------------------------------------------------------------------------- #


def waterfall(
    steps: Sequence[dict[str, Any]],
    *,
    title: str = "",
    formatter=krw,
    width: int = 860,
    height: int = 320,
) -> str:
    """`시행 전 → 시장 전체 추세 → 정책 순효과 → 시행 후` 를 눈으로 잇는다."""
    if not steps:
        return _empty("분해할 단계가 없습니다")
    left, right, top, bottom = 76, 20, 24, 66
    plot_w = width - left - right
    plot_h = height - top - bottom
    running = 0.0
    tops = []
    for step in steps:
        if step.get("absolute"):
            start, end = 0.0, float(step["value"])
        else:
            start, end = running, running + float(step["value"])
        tops.append((start, end))
        running = end
    values = [v for pair in tops for v in pair]
    ticks = _nice_ticks(min(0.0, min(values)), max(values))
    low, high = ticks[0], ticks[-1]

    def sy(value: float) -> float:
        if high == low:
            return top + plot_h / 2
        return top + plot_h - (value - low) / (high - low) * plot_h

    slot = plot_w / max(len(steps), 1)
    bar_w = min(74.0, slot * 0.56)
    out = [_open(width, height, title or "효과 분해")]
    out.append(_y_axis(ticks, plot_left=left, plot_right=width - right, scale=sy, formatter=formatter))
    previous_x = None
    for index, (step, (start, end)) in enumerate(zip(steps, tops)):
        cx = left + slot * (index + 0.5)
        x = cx - bar_w / 2
        y0, y1 = sy(max(start, end)), sy(min(start, end))
        if step.get("absolute"):
            color = SERIES_VARS[1]
        else:
            color = POS if end >= start else NEG
        out.append(
            f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{max(y1 - y0, 1.5):.1f}" '
            f'fill="{color}" fill-opacity="{0.95 if step.get("absolute") else 0.8}" rx="1"/>'
        )
        out.append(
            f'<text x="{cx:.1f}" y="{y0 - 6:.1f}" text-anchor="middle" fill="{INK}" font-size="11">'
            f'{escape(formatter(step["value"]))}</text>'
        )
        out.append(
            f'<text x="{cx:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" fill="{MUTED}" '
            f'font-size="11">{escape(str(step.get("label", "")))}</text>'
        )
        if previous_x is not None:
            out.append(
                f'<line x1="{previous_x:.1f}" y1="{sy(start):.1f}" x2="{x:.1f}" y2="{sy(start):.1f}" '
                f'stroke="{GRID}" stroke-width="1" stroke-dasharray="3 3"/>'
            )
        previous_x = x + bar_w
    out.append(f'<line x1="{left}" y1="{sy(0):.1f}" x2="{width - right}" y2="{sy(0):.1f}" stroke="{AXIS}"/>')
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 9. 산점도
# --------------------------------------------------------------------------- #


def scatter(
    points: Sequence[dict[str, Any]],
    *,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    x_format=num,
    y_format=krw,
    width: int = 720,
    height: int = 320,
) -> str:
    usable = [p for p in points if p.get("x") is not None and p.get("y") is not None]
    if not usable:
        return _empty("산점도를 그릴 점이 없습니다")
    left, right, top, bottom = 76, 24, 24, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [float(p["x"]) for p in usable]
    ys = [float(p["y"]) for p in usable]
    xt = _nice_ticks(min(0.0, min(xs)), max(xs))
    yt = _nice_ticks(min(0.0, min(ys)), max(ys))

    def sx(value: float) -> float:
        return left + (value - xt[0]) / (xt[-1] - xt[0] or 1) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - yt[0]) / (yt[-1] - yt[0] or 1) * plot_h

    out = [_open(width, height, title or "산점도")]
    out.append(_y_axis(yt, plot_left=left, plot_right=width - right, scale=sy, formatter=y_format))
    for tick in xt:
        out.append(
            f'<text x="{sx(tick):.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" fill="{MUTED}" '
            f'font-size="11" class="tick">{escape(x_format(tick))}</text>'
        )
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="{AXIS}"/>')
    for point in usable:
        color = point.get("color") or SERIES_VARS[0]
        r = 4.0 + min(float(point.get("size", 0) or 0) ** 0.5 / 6, 9.0)
        out.append(
            f'<circle cx="{sx(float(point["x"])):.1f}" cy="{sy(float(point["y"])):.1f}" r="{r:.1f}" '
            f'fill="{color}" fill-opacity="0.68" stroke="{color}"><title>'
            f'{escape(str(point.get("label", "")))} · {escape(x_format(point["x"]))} · '
            f'{escape(y_format(point["y"]))}</title></circle>'
        )
        if point.get("label") and len(usable) <= 18:
            out.append(
                f'<text x="{sx(float(point["x"])) + r + 4:.1f}" y="{sy(float(point["y"])) + 4:.1f}" '
                f'fill="{MUTED}" font-size="10">{escape(str(point["label"]))}</text>'
            )
    if x_label:
        out.append(
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" text-anchor="middle" fill="{MUTED}" '
            f'font-size="11">{escape(x_label)}</text>'
        )
    if y_label:
        out.append(
            f'<text x="14" y="{top + plot_h / 2:.1f}" fill="{MUTED}" font-size="11" '
            f'transform="rotate(-90 14 {top + plot_h / 2:.1f})" text-anchor="middle">{escape(y_label)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 10. 이벤트 스터디 (상대일 격차)
# --------------------------------------------------------------------------- #


def event_study_chart(
    points: Sequence[dict[str, Any]],
    *,
    title: str = "",
    width: int = 860,
    height: int = 300,
) -> str:
    """사전기간 평균을 0 으로 둔 처치−대조 로그격차. 0 이전이 평평해야 DID 가설이 산다."""
    usable = [p for p in points if p.get("normalized_gap") is not None]
    if not usable:
        return _empty("사전 추세를 확인할 관측이 없습니다")
    left, right, top, bottom = 76, 20, 24, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(p["normalized_gap"]) for p in usable]
    ticks = _nice_ticks(min(values), max(values))
    low, high = ticks[0], ticks[-1]

    def sy(value: float) -> float:
        if high == low:
            return top + plot_h / 2
        return top + plot_h - (value - low) / (high - low) * plot_h

    rels = [int(p["rel_day"]) for p in usable]
    lo, hi = min(rels), max(rels)

    def sx(rel: int) -> float:
        return left + (rel - lo) / ((hi - lo) or 1) * plot_w

    out = [_open(width, height, title or "이벤트 스터디")]
    out.append(
        _y_axis(ticks, plot_left=left, plot_right=width - right, scale=sy, formatter=lambda v: f"{v:+.2f}")
    )
    zero_x = sx(0)
    out.append(
        f'<rect x="{left}" y="{top}" width="{max(zero_x - left, 0):.1f}" height="{plot_h}" fill="{BAND}" '
        f'fill-opacity="0.35"/>'
    )
    out.append(
        f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" stroke="{NEG}" '
        f'stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    out.append(
        f'<text x="{zero_x + 6:.1f}" y="{top + 12}" fill="{NEG}" font-size="11">시행일</text>'
        f'<text x="{left + 6}" y="{top + 12}" fill="{MUTED}" font-size="11">사전기간(평평해야 함)</text>'
    )
    out.append(f'<line x1="{left}" y1="{sy(0):.1f}" x2="{width - right}" y2="{sy(0):.1f}" stroke="{AXIS}"/>')
    pts = [(sx(int(p["rel_day"])), sy(float(p["normalized_gap"]))) for p in usable]
    out.append(f'<path d="{_path(pts)}" fill="none" stroke="{SERIES_VARS[0]}" stroke-width="2.2"/>')
    for point, (x, y) in zip(usable, pts):
        color = SERIES_VARS[0] if int(point["rel_day"]) >= 0 else MUTED
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}"><title>'
            f'{escape(str(point["day"]))} (D{int(point["rel_day"]):+d}) · '
            f'{float(point["normalized_gap"]):+.4f}</title></circle>'
        )
    for point, (x, _) in zip(usable, pts):
        rel = int(point["rel_day"])
        if rel % max(1, math.ceil(len(usable) / 12)) == 0:
            out.append(
                f'<text x="{x:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="middle" fill="{MUTED}" '
                f'font-size="10" class="tick">D{rel:+d}</text>'
            )
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# 11. 도넛 (구성)
# --------------------------------------------------------------------------- #


def donut(items: Sequence[dict[str, Any]], *, title: str = "", size: int = 260, formatter=krw) -> str:
    usable = [i for i in items if (i.get("value") or 0) > 0]
    if not usable:
        return _empty("구성비를 그릴 값이 없습니다", width=size, height=size)
    total = sum(float(i["value"]) for i in usable)
    cx = cy = size / 2
    outer, inner = size / 2 - 8, size / 2 - 42
    out = [_open(size, size, title or "구성비")]
    angle = -math.pi / 2
    for index, item in enumerate(usable):
        fraction = float(item["value"]) / total
        end = angle + fraction * 2 * math.pi
        large = 1 if fraction > 0.5 else 0
        x0, y0 = cx + outer * math.cos(angle), cy + outer * math.sin(angle)
        x1, y1 = cx + outer * math.cos(end), cy + outer * math.sin(end)
        x2, y2 = cx + inner * math.cos(end), cy + inner * math.sin(end)
        x3, y3 = cx + inner * math.cos(angle), cy + inner * math.sin(angle)
        color = item.get("color") or SERIES_VARS[index % len(SERIES_VARS)]
        out.append(
            f'<path d="M {x0:.2f} {y0:.2f} A {outer:.2f} {outer:.2f} 0 {large} 1 {x1:.2f} {y1:.2f} '
            f'L {x2:.2f} {y2:.2f} A {inner:.2f} {inner:.2f} 0 {large} 0 {x3:.2f} {y3:.2f} Z" '
            f'fill="{color}"><title>{escape(str(item.get("label", "")))} · '
            f'{escape(formatter(item["value"]))} ({fraction * 100:.1f}%)</title></path>'
        )
        angle = end
    out.append(
        f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" fill="{INK}" font-size="15" font-weight="600">'
        f'{escape(formatter(total))}</text>'
        f'<text x="{cx}" y="{cy + 16}" text-anchor="middle" fill="{MUTED}" font-size="11">합계</text>'
    )
    out.append("</svg>")
    return "".join(out)
