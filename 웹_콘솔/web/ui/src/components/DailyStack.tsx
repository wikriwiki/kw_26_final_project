/**
 * 일별 막대 — 하루 총량 막대 안에 그중 일부(예: 정책 지갑에서 나간 돈)를 겹쳐 쌓는다.
 *
 * 선 두 개로 그리지 않는 이유: 정책 지급분은 총결제의 몇 %라서 같은 축의 선이면 바닥에
 * 붙어 보이지 않는다. 한 막대 안의 몫으로 그리면 "언제부터 얼마나"가 모양으로 읽힌다.
 *
 * 색은 토큰 두 개뿐이다 — 전체는 옅은 먹남색, 부분은 먹남색. 값이 없는 날은 막대를 그리지
 * 않는다(0 으로 채우지 않는다). 마우스를 올린 날의 숫자는 그래프 위 한 줄에 적는다.
 */
import { useMemo, useState } from 'react';

export interface DailyStackPoint {
  day: string;
  total: number | null;
  part: number | null;
}

const W = 760;
const H = 220;
const PAD = { top: 12, right: 8, bottom: 28, left: 56 };

function niceMax(value: number): number {
  if (value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const scaled = value / power;
  const step = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((s) => scaled <= s) ?? 10;
  return step * power;
}

export function DailyStack({
  points,
  format,
  tickFormat = format,
  totalLabel,
  partLabel,
  marker,
  title,
}: {
  points: DailyStackPoint[];
  format: (value: number) => string;
  /** 축 눈금용 짧은 표기. 없으면 format */
  tickFormat?: (value: number) => string;
  totalLabel: string;
  partLabel: string;
  /** 세로 기준선 — 그날 막대 왼쪽에 긋는다(예: 정책 시행일) */
  marker?: { day: string; label: string } | null;
  title: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const max = useMemo(() => niceMax(Math.max(0, ...points.map((p) => p.total ?? 0))), [points]);
  if (!points.length) return <p className="chart__empty">값 없음</p>;

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const slot = innerW / points.length;
  const barW = Math.max(2, Math.min(28, slot * 0.62));
  const y = (v: number) => PAD.top + innerH - (v / max) * innerH;
  const ticks = [0, 0.5, 1].map((r) => r * max);
  const markerIdx = marker ? points.findIndex((p) => p.day >= marker.day) : -1;
  const shown = hover ?? points.length - 1;
  const cur = points[shown];
  const share = cur.total && cur.part !== null ? cur.part / cur.total : null;
  // 날짜 눈금은 많아야 7개 — 겹치지 않게 건너뛴다
  const every = Math.max(1, Math.ceil(points.length / 7));

  return (
    <figure className="dstack">
      <figcaption className="dstack__readout" aria-live="polite">
        <span className="dstack__day mono">{cur.day}</span>
        <span className="dstack__val">
          <i className="dstack__key dstack__key--total" aria-hidden="true" />
          {totalLabel} <b className="num">{cur.total === null ? '—' : format(cur.total)}</b>
        </span>
        <span className="dstack__val">
          <i className="dstack__key dstack__key--part" aria-hidden="true" />
          {partLabel} <b className="num">{cur.part === null ? '—' : format(cur.part)}</b>
          {share !== null ? <span className="dstack__share num">({(share * 100).toFixed(1)}%)</span> : null}
        </span>
      </figcaption>
      <svg
        className="dstack__svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={title}
        onMouseLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className="dstack__grid" x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} />
            <text className="dstack__tick" x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end">
              {t === 0 ? '0' : tickFormat(t)}
            </text>
          </g>
        ))}
        {markerIdx > 0 ? (
          <g>
            <line
              className="dstack__marker"
              x1={PAD.left + slot * markerIdx}
              x2={PAD.left + slot * markerIdx}
              y1={PAD.top - 4}
              y2={PAD.top + innerH}
            />
            <text className="dstack__markerlabel" x={PAD.left + slot * markerIdx + 6} y={PAD.top + 8}>
              {marker?.label}
            </text>
          </g>
        ) : null}
        {points.map((p, i) => {
          const cx = PAD.left + slot * i + slot / 2;
          const x = cx - barW / 2;
          const on = hover === i;
          return (
            <g key={p.day} onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)}>
              {/* 칸 전체를 잡는 투명 판 — 막대가 얇아도 마우스가 쉽게 걸린다 */}
              <rect x={PAD.left + slot * i} y={PAD.top} width={slot} height={innerH} fill="transparent" />
              {p.total !== null ? (
                <rect
                  className={`dstack__total${on ? ' is-on' : ''}`}
                  x={x}
                  y={y(p.total)}
                  width={barW}
                  height={Math.max(0, PAD.top + innerH - y(p.total))}
                />
              ) : null}
              {p.part !== null && p.part > 0 ? (
                <rect
                  className="dstack__part"
                  x={x}
                  y={y(p.part)}
                  width={barW}
                  height={Math.max(1, PAD.top + innerH - y(p.part))}
                />
              ) : null}
              {i % every === 0 || i === points.length - 1 ? (
                <text className="dstack__tick" x={cx} y={H - 8} textAnchor="middle">
                  {p.day.slice(5)}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
