/**
 * 일별 추이 선 그래프 — SVG 하나, 의존성 없음.
 *
 * 색은 토큰만 쓴다(§2). 계열은 많아야 셋: 공통 사전(회색) · OFF(네이비) · ON(파랑).
 * 색만으로 구분하지 않도록 범례에 이름을 적고, ON 은 점선으로 그린다.
 * 값이 없는 날은 선을 끊는다 — 0 으로 잇지 않는다.
 */
import { useId, useMemo, useState } from 'react';

export interface LineSeries {
  key: string;
  label: string;
  tone: 'neutral' | 'primary' | 'alt';
  dashed?: boolean;
  points: Array<{ x: string; y: number | null }>;
}

const STROKE: Record<LineSeries['tone'], string> = {
  neutral: 'var(--fg-subtle)',
  primary: 'var(--bar)',
  alt: 'var(--bar-alt)',
};

const WIDTH = 720;
const HEIGHT = 240;
const PAD = { top: 16, right: 16, bottom: 32, left: 64 };

function niceMax(value: number): number {
  if (value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const scaled = value / power;
  const step = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return step * power;
}

export function LineChart({
  series,
  format,
  marker,
  title,
}: {
  series: LineSeries[];
  format: (value: number) => string;
  /** 세로 기준선(예: 정책 시행일) */
  marker?: { x: string; label: string };
  title: string;
}) {
  const id = useId();
  const [hover, setHover] = useState<string | null>(null);
  const xs = useMemo(() => {
    const all = new Set<string>();
    series.forEach((s) => s.points.forEach((p) => all.add(p.x)));
    return Array.from(all).sort();
  }, [series]);
  const max = useMemo(
    () => niceMax(Math.max(0, ...series.flatMap((s) => s.points.map((p) => p.y ?? 0)))),
    [series],
  );
  if (!xs.length) return <p className="chart__empty">값 없음</p>;

  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  const xAt = (x: string) => PAD.left + (xs.length === 1 ? innerW / 2 : (xs.indexOf(x) / (xs.length - 1)) * innerW);
  const yAt = (y: number) => PAD.top + innerH - (y / max) * innerH;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * max);
  const labelEvery = Math.max(1, Math.ceil(xs.length / 7));

  function path(points: LineSeries['points']): string {
    let d = '';
    let open = false;
    for (const point of points) {
      if (point.y === null || !Number.isFinite(point.y)) {
        open = false;
        continue;
      }
      d += `${open ? 'L' : 'M'}${xAt(point.x).toFixed(1)},${yAt(point.y).toFixed(1)}`;
      open = true;
    }
    return d;
  }

  const hovered = hover
    ? series.map((s) => ({ s, point: s.points.find((p) => p.x === hover) })).filter((v) => v.point && v.point.y !== null)
    : [];

  return (
    <figure className="chart" aria-labelledby={`${id}-title`}>
      <figcaption id={`${id}-title`} className="chart__title">
        {title}
      </figcaption>
      <ul className="chart__legend">
        {series.map((s) => (
          <li key={s.key}>
            <svg width="22" height="8" aria-hidden="true">
              <line x1="0" y1="4" x2="22" y2="4" stroke={STROKE[s.tone]} strokeWidth="2.5" strokeDasharray={s.dashed ? '5 3' : undefined} />
            </svg>
            {s.label}
          </li>
        ))}
      </ul>
      <svg
        className="chart__svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${title} — ${series.map((s) => s.label).join(', ')}`}
        onMouseLeave={() => setHover(null)}
      >
        {ticks.map((value) => (
          <g key={value}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={yAt(value)} y2={yAt(value)} className="chart__grid" />
            <text x={PAD.left - 8} y={yAt(value) + 4} className="chart__tick" textAnchor="end">
              {format(value)}
            </text>
          </g>
        ))}
        {xs.map((x, index) =>
          // 마지막 날은 앞 라벨과 붙어 겹치지 않을 때만 따로 적는다
          index % labelEvery === 0 || (index === xs.length - 1 && index % labelEvery >= Math.ceil(labelEvery / 2)) ? (
            <text key={x} x={xAt(x)} y={HEIGHT - 10} className="chart__tick" textAnchor="middle">
              {x.slice(5)}
            </text>
          ) : null,
        )}
        {marker && xs.includes(marker.x) ? (
          <g>
            <line x1={xAt(marker.x)} x2={xAt(marker.x)} y1={PAD.top} y2={PAD.top + innerH} className="chart__marker" />
            <text x={xAt(marker.x) + 4} y={PAD.top + 10} className="chart__tick">
              {marker.label}
            </text>
          </g>
        ) : null}
        {series.map((s) => (
          <path
            key={s.key}
            d={path(s.points)}
            fill="none"
            stroke={STROKE[s.tone]}
            strokeWidth="2.5"
            strokeDasharray={s.dashed ? '6 4' : undefined}
            strokeLinejoin="round"
          />
        ))}
        {series.map((s) =>
          s.points
            .filter((p) => p.y !== null)
            .map((p) => <circle key={`${s.key}-${p.x}`} cx={xAt(p.x)} cy={yAt(p.y as number)} r={hover === p.x ? 4 : 2.5} fill={STROKE[s.tone]} />),
        )}
        {xs.map((x) => (
          <rect
            key={`hit-${x}`}
            x={xAt(x) - innerW / Math.max(1, xs.length - 1) / 2}
            y={PAD.top}
            width={innerW / Math.max(1, xs.length - 1)}
            height={innerH}
            fill="transparent"
            onMouseEnter={() => setHover(x)}
          />
        ))}
      </svg>
      <p className="chart__readout" aria-live="polite">
        {hover && hovered.length
          ? `${hover} · ` + hovered.map(({ s, point }) => `${s.label} ${format(point!.y as number)}`).join(' · ')
          : ' '}
      </p>
    </figure>
  );
}
