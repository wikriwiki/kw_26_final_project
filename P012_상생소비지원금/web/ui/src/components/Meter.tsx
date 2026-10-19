/**
 * 막대 — 외부 차트 라이브러리 없이 CSS 로만 그린다.
 * 그라디언트를 쓰지 않고 단색 두 계열만 쓴다.
 *
 * 규칙: 비율을 모르면(`ratio === null`) 막대를 그리지 않는다.
 * 0% 로 그리면 "아무 일도 없었다"는 거짓말이 된다 (CONTRACT §3.3).
 */
import type { ReactNode } from 'react';

export interface MeterProps {
  /** 0~1. null 이면 호출하는 쪽에서 막대 대신 안내를 그려야 한다 */
  ratio: number;
  label: string;
  alt?: boolean;
  /** 트랙 높이 클래스 (예: progress__track) */
  className?: string;
}

export function Meter({ ratio, label, alt = false, className }: MeterProps) {
  const pct = Math.max(0, Math.min(1, ratio)) * 100;
  return (
    <span
      className={['meter', className ?? ''].filter(Boolean).join(' ')}
      role="progressbar"
      aria-label={label}
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <span
        className={alt ? 'meter__fill meter__fill--alt' : 'meter__fill'}
        style={{ width: `${pct}%` }}
      />
    </span>
  );
}

export interface BarItem {
  key: string;
  name: string;
  /** 이미 포맷된 표시 값 */
  display: string;
  /** 막대 길이 계산용 원값 */
  value: number;
  aside?: ReactNode;
}

/**
 * 가로 막대 목록. 이름과 값이 항상 함께 보이므로 축 라벨이 따로 필요 없고,
 * 폭이 좁아져도 이름이 wrap 될 뿐 가로 스크롤이 생기지 않는다.
 */
export function BarList({ items, alt = false }: { items: BarItem[]; alt?: boolean }) {
  const max = items.reduce((m, it) => Math.max(m, it.value), 0);
  return (
    <div className="barlist">
      {items.map((it) => (
        <div className="barrow" key={it.key}>
          <span className="barrow__name">{it.name}</span>
          <span className="barrow__value">{it.display}</span>
          <span className="barrow__track">
            <Meter ratio={max > 0 ? it.value / max : 0} label={`${it.name} ${it.display}`} alt={alt} />
          </span>
        </div>
      ))}
    </div>
  );
}
