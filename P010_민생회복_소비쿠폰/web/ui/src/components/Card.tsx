/**
 * 카드 · 지표.
 * 카드는 12열 그리드의 열 경계에만 놓인다 — 임의 width 를 받지 않는다.
 * 같은 행의 카드는 CSS 의 align-items: stretch 로 높이가 맞는다.
 */
import type { ReactNode } from 'react';
import { EMPTY } from '../lib/format';

export interface CardProps {
  title?: string;
  /** 제목 옆 보조 설명 또는 뱃지 */
  aside?: ReactNode;
  /** 제목 아래 한 줄 안내 */
  note?: string;
  children?: ReactNode;
  /** 표처럼 가장자리까지 채우는 내용일 때 */
  flush?: boolean;
  className?: string;
}

export function Card({ title, aside, note, children, flush = false, className }: CardProps) {
  return (
    <section className={['card', flush ? 'card--flush' : '', className ?? ''].filter(Boolean).join(' ')}>
      {title || aside ? (
        <header className="card__head">
          {title ? <h3 className="card__title">{title}</h3> : <span />}
          {aside}
        </header>
      ) : null}
      {note ? <p className="card__note">{note}</p> : null}
      {children}
    </section>
  );
}

/**
 * 지표 — 값 24px / 레이블 12px muted (§7b).
 * 상자에 담지 않는다. 같은 행의 지표들은 그리드 정렬과 간격만으로 묶인다.
 */
export interface StatProps {
  label: string;
  /** 이미 포맷된 문자열. 값이 없으면 null 을 넘긴다 — 0 을 지어내지 않는다 */
  value: string | null;
  unit?: string;
  hint?: string;
  /** 값이 없을 때 왜 없는지 */
  unknownReason?: string;
  /** 그리드 열 클래스 (예: c3) */
  className?: string;
}

export function Stat({ label, value, unit, hint, unknownReason, className }: StatProps) {
  const missing = value === null || value === EMPTY;
  return (
    <div className={['stat', className ?? ''].filter(Boolean).join(' ')}>
      <span className="stat__label">{label}</span>
      <span className={missing ? 'stat__value stat__value--unknown' : 'stat__value'}>
        {missing ? '알 수 없음' : value}
        {!missing && unit ? <span className="stat__unit">{unit}</span> : null}
      </span>
      {missing && unknownReason ? <span className="stat__hint">{unknownReason}</span> : null}
      {!missing && hint ? <span className="stat__hint">{hint}</span> : null}
    </div>
  );
}

/** 지표 4개를 한 행에 — 12열을 3열씩 나눠 쓴다 */
export function StatRow({ children }: { children: ReactNode }) {
  return <div className="grid statrow">{children}</div>;
}
