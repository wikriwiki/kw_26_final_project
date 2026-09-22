/**
 * 로딩 · 빈 상태 · 오류 · 안내 — 스펙 §8.
 * 세 화면이 같은 상태를 다르게 그리지 않도록 여기 한 벌만 둔다.
 */
import type { ReactNode } from 'react';
import { AlertCircleIcon, AlertTriangleIcon, InfoIcon } from './Icon';

/* --- 스켈레톤 (300ms 초과 로딩) ------------------------------------------- */

export function Skeleton({ height = 16, width = '100%' }: { height?: number; width?: string }) {
  return <span className="skeleton" style={{ display: 'block', height, width }} aria-hidden="true" />;
}

export function SkeletonText({ lines = 3 }: { lines?: number }) {
  const widths = ['100%', '92%', '76%', '84%', '68%'];
  return (
    <div className="stack-sm" role="status" aria-live="polite" aria-label="불러오는 중">
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} height={14} width={widths[i % widths.length]} />
      ))}
    </div>
  );
}

export function SkeletonCard({ rows = 3 }: { rows?: number }) {
  return (
    <div className="card">
      <Skeleton height={12} width="34%" />
      <Skeleton height={28} width="52%" />
      <SkeletonText lines={rows} />
    </div>
  );
}

/* --- 빈 상태 --------------------------------------------------------------- */

/**
 * 빈 상태에는 아이콘을 그리지 않는다 (§7b — 장식용 아이콘 금지).
 * 빈 칸에 큰 그림을 놓으면 "없다"가 아니라 "무언가 있다"로 읽힌다.
 */
export interface EmptyStateProps {
  title: string;
  /** 왜 비었는지. 추측하지 말고 아는 사실만 적는다 */
  body: string;
  /** 다음에 할 행동 */
  action?: ReactNode;
  /** 같은 행의 다른 카드 높이에 맞춰 늘어난 카드 안에서 세로 가운데 정렬 */
  fill?: boolean;
}

export function EmptyState({ title, body, action, fill = false }: EmptyStateProps) {
  return (
    <div className={fill ? 'notice notice--fill' : 'notice'}>
      <p className="notice__title">{title}</p>
      <p className="notice__body">{body}</p>
      {action}
    </div>
  );
}

/* --- 오류 ------------------------------------------------------------------ */

export interface ErrorStateProps {
  title: string;
  /** 원인 + 해결 방법 */
  body: string;
  action?: ReactNode;
  /** 접어 두는 원문 */
  detail?: string;
}

export function ErrorState({ title, body, action, detail }: ErrorStateProps) {
  return (
    <div className="notice notice--error" role="alert">
      <span className="notice__icon">
        <AlertCircleIcon size={22} />
      </span>
      <p className="notice__title">{title}</p>
      <p className="notice__body">{body}</p>
      {action}
      {detail ? (
        <details className="disclosure" style={{ width: '100%' }}>
          <summary className="disclosure__summary">원문 보기</summary>
          <div className="disclosure__body">
            <pre className="code">{detail}</pre>
          </div>
        </details>
      ) : null}
    </div>
  );
}

/* --- 인라인 안내 ----------------------------------------------------------- */

export function Callout({ tone = 'info', children }: { tone?: 'info' | 'warn'; children: ReactNode }) {
  return (
    <p className={tone === 'warn' ? 'callout callout--warn' : 'callout'}>
      {tone === 'warn' ? <AlertTriangleIcon size={18} /> : <InfoIcon size={18} />}
      <span>{children}</span>
    </p>
  );
}
