/**
 * 뱃지 — 색만으로 의미를 전달하지 않는다.
 * tone 을 주면 반드시 아이콘이 함께 붙고, 라벨 텍스트도 항상 있다 (스펙 §2).
 */
import type { ReactNode } from 'react';
import { AlertCircleIcon, AlertTriangleIcon, CheckCircleIcon, InfoIcon } from './Icon';

export type Tone = 'neutral' | 'ok' | 'warn' | 'danger' | 'info';

const TONE_ICON: Record<Exclude<Tone, 'neutral'>, typeof InfoIcon> = {
  ok: CheckCircleIcon,
  warn: AlertTriangleIcon,
  danger: AlertCircleIcon,
  info: InfoIcon,
};

export interface BadgeProps {
  tone?: Tone;
  children: ReactNode;
  /** 아이콘을 끄고 싶을 때만. 의미 전달용 뱃지에는 쓰지 않는다 */
  plain?: boolean;
}

export function Badge({ tone = 'neutral', children, plain = false }: BadgeProps) {
  const Icon = tone === 'neutral' ? null : TONE_ICON[tone];
  return (
    <span className={tone === 'neutral' ? 'badge' : `badge badge--${tone}`}>
      {Icon && !plain ? <Icon size={13} /> : null}
      {children}
    </span>
  );
}
