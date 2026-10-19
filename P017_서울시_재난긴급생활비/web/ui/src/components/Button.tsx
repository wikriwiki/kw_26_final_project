/**
 * 버튼 — 스펙 §8.
 * - 최소 44×44 히트 영역은 CSS(`--hit`)가 보장한다.
 * - 비동기 처리 중에는 비활성 + 스피너 (`busy`).
 * - 화면당 primary 는 한 개만 쓴다.
 *
 * ref 를 넘길 수 있다 — 패널을 닫을 때 포커스를 열었던 버튼으로 되돌리기 위해서다 (§8 escape-routes).
 */
import { forwardRef } from 'react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { LoaderIcon } from './Icon';

type Variant = 'primary' | 'secondary' | 'ghost';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  /** 처리 중. 자동으로 disabled + 스피너 + aria-busy */
  busy?: boolean;
  /** 처리 중에 대체할 라벨. 없으면 원래 라벨을 유지한다 */
  busyLabel?: string;
  icon?: ReactNode;
  block?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = 'secondary',
    busy = false,
    busyLabel,
    icon,
    block = false,
    disabled,
    children,
    className,
    type = 'button',
    ...rest
  },
  ref,
) {
  const classes = ['btn', `btn--${variant}`, block ? 'btn--block' : '', className ?? '']
    .filter(Boolean)
    .join(' ');

  return (
    <button
      ref={ref}
      type={type}
      className={classes}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      {...rest}
    >
      {busy ? <LoaderIcon size={18} className="btn__spinner" /> : icon}
      <span>{busy && busyLabel ? busyLabel : children}</span>
    </button>
  );
});
