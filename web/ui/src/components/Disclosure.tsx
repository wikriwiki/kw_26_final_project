/**
 * 접기 — 스펙 §7 progressive disclosure. **기본은 닫힘.**
 * 상세·원문·내부 산출물은 전부 이 안으로 들어간다.
 */
import type { ReactNode } from 'react';
import { ChevronRightIcon } from './Icon';

export interface DisclosureProps {
  title: string;
  /** 오른쪽 끝에 붙는 건수 등 (예: "15건") */
  meta?: string;
  children: ReactNode;
}

export function Disclosure({ title, meta, children }: DisclosureProps) {
  return (
    <details className="disclosure">
      <summary className="disclosure__summary">
        <ChevronRightIcon size={16} className="disclosure__chevron" />
        <span>{title}</span>
        {meta ? <span className="disclosure__count">{meta}</span> : null}
      </summary>
      <div className="disclosure__body">{children}</div>
    </details>
  );
}
