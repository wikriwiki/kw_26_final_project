/**
 * 사이드바 상태 — 스펙 §6.
 *
 * - 확장 여부는 localStorage 에 남는다 (새로고침해도 유지).
 * - 1024px 이상은 push, 미만은 overlay. 모드는 matchMedia 로 실시간 추적한다.
 * - 상태는 `document.documentElement[data-nav]` 에도 반영한다. CSS 가 그 한 곳만 보면 되도록.
 *
 * `suppress()` 는 시각화처럼 "레일로 진입해야 하는" 화면을 위한 것이다 (§7).
 * 저장된 선호값을 건드리지 않고 이번 화면에서만 접어 둔다 — 그 화면을 벗어나면
 * 원래 열어 두던 사용자는 다시 열린 상태로 돌아온다 (§9 state-preservation).
 */
import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'simconsole.nav.expanded';
const PUSH_QUERY = '(min-width: 1024px)';

function readStored(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

export interface NavState {
  expanded: boolean;
  /** true 면 push 모드(1024px 이상), false 면 overlay + 스크림 */
  push: boolean;
  toggle: () => void;
  close: () => void;
  /** 이번 화면에서만 레일로 접어 둔다. 저장된 선호값은 그대로 남는다 */
  suppress: (on: boolean) => void;
}

export function useNav(): NavState {
  // 사용자가 고른 값 — 저장된다
  const [pref, setPref] = useState<boolean>(() =>
    typeof window === 'undefined' ? false : readStored(),
  );
  // 화면이 강제로 접은 값 — 저장하지 않는다
  const [suppressed, setSuppressed] = useState(false);
  const [push, setPush] = useState<boolean>(() =>
    typeof window === 'undefined' ? true : window.matchMedia(PUSH_QUERY).matches,
  );

  const expanded = pref && !suppressed;

  useEffect(() => {
    const mq = window.matchMedia(PUSH_QUERY);
    const onChange = (e: MediaQueryListEvent) => setPush(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.nav = expanded ? 'expanded' : 'rail';
  }, [expanded]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, String(pref));
    } catch {
      /* 저장 실패는 기능에 영향이 없다 — 이번 세션에만 유지된다 */
    }
  }, [pref]);

  // 사용자가 직접 여닫으면 강제 접힘은 풀린다. 시각화 화면에서도 메뉴는 열 수 있어야 한다
  const close = useCallback(() => {
    setPref(false);
    setSuppressed(false);
  }, []);

  const toggle = useCallback(() => {
    setPref((v) => (suppressed ? true : !v));
    setSuppressed(false);
  }, [suppressed]);

  const suppress = useCallback((on: boolean) => setSuppressed(on), []);

  // overlay 모드에서만 Esc 로 닫는다. push 모드에서 닫히면 오히려 놀란다
  useEffect(() => {
    if (push || !expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [push, expanded, close]);

  return { expanded, push, toggle, close, suppress };
}
