/**
 * 화면 사이로 이어지는 상태 — 스펙 §7 "시각화 페이지 진입 흐름", §9 deep-linking.
 *
 * 어떤 실행(run)과 어떤 일자를 보고 있는지는 **주소에 담는다.** 그래야 링크를 그대로
 * 공유할 수 있고, 결과 → 시각화 → 결과로 오갈 때 보던 맥락이 유지된다.
 *
 * 주소에 값이 없으면 직전에 보던 실행(sessionStorage)을 이어받는다. 사이드바로 시각화에
 * 곧장 들어온 사용자도 "아무 실행이나"가 아니라 보던 실행을 계속 보게 된다.
 */
import { useCallback, useEffect } from 'react';
import { useLocation, useNavigationType, useSearchParams } from 'react-router-dom';
import { isRunId, RUN_IDS, runs } from '../lib/fixtures';
import type { RunId } from '../lib/fixtures';

const LAST_RUN_KEY = 'simconsole.run';

function readLastRun(): RunId | null {
  try {
    const v = window.sessionStorage.getItem(LAST_RUN_KEY);
    return v && isRunId(v) ? v : null;
  } catch {
    return null;
  }
}

function writeLastRun(id: RunId) {
  try {
    window.sessionStorage.setItem(LAST_RUN_KEY, id);
  } catch {
    /* 저장 실패는 기능에 영향이 없다 */
  }
}

/** 주소의 `?run=` — 없거나 모르는 값이면 직전 실행, 그것도 없으면 첫 실행 */
export function useRunParam(): [RunId, (id: RunId) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get('run');
  const runId: RunId = raw && isRunId(raw) ? raw : (readLastRun() ?? RUN_IDS[0]);

  useEffect(() => {
    writeLastRun(runId);
  }, [runId]);

  // 주소를 정규화한다. 히스토리를 늘리지 않도록 replace 로 — 뒤로 가기는 "화면을 떠나는" 동작으로 남긴다
  useEffect(() => {
    if (raw === runId) return;
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set('run', runId);
        return next;
      },
      { replace: true },
    );
  }, [raw, runId, setParams]);

  const setRun = useCallback(
    (id: RunId) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set('run', id);
          next.delete('day'); // 실행이 바뀌면 일자는 그 실행의 기준일로 다시 정한다
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  return [runId, setRun];
}

/** 주소의 `?day=` — 그 실행에 기록이 있는 일자만 받는다. 없으면 대표 일자 */
export function useDayParam(runId: RunId): [string, string[], (day: string) => void] {
  const [params, setParams] = useSearchParams();
  const bundle = runs[runId];
  const days = bundle.days.items.map((d) => d.day);
  const raw = params.get('day');
  const day = raw && days.includes(raw) ? raw : (bundle.focusDay ?? days[0] ?? '');

  const setDay = useCallback(
    (next: string) => {
      setParams(
        (prev) => {
          const p = new URLSearchParams(prev);
          p.set('day', next);
          return p;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  return [day, days, setDay];
}

/**
 * 스크롤 위치 기억 — 스펙 §7-4, §9 `back-behavior`.
 *
 * 시각화에서 "← 결과로" 를 눌러 돌아오면 보던 자리로 되돌아온다.
 * 새로 들어온 경우(PUSH)에는 맨 위에서 시작하고, 기억해 둔 값도 버린다.
 *
 * 화면을 떠나는 순간에는 자동 기록을 잠근다. 긴 화면(결과) → 짧은 화면(지도)으로 가면
 * 브라우저가 스크롤을 0 으로 끌어내리는데, 그 값이 기억을 덮어쓰면 돌아왔을 때 맨 위로 간다.
 */
let scrollLocked = false;

/** 떠나기 직전에 지금 위치를 못박는다. 이후의 자동 기록은 돌아올 때까지 무시된다 */
export function rememberScroll(key: string) {
  try {
    window.sessionStorage.setItem(`simconsole.scroll.${key}`, String(Math.round(window.scrollY)));
  } catch {
    /* 무시 */
  }
  scrollLocked = true;
}

export function useScrollMemory(key: string) {
  const navType = useNavigationType();
  const location = useLocation();
  const asked = (location.state as { restoreScroll?: boolean } | null)?.restoreScroll === true;
  const storageKey = `simconsole.scroll.${key}`;

  useEffect(() => {
    scrollLocked = false;
    const restore = navType === 'POP' || asked;
    let stored = 0;
    try {
      stored = Number(window.sessionStorage.getItem(storageKey) ?? '0');
    } catch {
      stored = 0;
    }

    if (restore && stored > 0) {
      // 레이아웃이 잡힌 다음 한 번 더 맞춘다 (표·막대가 늦게 자리를 잡는 경우)
      window.scrollTo(0, stored);
      const id = window.requestAnimationFrame(() => window.scrollTo(0, stored));
      return () => window.cancelAnimationFrame(id);
    }

    window.scrollTo(0, 0);
    try {
      window.sessionStorage.removeItem(storageKey);
    } catch {
      /* 무시 */
    }
    return undefined;
    // 마운트 시 한 번만. 같은 화면 안에서 필터를 바꿀 때는 스크롤을 건드리지 않는다
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const save = () => {
      if (scrollLocked) return;
      try {
        window.sessionStorage.setItem(storageKey, String(Math.round(window.scrollY)));
      } catch {
        /* 무시 */
      }
    };
    window.addEventListener('scroll', save, { passive: true });
    return () => window.removeEventListener('scroll', save);
  }, [storageKey]);
}
