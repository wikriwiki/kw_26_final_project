/**
 * 실행(run)·정책 데이터 — 화면은 전부 API 에서 받는다.
 *
 * 예전에는 `web/fixtures/` 의 SEOUL7500 응답을 번들에 그대로 넣어 썼다(약 210KB).
 * 그래서 첫 화면 JS 가 무거웠고, 서버에 실제 실행이 늘어도 화면에는 그 하나만 보였다.
 * 이제 서버가 가진 실행이 곧 목록이다. 시연·테스트용 고정 응답이 필요하면 서버를
 * `SIM_USE_FIXTURES=1` 로 띄운다 — 같은 응답이 같은 경로로 온다.
 *
 * 같은 자원은 한 번만 받는다(화면을 오가도 다시 받지 않는다). 실패한 요청은 기억하지
 * 않아서 "다시 시도"가 실제로 다시 요청한다. 없는 값은 지어내지 않고 null 로 둔다.
 */
import { useEffect, useState } from 'react';
import {
  api,
  request,
  type ApiErrorShape,
  type Bottlenecks,
  type DayAggregate,
  type PolicyDetail,
  type PolicyIndexItem,
  type PolicyValidation,
  type RunDays,
  type RunDetail,
  type UnknownResource,
} from './api';
import { failed, loading, ready, type AsyncState } from './async';

/* --- CONTRACT 에 있으나 api.ts 에 아직 없던 리소스 타입 ------------------------ */

export interface SlowItem {
  aid: string;
  slow: Record<string, number>;
  s1_attempts?: number;
  s2_attempts?: number;
  tokens_in?: number;
  tokens_out?: number;
}

export interface SlowPage {
  run_id: string;
  day: string;
  available: boolean;
  reason: string | null;
  total: number | null;
  limit: number;
  sorted_by: string;
  phase_counts: Record<string, number> | null;
  items: SlowItem[];
  unknown: string[];
}

export interface FailureItem {
  aid: string;
  day: string;
  attempt: number;
  temp: number;
  error_type: string;
  error: string;
  finish_reason: string;
  raw_excerpt: string;
}

export interface FailuresPage {
  run_id: string;
  available: boolean;
  reason: string | null;
  total: number | null;
  by_day: Record<string, number> | null;
  by_error_type: Record<string, number> | null;
  limit: number;
  items: FailureItem[];
  unknown: string[];
}

export interface EventTotals {
  events: number;
  amt: number;
  policy_paid: number;
  extra_spent: number;
  coupon_eligible_events: number;
  would_buy_anyway: number;
}

export interface EventsSummary {
  run_id: string;
  available: boolean;
  reason: string | null;
  source: string | null;
  poi_summary: { poi_total: number; poi_eligible: number } | null;
  totals: EventTotals | null;
  day_type_counts: Record<string, number> | null;
  policy_paid_by_policy_id: Record<string, number> | null;
  by_day: Array<EventTotals & { day: string }> | null;
  by_l1: Array<EventTotals & { l1: string }> | null;
  by_day_l1: Array<{ day: string; l1: string; events: number; amt: number; policy_paid: number }> | null;
  /** available:false 인 실행에서는 null 이다 */
  null_only_fields: string[] | null;
  unknown: string[];
}

export interface DecileBucket {
  spend_decile: number | null;
  agents: number;
  grant_applied_today: number;
  grant_remaining_total: number;
  policy_spend_today: number;
  cm_policy_allocated_total: number;
  cm_today_total_incl_online: number;
}

export type PoliciesIndex = { total: number; items: PolicyIndexItem[]; source_dir?: string } & UnknownResource;

/** 실행 id 는 서버가 정한다. 주소에 온 값의 존재 여부는 서버가 404 로 답한다 */
export type RunId = string;

export interface RunBundle {
  detail: RunDetail;
  days: RunDays;
  /**
   * 상세를 열어 볼 대표 일자 — 기록이 남은 마지막 날. 일자 기록이 하나도 없으면 null 이고,
   * 그때는 그날의 집계·병목·느린 대상자도 null 이다.
   */
  focusDay: string | null;
  dayAggregate: DayAggregate | null;
  bottlenecks: Bottlenecks | null;
  slow: SlowPage | null;
  failures: FailuresPage;
  events: EventsSummary;
}

/* --- 한 번만 받기 ------------------------------------------------------------ */

const cache = new Map<string, Promise<unknown>>();

function once<T>(key: string, load: () => Promise<T>): Promise<T> {
  const hit = cache.get(key) as Promise<T> | undefined;
  if (hit) return hit;
  const pending = load();
  cache.set(key, pending);
  pending.catch(() => cache.delete(key));
  return pending;
}

/** 실행 목록처럼 새로 생길 수 있는 자원을 다시 받게 한다 */
export function forget(prefix: string): void {
  for (const key of [...cache.keys()]) if (key === prefix || key.startsWith(`${prefix}:`)) cache.delete(key);
}

const enc = encodeURIComponent;

export const loadRunsIndex = () => once('runs', () => api.listRuns());
export const loadPolicies = () => once<PoliciesIndex>('policies', () => api.listPolicies());
export const loadPolicyDetail = (id: string) => once<PolicyDetail>(`policy:${id}`, () => api.getPolicy(id));
export const loadPolicyValidation = (id: string) =>
  once<PolicyValidation>(`policy-validate:${id}`, () => api.validatePolicy(id));
export const loadEventsSummary = (id: RunId) =>
  once<EventsSummary>(`events:${id}`, () => request<EventsSummary>(`/runs/${enc(id)}/events/summary`));

/** 기록이 남은 마지막 날. 결제 기록이 있는 날을 먼저 고른다 */
export function pickFocusDay(days: RunDays): string | null {
  const items = days.items;
  for (let i = items.length - 1; i >= 0; i -= 1) if (items[i].metrics_rows > 0) return items[i].day;
  return items.length ? items[items.length - 1].day : null;
}

export function loadRunBundle(id: RunId): Promise<RunBundle> {
  return once(`run:${id}`, async () => {
    const [detail, days, events, failures] = await Promise.all([
      api.getRun(id),
      api.getDays(id),
      loadEventsSummary(id),
      request<FailuresPage>(`/runs/${enc(id)}/failures`),
    ]);
    const focusDay = pickFocusDay(days);
    const [dayAggregate, bottlenecks, slow] = focusDay
      ? await Promise.all([
          api.getDay(id, focusDay),
          api.getBottlenecks(id, focusDay),
          request<SlowPage>(`/runs/${enc(id)}/days/${enc(focusDay)}/slow`),
        ])
      : [null, null, null];
    return { detail, days, focusDay, dayAggregate, bottlenecks, slow, failures, events };
  });
}

/* --- 화면에서 쓰기 ------------------------------------------------------------ */

function toError(error: unknown) {
  if (error instanceof Error || typeof error === 'string') return failed(error);
  const shape = (error ?? {}) as Partial<ApiErrorShape>;
  return failed({
    message: shape.message ?? '데이터를 받지 못했습니다',
    detail: shape.detail,
    code: shape.status,
    retryable: shape.status === undefined || shape.status >= 500,
  });
}

/**
 * `key` 가 바뀔 때마다 `load` 를 부른다. 이미 받은 자원은 캐시에서 바로 온다.
 * `retry` 는 실패한 요청을 다시 보낸다(실패는 캐시에 남지 않는다).
 */
export function useLoad<T>(key: string | null, load: () => Promise<T>): AsyncState<T> & { retry: () => void } {
  const [state, setState] = useState<AsyncState<T>>(loading());
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (key === null) return;
    let live = true;
    setState(loading());
    load().then(
      (value) => live && setState(ready(value)),
      (error) => live && setState(toError(error)),
    );
    return () => {
      live = false;
    };
    // load 는 key 로 식별한다 — 화면마다 새로 만든 화살표 함수가 재요청을 일으키지 않게
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, nonce]);
  return { ...state, retry: () => setNonce((n) => n + 1) };
}

/** 서버가 "그런 실행 없음"이라고 답했는가 */
export function isNotFound(state: AsyncState<unknown>): boolean {
  return state.status === 'error' && state.error.code === 404;
}
