/**
 * 화면을 채우는 데이터.
 *
 * 전부 `web/fixtures/` 의 실물 응답(= 실측 run 3종 + 실측 정책 4종)이다.
 * 이 파일은 읽기만 한다 — 픽스처를 고치지 않고, 없는 값을 지어내지도 않는다.
 * `?raw` 로 문자열째 받아 파싱하므로 타입은 아래 인터페이스가 유일한 계약이다.
 *
 * 이번 단계는 디자인 전용이라 API 연동이 없다. 나중에 `lib/api.ts` 로 갈아끼울 때
 * 화면이 기대하는 모양이 바뀌지 않도록, 타입은 CONTRACT.md 의 리소스 스키마를 따른다.
 */
import type {
  Bottlenecks,
  DayAggregate,
  PolicyDetail,
  PolicyIndexItem,
  PolicyValidation,
  RunDays,
  RunDetail,
  RunsIndex,
} from './api';

import policiesIndexRaw from '../../../fixtures/policies.index.json?raw';
import p008DetailRaw from '../../../fixtures/policy.P008.detail.json?raw';
import p009DetailRaw from '../../../fixtures/policy.P009.detail.json?raw';
import p010DetailRaw from '../../../fixtures/policy.P010.detail.json?raw';
import p011DetailRaw from '../../../fixtures/policy.P011.detail.json?raw';
import p008ValidateRaw from '../../../fixtures/policy.P008.validate.json?raw';
import p009ValidateRaw from '../../../fixtures/policy.P009.validate.json?raw';
import p010ValidateRaw from '../../../fixtures/policy.P010.validate.json?raw';
import p011ValidateRaw from '../../../fixtures/policy.P011.validate.json?raw';

import runsIndexRaw from '../../../fixtures/runs.index.json?raw';
import baseDetailRaw from '../../../fixtures/run.BASE.detail.json?raw';
import finalDetailRaw from '../../../fixtures/run.FINAL.detail.json?raw';
import rescueDetailRaw from '../../../fixtures/run.BASE7500.detail.json?raw';
import baseDaysRaw from '../../../fixtures/run.BASE.days.json?raw';
import finalDaysRaw from '../../../fixtures/run.FINAL.days.json?raw';
import rescueDaysRaw from '../../../fixtures/run.BASE7500.days.json?raw';
import baseDayRaw from '../../../fixtures/run.BASE.day.2025-07-21.json?raw';
import finalDayRaw from '../../../fixtures/run.FINAL.day.2025-08-17.json?raw';
import rescueDayRaw from '../../../fixtures/run.BASE7500.day.2025-07-14.json?raw';
import baseBottlenecksRaw from '../../../fixtures/run.BASE.day.2025-07-21.bottlenecks.json?raw';
import finalBottlenecksRaw from '../../../fixtures/run.FINAL.day.2025-08-17.bottlenecks.json?raw';
import rescueBottlenecksRaw from '../../../fixtures/run.BASE7500.day.2025-07-14.bottlenecks.json?raw';
import baseSlowRaw from '../../../fixtures/run.BASE.day.2025-07-21.slow.json?raw';
import finalSlowRaw from '../../../fixtures/run.FINAL.day.2025-08-17.slow.json?raw';
import rescueSlowRaw from '../../../fixtures/run.BASE7500.day.2025-07-14.slow.json?raw';
import baseFailuresRaw from '../../../fixtures/run.BASE.failures.json?raw';
import finalFailuresRaw from '../../../fixtures/run.FINAL.failures.json?raw';
import rescueFailuresRaw from '../../../fixtures/run.BASE7500.failures.json?raw';
import baseEventsRaw from '../../../fixtures/run.BASE.events.summary.json?raw';
import finalEventsRaw from '../../../fixtures/run.FINAL.events.summary.json?raw';
import rescueEventsRaw from '../../../fixtures/run.BASE7500.events.summary.json?raw';
import expDetailRaw from '../../../fixtures/run.EXP7500.detail.json?raw';
import expDaysRaw from '../../../fixtures/run.EXP7500.days.json?raw';
import expDayRaw from '../../../fixtures/run.EXP7500.day.2025-07-21.json?raw';
import expBottlenecksRaw from '../../../fixtures/run.EXP7500.day.2025-07-21.bottlenecks.json?raw';
import expSlowRaw from '../../../fixtures/run.EXP7500.day.2025-07-21.slow.json?raw';
import expFailuresRaw from '../../../fixtures/run.EXP7500.failures.json?raw';
import expEventsRaw from '../../../fixtures/run.EXP7500.events.summary.json?raw';

/* --- CONTRACT 에 있으나 api.ts 에 아직 없는 리소스 타입 -------------------- */

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
  /** available:false 인 실행에서는 null 이다 (실측 run.BASE7500.events.summary.json) */
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

const parse = <T,>(raw: string): T => JSON.parse(raw) as T;

/* --- 정책 ----------------------------------------------------------------- */

export const policiesIndex = parse<{ total: number; items: PolicyIndexItem[]; source_dir: string }>(
  policiesIndexRaw,
);

export const policyDetails: Record<string, PolicyDetail> = {
  P008: parse<PolicyDetail>(p008DetailRaw),
  P009: parse<PolicyDetail>(p009DetailRaw),
  P010: parse<PolicyDetail>(p010DetailRaw),
  P011: parse<PolicyDetail>(p011DetailRaw),
};

export const policyValidations: Record<string, PolicyValidation> = {
  P008: parse<PolicyValidation>(p008ValidateRaw),
  P009: parse<PolicyValidation>(p009ValidateRaw),
  P010: parse<PolicyValidation>(p010ValidateRaw),
  P011: parse<PolicyValidation>(p011ValidateRaw),
};

/* --- 실행(run) ------------------------------------------------------------ */

export const runsIndex = parse<RunsIndex>(runsIndexRaw);

export type RunId = 'BASE' | 'FINAL' | 'BASE7500' | 'EXP7500';

export interface RunBundle {
  detail: RunDetail;
  days: RunDays;
  /** 상세를 열어 볼 대표 일자 — 픽스처가 존재하는 일자만 고른다 */
  focusDay: string;
  dayAggregate: DayAggregate;
  bottlenecks: Bottlenecks;
  slow: SlowPage;
  failures: FailuresPage;
  events: EventsSummary;
}

export const runs: Record<RunId, RunBundle> = {
  BASE: {
    detail: parse<RunDetail>(baseDetailRaw),
    days: parse<RunDays>(baseDaysRaw),
    focusDay: '2025-07-21',
    dayAggregate: parse<DayAggregate>(baseDayRaw),
    bottlenecks: parse<Bottlenecks>(baseBottlenecksRaw),
    slow: parse<SlowPage>(baseSlowRaw),
    failures: parse<FailuresPage>(baseFailuresRaw),
    events: parse<EventsSummary>(baseEventsRaw),
  },
  FINAL: {
    detail: parse<RunDetail>(finalDetailRaw),
    days: parse<RunDays>(finalDaysRaw),
    focusDay: '2025-08-17',
    dayAggregate: parse<DayAggregate>(finalDayRaw),
    bottlenecks: parse<Bottlenecks>(finalBottlenecksRaw),
    slow: parse<SlowPage>(finalSlowRaw),
    failures: parse<FailuresPage>(finalFailuresRaw),
    events: parse<EventsSummary>(finalEventsRaw),
  },
  BASE7500: {
    detail: parse<RunDetail>(rescueDetailRaw),
    days: parse<RunDays>(rescueDaysRaw),
    focusDay: '2025-07-14',
    dayAggregate: parse<DayAggregate>(rescueDayRaw),
    bottlenecks: parse<Bottlenecks>(rescueBottlenecksRaw),
    slow: parse<SlowPage>(rescueSlowRaw),
    failures: parse<FailuresPage>(rescueFailuresRaw),
    events: parse<EventsSummary>(rescueEventsRaw),
  },
  EXP7500: {
    detail: parse<RunDetail>(expDetailRaw),
    days: parse<RunDays>(expDaysRaw),
    focusDay: '2025-07-21',
    dayAggregate: parse<DayAggregate>(expDayRaw),
    bottlenecks: parse<Bottlenecks>(expBottlenecksRaw),
    slow: parse<SlowPage>(expSlowRaw),
    failures: parse<FailuresPage>(expFailuresRaw),
    events: parse<EventsSummary>(expEventsRaw),
  },
};

export const RUN_IDS: RunId[] = ['BASE', 'FINAL', 'BASE7500', 'EXP7500'];

export function isRunId(value: string): value is RunId {
  return (RUN_IDS as string[]).includes(value);
}
