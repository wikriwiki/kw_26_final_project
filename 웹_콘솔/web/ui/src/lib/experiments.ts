/**
 * 실험 콘솔 API — `web/api/experiment_routes.py` 의 응답 모양.
 *
 * 실험(experiment)은 여러 팔(arm)을 가진다. 분기 실험은 공통 사전 팔에서 OFF/ON 팔이 갈라진다.
 * 값이 없을 때 서버는 null 을 준다. 화면은 그걸 0 으로 바꾸지 않는다.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { request } from './api';
import type { ApiErrorShape } from './api';

export type ArmRole = 'single' | 'shared_pre' | 'control' | 'treatment';
export type ArmPhase = 'not_synced' | 'not_started' | 'running' | 'complete';

export interface ArmDay {
  day: string;
  present: boolean;
  rows?: number;
  ok?: number;
  skipped?: number;
  complete?: boolean;
  integrity_failures?: number;
  partial_tail?: boolean;
  night_done: boolean;
  backup_done: boolean;
}

export interface ArmStatus {
  arm_id: string;
  role: ArmRole;
  label: string;
  policy: string;
  start: string;
  days: number;
  parent: string | null;
  phase: ArmPhase;
  manifest: {
    status: string | null;
    model: string | null;
    workers: number | null;
    cohort_size: number;
    assignment_seed: number | null;
    simulation_seed: number | null;
    interpretation: string | null;
  } | null;
  days_present: number;
  days_complete: number;
  current_day: ArmDay | null;
  skipped_total: number;
  integrity_failures: number;
  day_status?: ArmDay[];
  synced_at: string | null;
}

export interface ExperimentSummary {
  experiment_id: string;
  title: string;
  kind: string;
  summary: string;
  interpretation: string;
  window: { start?: string; days?: number; effective_date?: string };
  cohort_size: number;
  arms: ArmStatus[];
  analysis: {
    facilities?: Record<string, string>;
    districts?: Record<string, string>;
    limitations?: string[];
  };
  source_branch: string | null;
  origin: 'repo' | 'local';
  remote_profile: string | null;
  mirror: boolean;
  bundle_present?: boolean;
  /** 시민 문답에서 누를 질문 예시 */
  questions?: string[];
  /** 실측 연구와의 비교 (평가_통합성능에서 옮김) */
  validation?: Validation | null;
}

export interface ValidationRow {
  label: string;
  sim_pct: number;
  sim_ci95_pct: [number, number] | null;
  truth_pct: number;
  source: string;
}

export interface Validation {
  basis: string;
  note: string;
  rows: ValidationRow[];
}

export interface ExperimentList {
  total: number;
  items: ExperimentSummary[];
  problems: Array<{ file: string; error: string }>;
  unknown: string[];
}

export interface Counted {
  revenue_krw: number;
  payment_count: number;
}

export interface DailyPoint {
  day: string;
  partial_tail: boolean;
  integrity_failures: number;
  status: Record<string, number>;
  unique_agents: number;
  totals: Record<string, number>;
  means: Record<string, number>;
  by_category: Record<string, { count: number; amount: number }>;
  target?: { total: Counted; groups: Record<string, Counted> };
}

export interface DailySeries {
  arm_id: string;
  label: string;
  role: ArmRole;
  points: DailyPoint[];
}

export interface Comparison {
  experiment_id: string;
  available: boolean;
  status?: 'interim_preview' | 'complete_preview';
  reason: string | null;
  rules?: string;
  pre?: { days: number; totals: Record<string, Counted> };
  post?: {
    days: string[];
    observed_paired_agent_days: number;
    excluded_agent_days: number;
    totals: { off: Record<string, Counted>; on: Record<string, Counted> };
    daily: Array<{ day: string; off: Counted; on: Counted }>;
  };
  comparisons?: Array<{
    group: string;
    metric: 'revenue_krw' | 'payment_count';
    off: number;
    on: number;
    on_minus_off: number;
    percent_change: number | null;
    status: string;
  }>;
  limitations?: string[];
  /* 정책 실험(policy_paired) — 같은 시민·같은 날의 정책 있음 − 없음 */
  kind?: 'policy_paired';
  days?: string[];
  agents?: number;
  paired_agent_days?: number;
  rows?: PairedRow[];
  wallet?: { on_total: number; per_agent_day: number };
  daily?: Array<{ day: string; pairs: number; off: number; on: number; effect: number | null }>;
  method?: { pairing: string; effect: string; ci: string };
}

export interface PairedRow {
  key: string;
  label: string;
  group: 'total' | 'category';
  off: number;
  on: number;
  off_per_agent_day: number;
  on_per_agent_day: number;
  effect: number | null;
  ci95: [number, number] | null;
  significant: boolean;
}

export interface RosterAgent {
  id: string;
  gu?: string | null;
  dong?: string | null;
  district_code?: string | null;
  age?: number | null;
  age_group?: string | null;
  sex?: string | null;
  job?: string | null;
  income_level?: string | null;
  life_stage?: string | null;
  daily_spending_weekday?: number | null;
  smoking_status?: string | null;
}

export interface Roster {
  experiment_id: string;
  source: 'bundle' | 'metrics';
  total: number;
  items: RosterAgent[];
}

export interface AgentReceipt {
  time: string | null;
  category: string | null;
  amount: number;
  own_paid: number;
  status: string | null;
  poi_id: string | null;
  event_id: string | null;
}

export interface AgentDay {
  day: string;
  status: string;
  receipts: AgentReceipt[];
  spend: number | null;
  spend_incl_online: number | null;
  mood: number | null;
  fatigue: number | null;
  avg_sat: number | null;
  balance: number | null;
  evidence: boolean;
  skip_kind?: string;
  no_smoking?: {
    policy_active: boolean | null;
    smoking_status: string | null;
    payment_count: number | null;
    revenue_krw: number | null;
    by_poi: Array<{ poi_id: string; facility: string; district_code: string; revenue_krw: number; payment_count: number }>;
  };
}

export interface AgentDetail {
  experiment_id: string;
  agent: RosterAgent;
  arms: Array<{ arm_id: string; label: string; role: ArmRole; days: AgentDay[] }>;
}

export interface Citation {
  n: number;
  evidence_id: string;
  day: string | null;
  kind: string;
  text: string;
}

export interface InterviewReply {
  experiment_id: string;
  arm_id: string;
  agent_id: string;
  through_day: string;
  answer: string;
  model_label: string | null;
  citations: Citation[];
  warnings: string[];
  unsupported_numbers: string[];
  packet: {
    lineage: string[];
    days: number;
    missing_days: string[];
    skipped_days: string[];
    exposed_items: number;
    total_items: number;
    journal_days_used: string[];
    journal_days_missing: string[];
  };
  evidence_fetch: { state: string; fetched?: number; error?: string } | null;
  console_only: boolean;
}

export interface GroupDescribe {
  agents: number;
  observed_agent_days: number;
  skipped_agent_days: number;
  spend_per_observed_day: number | null;
  receipts_per_observed_day: number | null;
  target_payments: number;
  target_revenue_krw: number;
  agents_with_target_visit: number;
  mood_mean: number | null;
}

export interface GroupStats {
  experiment_id: string;
  arm_id: string;
  through_day: string | null;
  days: number;
  overall: GroupDescribe;
  breakdowns: Record<string, Record<string, GroupDescribe>>;
  notes: string[];
}

export interface GroupJob {
  job_id: string;
  state: 'running' | 'done';
  question: string;
  matched: number;
  sampled: string[];
  done: number;
  results: Array<{
    agent_id: string;
    gu?: string | null;
    sex?: string | null;
    age_group?: string | null;
    smoking_status?: string | null;
    answer?: string;
    error?: string;
    citations?: Citation[];
    warnings?: string[];
    through_day?: string;
  }>;
  summary: { text?: string; model_label?: string; verified?: boolean; error?: string } | null;
}

export interface SyncStatus {
  state: 'idle' | 'running' | 'done' | 'failed';
  message?: string;
  error?: string | null;
  started_at?: string;
  finished_at?: string;
  remote_configured: boolean;
  remote_profile: string | null;
  report?: { bytes_received?: number; arms?: Record<string, { downloaded: number; appended: number; unchanged: number }> } | null;
}

export interface InterviewStatus {
  ready: boolean;
  backend: string;
  model_label: string;
  reason: string | null;
}

export type GroupFilters = Partial<Record<'gu' | 'sex' | 'age_group' | 'smoking_status' | 'income_level' | 'life_stage', string[]>>;

const enc = encodeURIComponent;
const post = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) });

export const experimentsApi = {
  list: () => request<ExperimentList>('/experiments'),
  detail: (id: string) => request<ExperimentSummary>(`/experiments/${enc(id)}`),
  daily: (id: string) => request<{ experiment_id: string; kind: string; series: DailySeries[] }>(`/experiments/${enc(id)}/daily`),
  comparison: (id: string) => request<Comparison>(`/experiments/${enc(id)}/comparison`),
  roster: (id: string) => request<Roster>(`/experiments/${enc(id)}/agents`),
  agent: (id: string, agentId: string) => request<AgentDetail>(`/experiments/${enc(id)}/agents/${enc(agentId)}`),
  interview: (id: string, agentId: string, body: { arm_id: string; question: string; through_day?: string | null; history?: Array<{ role: string; content: string }> }) =>
    request<InterviewReply>(`/experiments/${enc(id)}/agents/${enc(agentId)}/interview`, post(body)),
  groupStats: (id: string, body: { arm_id: string; filters: GroupFilters; through_day?: string | null }) =>
    request<GroupStats>(`/experiments/${enc(id)}/query/stats`, post(body)),
  groupInterview: (id: string, body: { arm_id: string; filters: GroupFilters; question: string; sample: number; seed: number; through_day?: string | null }) =>
    request<GroupJob>(`/experiments/${enc(id)}/query/interviews`, post(body)),
  groupInterviewStatus: (id: string, jobId: string) => request<GroupJob>(`/experiments/${enc(id)}/query/interviews/${enc(jobId)}`),
  syncStatus: (id: string) => request<SyncStatus>(`/experiments/${enc(id)}/sync`),
  syncStart: (id: string) => request<SyncStatus>(`/experiments/${enc(id)}/sync`, { method: 'POST' }),
  interviewStatus: () => request<InterviewStatus>('/interview/status'),
};

export interface Resource<T> {
  data: T | null;
  error: ApiErrorShape | null;
  loading: boolean;
  reload: () => void;
}

/** 한 API 를 부르고 상태를 들고 있는다. 키가 바뀌면 다시 부르고, 늦게 온 예전 응답은 버린다. */
export function useResource<T>(load: () => Promise<T>, key: string): Resource<T> {
  const [state, setState] = useState<{ data: T | null; error: ApiErrorShape | null; loading: boolean }>({
    data: null,
    error: null,
    loading: true,
  });
  const [tick, setTick] = useState(0);
  const loader = useRef(load);
  loader.current = load;

  useEffect(() => {
    let alive = true;
    setState((prev) => ({ ...prev, loading: true, error: null }));
    loader.current().then(
      (data) => alive && setState({ data, error: null, loading: false }),
      (error: ApiErrorShape) => alive && setState({ data: null, error, loading: false }),
    );
    return () => {
      alive = false;
    };
  }, [key, tick]);

  const reload = useCallback(() => setTick((n) => n + 1), []);
  return { ...state, reload };
}

export const ROLE_LABEL: Record<ArmRole, string> = {
  single: '실행',
  shared_pre: '정책 전',
  control: '정책 없음',
  treatment: '정책 있음',
};

export const PHASE_TEXT: Record<ArmPhase, string> = {
  not_synced: '아직 받지 않음',
  not_started: '시작 전',
  running: '진행 중',
  complete: '완료',
};

export const SMOKING_TEXT: Record<string, string> = {
  smoker: '흡연',
  non_smoker: '비흡연',
  unknown: '미상',
};
