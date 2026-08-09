export interface ApiErrorShape {
  message: string;
  status?: number;
  detail?: string;
}

export interface UnknownResource {
  unknown: string[];
}

export interface PolicyIndexItem extends UnknownResource {
  id: string;
  file: string;
  name: string;
  type: string;
  announce_date: string | null;
  effective_from: string | null;
  effective_until: string | null;
  target_districts: string[] | null;
  benefit_categories: string[] | null;
  grant_key: string | null;
  grant_key_effective: string;
  grant_key_source: 'file' | 'default';
  poi_restricted: boolean;
  has_decile_grants: boolean;
  has_income_grants: boolean;
}

export interface PolicyPayload {
  id: string;
  name: string;
  type: string;
  description: string;
  announce_date?: string | null;
  effective_from: string;
  effective_until: string;
  target_districts?: string[];
  benefit_categories?: string[];
  poi_restricted?: boolean;
  grant_key?: string;
  income_grants?: Record<string, number>;
  decile_grants?: Record<string, number>;
  excluded_income?: string[];
  excluded_deciles?: string[];
  [key: string]: unknown;
}

export interface PolicyDetail extends UnknownResource {
  file: string;
  source_dir: string;
  policy: PolicyPayload;
  grant_key_effective: string;
  grant_key_source: 'file' | 'default';
}

export interface PreflightCheck {
  grade: 'pass' | 'warn' | 'fail';
  message: string;
}

export interface PolicyValidation extends UnknownResource {
  policy_id: string;
  exit_code: number;
  ok: boolean;
  verdict: string | null;
  counts: { pass?: number; warn?: number; fail?: number };
  checks: PreflightCheck[];
  prompt_preview: string;
  prompt_preview_persona: string | null;
  db_wiring_checked: boolean;
  stderr: string;
  command: string[];
}

export interface RunIndexItem extends UnknownResource {
  run_id: string;
  root: string;
  status: 'completed' | 'incomplete';
  first_day: string | null;
  last_day: string | null;
  days_present: number;
  days_planned: number | null;
  agents_target: number | null;
  completed_at: string | null;
  artifacts: Record<string, boolean>;
}

export interface RunsIndex extends UnknownResource {
  total: number;
  items: RunIndexItem[];
}

export interface RunDetail extends UnknownResource {
  run_id: string;
  root: string;
  status: 'completed' | 'incomplete';
  artifacts: Record<string, boolean>;
  days_present: string[];
  days_with_timing: string[];
  days_with_done_checkpoint: string[];
  days_with_failed_checkpoint: string[];
  plan: {
    source: string | null;
    start_day: string | null;
    planned_days: number | null;
    agents_target: number | null;
    workers: number | null;
  };
  completed_at: string | null;
  updated_at: string | null;
  log_hint: Record<string, unknown> | null;
  day_summaries: Array<Record<string, unknown>>;
}

export interface RunDay {
  day: string;
  agents_ok: number;
  agents_error: number;
  metrics_rows: number;
  counts_source: 'status_scan' | 'metrics_aggregate';
  checkpoint_done_count: number | null;
  checkpoint_failed_count: number | null;
  agents_target: number | null;
  progress_ratio: number | null;
  day_complete: boolean;
  elapsed_sec: number | null;
  agent_elapsed_sec: number | null;
  night2_elapsed_sec: number | null;
  timing_report_present: boolean;
  policy_payment: Record<string, unknown> | null;
  metrics_bytes: number;
  unknown: string[];
}

export interface RunDays extends UnknownResource {
  run_id: string;
  total: number;
  items: RunDay[];
}

export interface DayAggregate extends UnknownResource {
  run_id: string;
  day: string;
  source_file: string;
  source_bytes: number;
  aggregated_server_side: boolean;
  rows: number;
  status_counts: Record<string, number>;
  agents_ok: number;
  agents_error: number;
  sums: Record<string, number>;
  distributions: Record<string, { n: number; total: number; avg: number; p50: number; p95: number; max: number }>;
  fallback_counts: Record<string, number>;
  attempt_counts: Record<string, number>;
  llm_call_totals: Record<string, number>;
  cache: { persona_hit_rate: number | null; policy_hit_rate: number | null };
  by_spend_decile: Array<Record<string, number | null>>;
  spend_decile_unknown_agents: number;
  error_samples: Array<Record<string, unknown>>;
  _fields_not_aggregated: string[];
}

export interface Bottlenecks extends UnknownResource {
  run_id: string;
  day: string;
  available: boolean;
  degraded: boolean;
  reason: string | null;
  /** available:false 일 때 무엇까지 재계산했는지 (CONTRACT §3.5) */
  degraded_note: string | null;
  fallback_source: string | null;
  bottleneck_rank: Array<Record<string, unknown>> | null;
  fallback_rank: Array<Record<string, unknown>> | null;
  cache: Record<string, unknown> | null;
  policy_payment: Record<string, unknown> | null;
  counters: Record<string, unknown> | null;
  timings: Record<string, unknown> | null;
  agents_ok: number | null;
  agents_error: number | null;
}

export interface FailedPage extends UnknownResource {
  run_id: string;
  day: string;
  available: boolean;
  reason: string | null;
  total: number | null;
  items: Array<Record<string, unknown>>;
}

export interface LockStatus extends UnknownResource {
  locked: boolean;
  owner: Record<string, unknown> | null;
  process_alive?: boolean | null;
  stale?: boolean;
}

export interface ReportAnalysis extends UnknownResource {
  id: string;
  label: string;
  description: string;
  applicable: boolean;
  disabled_reason: string | null;
}

/** 보고서 v2 의 절 하나. `required` 인 절은 사용자가 꺼도 서버가 되돌린다 */
export interface ReportSection extends ReportAnalysis {
  required?: boolean;
}

export interface ReportEngineInfo {
  id: 'v2' | 'dasol' | string;
  label: string;
  description: string;
  available: boolean;
  reason: string | null;
}

export interface LlmStatus extends UnknownResource {
  provider: 'gemini' | 'openai' | 'none' | string;
  configured: boolean;
  model: string;
  reason: string | null;
  env_files: string[];
  expects: Record<string, string[]>;
  key_present: Record<string, boolean>;
  /** `POST /api/llm/ping` 응답에만 있다 */
  reachable?: boolean;
  sample?: string | null;
  error?: string | null;
  latency_ms?: number | null;
  checked_at?: string;
}

export interface ReportCatalog extends UnknownResource {
  run: {
    run_id: string;
    status: 'completed' | 'incomplete';
    root: string | null;
    days_present: string[];
    plan: Record<string, unknown>;
    policy_id?: string | null;
    policy_sha256?: string | null;
    manifest_sha256?: string | null;
    unknown: string[];
  };
  policy: {
    id: string;
    file: string | null;
    effective_from: string | null;
    name: string;
    unknown: string[];
  };
  analyses: ReportAnalysis[];
  v2_sections: ReportSection[];
  v2_required: string[];
  engines: ReportEngineInfo[];
  engine_v2: {
    available: boolean;
    run_root: string | null;
    events_present: boolean;
    events_bytes?: number;
    reason: string | null;
    unknown: string[];
  };
  llm: LlmStatus;
  report_artifacts: Array<{ path: string; bytes: number }>;
  report_lock: LockStatus;
  snapshot: {
    ready: boolean;
    run_id: string;
    root_relative: string | null;
    days: string[];
    source_count: number;
    reason?: string;
    unknown: string[];
  };
  engine: {
    configured: boolean;
    snapshot_bound: boolean;
    binding_declared?: boolean;
    binding_verified?: boolean;
    verification_level?: 'environment_only' | 'verified' | 'unconfigured' | string;
    binding_run_id?: string | null;
    uri: string;
    reason: string | null;
    unknown: string[];
  };
}

export interface ReportJob extends UnknownResource {
  job_id: string;
  state: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  run_id: string;
  policy_id: string;
  start: string;
  days: number;
  policy_from: string | null;
  analyses: string[];
  include_interview: boolean;
  output_path: string;
  logs: string[];
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  artifacts: string[];
  exit_code?: number;
  engine?: 'v2' | 'dasol' | string;
  use_llm?: boolean;
  /** v2 엔진만. 일관성 검사가 모두 통과했는가 */
  consistent?: boolean;
  snapshot_manifest_path?: string;
  snapshot_id?: string;
  policy_snapshot_path?: string;
  policy_sha256?: string;
}

const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'API에 연결할 수 없습니다';
    throw { message, detail: String(error) } satisfies ApiErrorShape;
  }
  const raw = await response.text();
  let body: unknown = null;
  try {
    body = raw ? JSON.parse(raw) : null;
  } catch {
    body = raw;
  }
  if (!response.ok) {
    const record = body && typeof body === 'object' ? (body as Record<string, unknown>) : {};
    throw {
      message: typeof record.error === 'string' ? record.error : `API 오류 ${response.status}`,
      status: response.status,
      detail: typeof record.detail === 'string' ? record.detail : raw,
    } satisfies ApiErrorShape;
  }
  return body as T;
}

export const api = {
  health: () => request<{ status: string; contract_version: string; unknown: string[] }>('/health'),
  listPolicies: () => request<{ total: number; items: PolicyIndexItem[] } & UnknownResource>('/policies'),
  getPolicy: (id: string) => request<PolicyDetail>(`/policies/${encodeURIComponent(id)}`),
  validatePolicy: (id: string) => request<PolicyValidation>(`/policies/${encodeURIComponent(id)}/validate`),
  validatePolicyPayload: (id: string, payload: PolicyPayload) =>
    request<PolicyValidation>(`/policies/${encodeURIComponent(id)}/validate`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  savePolicy: (id: string, payload: PolicyPayload) =>
    request<PolicyDetail>(`/policies/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  createPolicy: (payload: PolicyPayload) =>
    request<PolicyDetail>('/policies', { method: 'POST', body: JSON.stringify(payload) }),
  deletePolicy: (id: string) =>
    request<{ deleted: boolean; policy_id: string }>(`/policies/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  nextPolicyId: () =>
    request<{ policy_id: string; existing: string[]; unknown: string[] }>('/policies/next-id'),
  /** 저장하지 않고 초안만 검증한다 — 새 정책 화면이 입력 중에 부른다 */
  validatePolicyDraft: (payload: PolicyPayload) =>
    request<PolicyValidation>('/policies/draft/validate', {
      method: 'POST',
      body: JSON.stringify({ policy: payload }),
    }),
  listRuns: () => request<RunsIndex>('/runs'),
  getRun: (id: string) => request<RunDetail>(`/runs/${encodeURIComponent(id)}`),
  getDays: (id: string) => request<RunDays>(`/runs/${encodeURIComponent(id)}/days`),
  getDay: (id: string, day: string) =>
    request<DayAggregate>(`/runs/${encodeURIComponent(id)}/days/${encodeURIComponent(day)}`),
  getBottlenecks: (id: string, day: string) =>
    request<Bottlenecks>(`/runs/${encodeURIComponent(id)}/days/${encodeURIComponent(day)}/bottlenecks`),
  getFailed: (id: string, day: string) =>
    request<FailedPage>(`/runs/${encodeURIComponent(id)}/days/${encodeURIComponent(day)}/failed`),
  getEventsSummary: (id: string) => request<Record<string, unknown>>(`/runs/${encodeURIComponent(id)}/events/summary`),
  listArtifacts: () => request<{ total: number; items: Array<{ path: string; bytes: number }>; unknown: string[] }>('/artifacts'),
  getLock: () => request<LockStatus>('/runner/lock'),
  /**
   * 실행 시작. `policy` 를 함께 넘기면 **새 정책을 그 자리에서 주입**한다 —
   * 서버가 preflight 를 통과시킨 뒤에만 저장하고 실행한다.
   */
  startRun: (payload: {
    run_id: string;
    policy_id: string;
    policy?: PolicyPayload | null;
    start_day?: string | null;
    days?: number | null;
    agents?: number | null;
  }) =>
    request<{
      accepted?: boolean;
      lock?: Record<string, unknown>;
      command?: string[];
      injected_policy?: PolicyDetail;
    }>('/runner/start', { method: 'POST', body: JSON.stringify(payload) }),
  stopRun: () => request<Record<string, unknown>>('/runner/stop', { method: 'POST' }),
  releaseRun: () => request<Record<string, unknown>>('/runner/release', { method: 'POST' }),
  llmStatus: () => request<LlmStatus>('/llm/status'),
  llmPing: () => request<LlmStatus>('/llm/ping', { method: 'POST' }),
  reportCatalog: (runId: string, policyId: string) =>
    request<ReportCatalog>(`/reports/catalog?run_id=${encodeURIComponent(runId)}&policy_id=${encodeURIComponent(policyId)}`),
  listReportJobs: (runId?: string) =>
    request<{ total: number; items: ReportJob[]; unknown: string[] }>(
      `/reports/jobs${runId ? `?run_id=${encodeURIComponent(runId)}` : ''}`,
    ),
  getReportJob: (jobId: string) => request<ReportJob>(`/reports/jobs/${encodeURIComponent(jobId)}`),
  startReportJob: (payload: {
    run_id: string;
    policy_id: string;
    start: string;
    days: number;
    policy_from?: string | null;
    analyses: string[];
    include_interview: boolean;
    engine?: 'v2' | 'dasol';
    use_llm?: boolean;
  }) => request<ReportJob>('/reports/jobs', { method: 'POST', body: JSON.stringify(payload) }),
};

/** 산출물을 브라우저가 직접 받을 수 있는 절대 경로 */
export function artifactUrl(path: string): string {
  return `${API_BASE}/artifacts/${path.split('/').map(encodeURIComponent).join('/')}`;
}
