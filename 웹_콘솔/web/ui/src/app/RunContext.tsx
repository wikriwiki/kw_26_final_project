/**
 * 현재 실행(run) 컨텍스트 — `docs/DESIGN_IA_RUN_FIRST.md` §4.
 *
 * 정보구조가 "기능 → 실행"에서 "실행 → 기능"으로 바뀌었다. run 은 주소(`/runs/:runId`)가
 * 소유하고, 그 아래 모든 화면은 **자기 안에 run 선택기를 두지 않는다** (설계도 §5).
 * 화면은 `useParams` 대신 `useRun()` 으로 이미 정해진 run 을 받아 쓴다.
 *
 * 여기에 모인 파생값(적용 정책, 기능별 가용 여부)은 API 가 준 실측 필드만 조합한 것이다.
 * 모르는 값을 지어내지 않는다 — 모르면 `null` 로 두고 화면이 "알 수 없음"이라 적는다.
 *
 * 목록(홈·실행 전환)은 가벼운 색인만 받는다. 실행 하나의 상세 묶음은 그 실행을 열 때만 받는다.
 */
import { createContext, useContext, useMemo } from 'react';
import type { ReactNode } from 'react';
import type { PolicyIndexItem, RunIndexItem, RunMeta } from '../lib/api';
import { mapAsync, type AsyncState } from '../lib/async';
import {
  loadEventsSummary,
  loadPolicies,
  loadRunBundle,
  loadRunsIndex,
  useLoad,
  type EventsSummary,
  type RunBundle,
  type RunId,
} from '../lib/runData';

/* --- 실행 상태 -------------------------------------------------------------- */

/**
 * 목록에서 묶는 단위 (설계도 §3: 진행중 → 완료 → 중단).
 *
 * `web/CONTRACT.md` §3.2 의 `status` 는 `completed | incomplete` 두 값뿐이라
 * 지금 픽스처에는 "진행중"인 실행이 없다. 그래도 묶음을 미리 만들어 둔다 —
 * 서버가 진행 중인 실행을 내보내기 시작하면 그대로 첫 묶음에 들어간다.
 * 없는 묶음은 화면에 그리지 않는다.
 */
export type RunPhase = 'running' | 'completed' | 'stopped';

export function runPhase(item: RunIndexItem): RunPhase {
  const status: string = item.status;
  if (status === 'completed') return 'completed';
  if (status === 'running' || status === 'in_progress') return 'running';
  return 'stopped';
}

export const PHASE_LABEL: Record<RunPhase, string> = {
  running: '진행중',
  completed: '완료',
  stopped: '중단됨',
};

/* --- 기능별 가용 여부 -------------------------------------------------------- */

export type FeatureKey =
  | 'overview'
  | 'monitor'
  | 'results'
  | 'visualize'
  | 'agents'
  | 'report'
  | 'policy';

export interface RunFeature {
  key: FeatureKey;
  /** `/runs/:runId` 아래에 붙는 조각. 개요는 빈 문자열 */
  segment: string;
  label: string;
  /** false 면 **숨기지 않고** 비활성 + 이유를 적는다 (설계도 §4) */
  available: boolean;
  /** 비활성 이유 한 줄. 활성일 때는 null */
  reason: string | null;
}

export const FEATURE_ORDER: FeatureKey[] = [
  'overview',
  'monitor',
  'results',
  'visualize',
  'agents',
  'report',
  'policy',
];

/* --- 적용 정책 --------------------------------------------------------------- */

export interface AppliedPolicy {
  id: string;
  /** 정책 목록에 없는 id 면 null — 이름을 지어내지 않는다 */
  name: string | null;
}

/**
 * 그 실행에서 실제로 정책 지갑이 쓰인 기록(`events.summary.policy_paid_by_policy_id`)으로
 * 적용 정책을 읽는다. 실행 산출물에 "적용 정책 id" 필드가 따로 없기 때문이다.
 * 결제 기록 자체가 없는 실행은 `known:false` — "무정책"과 "알 수 없음"을 섞지 않는다.
 */
export interface PolicyBinding {
  known: boolean;
  items: AppliedPolicy[];
  /** 정책 본런 실행의 이름 (예: 사회적 거리두기 2단계 격상). 지원금이 없는 정책도 이름으로 보인다 */
  title: string | null;
  /** 정책 파일이 콘솔에 있는지 — 정책 화면을 열 수 있는지 */
  fileKnown: boolean;
}

function readPolicy(events: EventsSummary, policies: PolicyIndexItem[], meta: RunMeta | null): PolicyBinding {
  if (meta) {
    const file = meta.policy_id ? policies.find((p) => p.id === meta.policy_id) : undefined;
    return {
      known: true,
      items: meta.policy_id ? [{ id: meta.policy_id, name: meta.title ?? file?.name ?? null }] : [],
      title: meta.title,
      fileKnown: Boolean(file),
    };
  }
  if (!events.available || events.policy_paid_by_policy_id === null) {
    return { known: false, items: [], title: null, fileKnown: false };
  }
  const items = Object.keys(events.policy_paid_by_policy_id).map((id) => ({
    id,
    name: policies.find((p) => p.id === id)?.name ?? null,
  }));
  return { known: true, items, title: null, fileKnown: items.some((p) => p.name !== null) };
}

/* --- 실행 한 줄 요약 --------------------------------------------------------- */

/** 목록 한 줄 — 색인만으로 만든다 */
export interface RunListItem {
  id: RunId;
  index: RunIndexItem;
  phase: RunPhase;
  /** 기록이 남은 일 수 */
  daysPresent: number;
  /** 계획 일 수. 모르면 null → 진행률 막대를 그리지 않는다 */
  daysPlanned: number | null;
  agentsTarget: number | null;
  firstDay: string | null;
  lastDay: string | null;
  /** 마지막 갱신 시각 */
  updatedAt: string | null;
}

/** 실행 하나를 연 뒤의 전부 — 상세 묶음·적용 정책·기능 가용 여부 */
export interface RunSummary extends RunListItem {
  bundle: RunBundle;
  policy: PolicyBinding;
  features: RunFeature[];
}

function buildFeatures(bundle: RunBundle, policy: PolicyBinding): RunFeature[] {
  const hasDays = bundle.days.items.length > 0;
  const events = bundle.events;

  const feature = (
    key: FeatureKey,
    segment: string,
    label: string,
    available: boolean,
    reason: string,
  ): RunFeature => ({ key, segment, label, available, reason: available ? null : reason });

  const running = bundle.detail.status !== 'completed';
  return [
    feature('overview', '', '개요', true, ''),
    ...(running ? [feature('monitor', 'monitor', '실행 모니터', hasDays, '일자 기록 없음')] : []),
    feature('results', 'results', '결과', events.available, '결제 기록 없음'),
    feature('visualize', 'visualize', '시각화', hasDays, '지도 데이터 없음'),
    /*
     * 대상자 문답·보고서는 run 산출물이 아니라 **정적 시연 표본**을 읽는다.
     * 그래서 실행 상태와 무관하게 항상 열려 있다. 대신 각 화면 상단에
     * "이 실행의 산출물이 아니다" 를 명시한다 — 시각화의 EMBED_CASE 와 같은 처리.
     */
    feature('agents', 'agents', '시민 인터뷰', true, ''),
    feature('report', 'report', '보고서', true, ''),
    feature('policy', 'policy', '정책', policy.known && policy.fileKnown, '정책 파일 없음'),
  ];
}

function listItem(index: RunIndexItem): RunListItem {
  return {
    id: index.run_id,
    index,
    phase: runPhase(index),
    daysPresent: index.days_present,
    daysPlanned: index.days_planned,
    agentsTarget: index.agents_target,
    firstDay: index.first_day,
    lastDay: index.last_day,
    updatedAt: index.completed_at,
  };
}

/** 색인에 없는 실행(방금 생겼거나 색인이 늦은 경우)은 상세에서 같은 모양을 만든다 */
function indexFromBundle(id: RunId, bundle: RunBundle): RunIndexItem {
  return {
    run_id: id,
    root: bundle.detail.root,
    status: bundle.detail.status,
    first_day: bundle.days.items[0]?.day ?? null,
    last_day: bundle.days.items[bundle.days.items.length - 1]?.day ?? null,
    days_present: bundle.detail.days_present.length,
    days_planned: bundle.detail.plan.planned_days,
    agents_target: bundle.detail.plan.agents_target,
    meta: bundle.detail.meta ?? null,
    completed_at: bundle.detail.completed_at,
    artifacts: bundle.detail.artifacts,
    unknown: [],
  };
}

/** 실행 하나를 열 때 필요한 것을 한꺼번에 받는다 */
export async function loadRunSummary(id: RunId): Promise<RunSummary> {
  const [bundle, runsIndex, policies] = await Promise.all([loadRunBundle(id), loadRunsIndex(), loadPolicies()]);
  const index = runsIndex.items.find((it) => it.run_id === id) ?? indexFromBundle(id, bundle);
  const policy = readPolicy(bundle.events, policies.items, index.meta ?? bundle.detail.meta ?? null);
  return {
    ...listItem(index),
    bundle,
    policy,
    updatedAt: bundle.detail.updated_at ?? index.completed_at,
    features: buildFeatures(bundle, policy),
  };
}

/** 목록 순서는 서버가 준 순서를 그대로 따른다 — 임의로 재정렬하지 않는다 */
export function useRunList(): AsyncState<RunListItem[]> & { retry: () => void } {
  const state = useLoad('runs', loadRunsIndex);
  return { ...mapAsync(state, (index) => index.items.map(listItem)), retry: state.retry };
}

/** 목록 한 줄의 적용 정책. 실행마다 결제 요약을 받아야 해서 줄이 먼저 그려진 뒤 채운다 */
export function useRunPolicy(id: RunId): AsyncState<PolicyBinding> {
  return useLoad(`policy-binding:${id}`, async () => {
    const [runsIndex, policies] = await Promise.all([loadRunsIndex(), loadPolicies()]);
    const meta = runsIndex.items.find((it) => it.run_id === id)?.meta ?? null;
    // 정책 본런 실행은 이름표로 충분하다 — 결제 기록 전체를 훑지 않는다
    if (meta) return readPolicy({ available: false } as EventsSummary, policies.items, meta);
    return readPolicy(await loadEventsSummary(id), policies.items, null);
  });
}

/* --- 컨텍스트 ---------------------------------------------------------------- */

export interface RunContextValue extends RunSummary {
  /** 그 실행에서 쓸 수 있는 기능인가 */
  can: (key: FeatureKey) => boolean;
  /** `/runs/:runId/...` 절대 경로를 만든다 */
  path: (segment?: string) => string;
}

const RunCtx = createContext<RunContextValue | null>(null);

/**
 * run 은 주소가 소유한다. 화면이 자기 상태로 들고 있지 않으므로 저장·복원이 없고,
 * 주소 하나로 링크가 완결된다 (스펙 §9 deep-linking).
 */
export function RunProvider({ summary, children }: { summary: RunSummary; children: ReactNode }) {
  const runId = summary.id;
  const value = useMemo<RunContextValue>(() => {
    const can = (key: FeatureKey) =>
      summary.features.find((f) => f.key === key)?.available ?? false;
    const path = (segment?: string) =>
      segment ? `/runs/${runId}/${segment}` : `/runs/${runId}`;
    return { ...summary, can, path };
  }, [runId, summary]);

  return <RunCtx.Provider value={value}>{children}</RunCtx.Provider>;
}

/** run 셸 안에서만 부른다. 밖에서 부르면 개발 중에 바로 드러나도록 던진다 */
export function useRun(): RunContextValue {
  const value = useContext(RunCtx);
  if (!value) throw new Error('useRun 은 RunProvider 안에서만 쓸 수 있습니다');
  return value;
}

/** 셸 밖(홈·리다이렉트)에서도 안전하게 물어본다 */
export function useRunOptional(): RunContextValue | null {
  return useContext(RunCtx);
}
