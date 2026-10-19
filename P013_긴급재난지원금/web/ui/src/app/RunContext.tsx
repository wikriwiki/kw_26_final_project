/**
 * 현재 실행(run) 컨텍스트 — `docs/DESIGN_IA_RUN_FIRST.md` §4.
 *
 * 정보구조가 "기능 → 실행"에서 "실행 → 기능"으로 바뀌었다. run 은 주소(`/runs/:runId`)가
 * 소유하고, 그 아래 모든 화면은 **자기 안에 run 선택기를 두지 않는다** (설계도 §5).
 * 화면은 `useParams` 대신 `useRun()` 으로 이미 정해진 run 을 받아 쓴다.
 *
 * 여기에 모인 파생값(적용 정책, 기능별 가용 여부)은 픽스처의 실측 필드만 조합한 것이다.
 * 모르는 값을 지어내지 않는다 — 모르면 `null` 로 두고 화면이 "알 수 없음"이라 적는다.
 */
import { createContext, useContext, useMemo } from 'react';
import type { ReactNode } from 'react';
import type { RunIndexItem } from '../lib/api';
import { policiesIndex, runs, runsIndex } from '../lib/fixtures';
import type { RunBundle, RunId } from '../lib/fixtures';

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
}

function readPolicy(bundle: RunBundle): PolicyBinding {
  const events = bundle.events;
  if (!events.available || events.policy_paid_by_policy_id === null) {
    return { known: false, items: [] };
  }
  const items = Object.keys(events.policy_paid_by_policy_id).map((id) => ({
    id,
    name: policiesIndex.items.find((p) => p.id === id)?.name ?? null,
  }));
  return { known: true, items };
}

/* --- 실행 한 줄 요약 --------------------------------------------------------- */

export interface RunSummary {
  id: RunId;
  index: RunIndexItem;
  bundle: RunBundle;
  phase: RunPhase;
  /** 기록이 남은 일 수 */
  daysPresent: number;
  /** 계획 일 수. 모르면 null → 진행률 막대를 그리지 않는다 */
  daysPlanned: number | null;
  agentsTarget: number | null;
  firstDay: string | null;
  lastDay: string | null;
  policy: PolicyBinding;
  /** 마지막 갱신 시각 */
  updatedAt: string | null;
  features: RunFeature[];
}

function buildFeatures(bundle: RunBundle): RunFeature[] {
  const hasDays = bundle.days.items.length > 0;
  const events = bundle.events;
  const policy = readPolicy(bundle);

  const feature = (
    key: FeatureKey,
    segment: string,
    label: string,
    available: boolean,
    reason: string,
  ): RunFeature => ({ key, segment, label, available, reason: available ? null : reason });

  return [
    feature('overview', '', '개요', true, ''),
    feature('monitor', 'monitor', '실행 모니터', hasDays, '일자 기록 없음'),
    feature('results', 'results', '결과', events.available, '결제 기록 없음'),
    feature('visualize', 'visualize', '시각화', hasDays, '지도 데이터 없음'),
    /*
     * 대상자 문답·보고서는 run 산출물이 아니라 **정적 시연 표본**을 읽는다.
     * 그래서 실행 상태와 무관하게 항상 열려 있다. 대신 각 화면 상단에
     * "이 실행의 산출물이 아니다" 를 명시한다 — 시각화의 EMBED_CASE 와 같은 처리.
     */
    feature('agents', 'agents', '대상자 문답', true, ''),
    feature('report', 'report', '보고서', true, ''),
    feature('policy', 'policy', '정책', policy.known, '적용 정책 기록 없음'),
  ];
}

function summarize(id: RunId): RunSummary {
  const bundle = runs[id];
  const index =
    runsIndex.items.find((it) => it.run_id === id) ??
    ({
      run_id: id,
      root: bundle.detail.root,
      status: bundle.detail.status,
      first_day: bundle.days.items[0]?.day ?? null,
      last_day: bundle.days.items[bundle.days.items.length - 1]?.day ?? null,
      days_present: bundle.detail.days_present.length,
      days_planned: bundle.detail.plan.planned_days,
      agents_target: bundle.detail.plan.agents_target,
      completed_at: bundle.detail.completed_at,
      artifacts: bundle.detail.artifacts,
      unknown: [],
    } satisfies RunIndexItem);

  return {
    id,
    index,
    bundle,
    phase: runPhase(index),
    daysPresent: index.days_present,
    daysPlanned: index.days_planned,
    agentsTarget: index.agents_target,
    firstDay: index.first_day,
    lastDay: index.last_day,
    policy: readPolicy(bundle),
    updatedAt: bundle.detail.updated_at ?? index.completed_at,
    features: buildFeatures(bundle),
  };
}

/**
 * 픽스처는 빌드 타임 상수라 한 번만 계산한다.
 * 목록 순서는 `runs.index` 가 준 순서를 그대로 따른다 — 임의로 재정렬하지 않는다.
 */
const SUMMARIES: RunSummary[] = runsIndex.items
  .filter((it): it is RunIndexItem & { run_id: RunId } => it.run_id in runs)
  .map((it) => summarize(it.run_id));

export function useRunSummaries(): RunSummary[] {
  return SUMMARIES;
}

export function findSummary(id: RunId): RunSummary {
  return SUMMARIES.find((s) => s.id === id) ?? summarize(id);
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
export function RunProvider({ runId, children }: { runId: RunId; children: ReactNode }) {
  const value = useMemo<RunContextValue>(() => {
    const summary = findSummary(runId);
    const can = (key: FeatureKey) =>
      summary.features.find((f) => f.key === key)?.available ?? false;
    const path = (segment?: string) =>
      segment ? `/runs/${runId}/${segment}` : `/runs/${runId}`;
    return { ...summary, can, path };
  }, [runId]);

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
