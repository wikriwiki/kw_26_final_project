/**
 * 내부 식별자 → 정책 담당자의 말 (스펙 §1).
 *
 * 읽는 사람은 시뮬레이션 내부 구조를 모른다. 그래서 `t_s2` 같은 키를 화면에 그대로
 * 노출하지 않는다. 다만 **모르는 값을 아는 척 번역하지도 않는다** — 매핑에 없는 값은
 * 원문 그대로 돌려주고, 호출하는 쪽이 mono 서체로 원문임을 드러낸다 (CONTRACT §2.7).
 */

export interface Translated {
  label: string;
  /** 매핑에 있던 값인가. false 면 원문 그대로다 */
  known: boolean;
}

const raw = (value: string): Translated => ({ label: value, known: false });

/* --- 정책 ----------------------------------------------------------------- */

const POLICY_TYPE: Record<string, string> = {
  grant: '지원금',
  subsidy: '보조금',
  regulation: '규제',
  facility: '시설 조성',
  campaign: '캠페인',
  tax: '세제',
  transit: '교통',
  environment: '환경',
};

export function policyType(value: string): Translated {
  const label = POLICY_TYPE[value];
  return label ? { label, known: true } : raw(value);
}

const GRANT_KEY: Record<string, string> = {
  spend_decile: '소비 10분위',
  income: '소득 구간',
};

export function grantKey(value: string): Translated {
  const label = GRANT_KEY[value];
  return label ? { label, known: true } : raw(value);
}

/** 소비 분위 라벨. null 은 "분위 미상" — 버리지 않고 반드시 표시한다 (CONTRACT §3.4) */
export function decile(value: number | null): string {
  return value === null ? '분위 미상' : `${value}분위`;
}

/* --- 실행 ----------------------------------------------------------------- */

export function runStatus(value: 'completed' | 'incomplete'): Translated {
  // "중단됨"이지 "실패함"이 아니다 (CONTRACT §4.1-5)
  return value === 'completed'
    ? { label: '완료', known: true }
    : { label: '중단됨', known: true };
}

/* --- 단계 소요 ------------------------------------------------------------- */

/** `timing/day_*.json` 의 path 네임스페이스. 그룹만 번역하고 leaf 는 원문을 남긴다 */
const PATH_GROUP: Record<string, string> = {
  phase: '단계 합계',
  dawn: '하루 시작 준비',
  prompt: '프롬프트 구성',
  stage1: '행동 의도 결정',
  stage2: '장소·지출 결정',
};

/** `phase.t_s1` → 정책 담당자용 이름. 4개 대표 단계는 완전히 번역한다 */
const PHASE_LEAF: Record<string, string> = {
  'phase.t_dawn': '하루 시작 준비',
  'phase.t_s1': '행동 의도 결정',
  'phase.t_s2': '장소·지출 결정',
  'phase.t_write_plan': '하루 계획 기록',
  'phase.t_night_finalize': '전날 정산',
};

export interface TimingPath {
  /** 화면에 크게 보일 이름 */
  label: string;
  /** 원문 path. 매핑이 없으면 이걸 mono 로 함께 보여준다 */
  detail: string | null;
}

export function timingPath(path: string): TimingPath {
  const exact = PHASE_LEAF[path];
  if (exact) return { label: exact, detail: null };

  const [group, ...restParts] = path.split('.');
  const rest = restParts.join('.');
  const groupLabel = PATH_GROUP[group ?? ''];
  if (groupLabel && rest) return { label: groupLabel, detail: rest };
  return { label: path, detail: null };
}

/* --- 응답 오류 (stage1_failures) ------------------------------------------- */

const ERROR_TYPE: Record<string, string> = {
  ValidationError: '형식 검증 실패',
  JSONDecodeError: 'JSON 해석 실패',
  ValueError: '값 범위 오류',
};

export function errorType(value: string): Translated {
  const label = ERROR_TYPE[value];
  return label ? { label, known: true } : raw(value);
}

const FINISH_REASON: Record<string, string> = {
  stop: '정상 종료',
  length: '길이 초과로 잘림',
};

export function finishReason(value: string): Translated {
  const label = FINISH_REASON[value];
  return label ? { label, known: true } : raw(value);
}

/* --- 결과 지표 ------------------------------------------------------------- */

export const METRIC_HELP: Record<string, string> = {
  amt: '시뮬레이션 기간에 발생한 모든 결제 금액의 합계입니다.',
  policy_paid: '정책 지갑에서 실제로 빠져나간 금액입니다. 자기 자금보다 먼저 사용됩니다.',
  extra_spent: '정책이 없었다면 쓰지 않았을 것으로 판단된 금액입니다. 값이 없는 건이 많아 하한값으로 읽어야 합니다.',
  coupon_eligible_events: '쿠폰을 쓸 수 있는 매장에서 일어난 결제 건수입니다.',
};

/** 실행을 사람 말로 한 줄 요약. 픽스처 값만 조합한다 */
export function runSummaryLine(input: {
  days_present: number;
  days_planned: number | null;
  agents_target: number | null;
}): string {
  const days =
    input.days_planned === null
      ? `${input.days_present}일 기록됨 (계획 일수 알 수 없음)`
      : `${input.days_present}일 / 계획 ${input.days_planned}일`;
  const agents = input.agents_target === null ? '하루 목표 인원 알 수 없음' : `하루 ${input.agents_target}명`;
  return `${days} · ${agents}`;
}
