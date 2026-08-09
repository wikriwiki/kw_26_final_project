/**
 * 새 시뮬레이션 만들기 — 설계도 §6, 라우트 `/new`.
 *
 * 단계는 넷이고 **한 화면에 다 펼치지 않는다** (스펙 §8 progressive-disclosure).
 *   1단계 정책 고르기 → (새 정책이면) 2단계 정책 작성 → 3단계 설정 확인 → 4단계 실행
 * 각 단계에 뒤로 가기가 있고, 마지막 단계 전까지는 아무 일도 일어나지 않는다.
 *
 * **정책 주입 (이번 라운드에서 추가).**
 * 기존 정책 목록에서 고르는 것 말고, 이 화면에서 **새 정책을 만들어 그대로 주입**할 수 있다.
 * 저장은 서버가 preflight 를 통과시킨 경우에만 이뤄지고(`POST /api/policies`),
 * 실행 요청은 정책 본문을 함께 실어 보낸다(`POST /api/runner/start`). 즉
 * **검증되지 않은 정책으로 시뮬레이션이 시작되는 경로가 없다.**
 *
 * **되는 척하지 않는다 (기준 B1).**
 * API 가 없거나 실행 명령이 구성되지 않았으면 그 사실을 그대로 적는다.
 * 가짜 진행률·가짜 성공·가짜 run_id 를 만들지 않는다.
 *
 * **중복 실행 안전장치 (기준 B8).** 서버 lock(`GET /api/runner/lock`)을 실제로 조회해
 * 보유자·시작시각을 화면에 표시하고, 잠겨 있으면 실행 버튼을 막는다. UI 비활성화만으로는
 * 기준 미달이므로 서버가 물리적으로 막는다는 사실도 함께 적는다.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Badge } from '../components/Badge';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Disclosure } from '../components/Disclosure';
import { Callout, EmptyState, ErrorState, SkeletonText } from '../components/Feedback';
import { SelectField, TextAreaField, TextField } from '../components/Field';
import {
  AlertCircleIcon,
  AlertTriangleIcon,
  ArrowLeftIcon,
  CheckCircleIcon,
  RefreshIcon,
} from '../components/Icon';
import { BarList } from '../components/Meter';
import type { BarItem } from '../components/Meter';
import { api } from '../lib/api';
import type {
  ApiErrorShape,
  LockStatus,
  PolicyDetail,
  PolicyIndexItem,
  PolicyPayload,
  PolicyValidation,
} from '../lib/api';
import { dateTime, int, krw } from '../lib/format';
import { grantKey, policyType } from '../lib/labels';

import lockEvidenceRaw from '../../../fixtures/runner.lock.evidence.json?raw';

/* --- 중복 실행이 왜 위험한지의 실측 근거 (CONTRACT §3.13) ------------------- */

interface LockEvidence {
  source: string;
  note: string;
  timeline: Array<{ line_no: number; text: string }>;
  killed_run: { run_id: string; run_root: string; log: string } | null;
}

const lockEvidence = JSON.parse(lockEvidenceRaw) as LockEvidence;

/* --- 상수 ------------------------------------------------------------------ */

const STEPS = [
  { id: 'policy', label: '정책 고르기' },
  { id: 'compose', label: '정책 작성' },
  { id: 'settings', label: '설정 확인' },
  { id: 'launch', label: '실행' },
] as const;

/** 정책을 고르지 않는 선택지 */
const CONTROL = '__control__';
/** 새 정책을 만드는 선택지 */
const NEW_POLICY = '__new__';

const INCOME_ORDER = ['하', '중하', '중', '중상', '상'];
const DECILES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];

/** `data/neo4j_load/categories/categories.yaml` 의 12대분류 */
const L1_CATEGORIES = [
  '식사',
  '카페',
  '디저트',
  '주점',
  '편의점',
  '마트',
  '미용',
  '쇼핑',
  '여가',
  '건강',
  '교육',
  '기타',
];

const POLICY_TYPES = [
  { value: 'grant', label: '지원금 — 대상자에게 금액을 지급' },
  { value: 'facility', label: '시설 조성 — 장소의 매력도를 바꿈' },
  { value: 'campaign', label: '캠페인 — 인지·의향을 바꿈' },
  { value: 'regulation', label: '규제 — 이용을 제한' },
];

const ISO_DAY = /^(\d{4})-(\d{2})-(\d{2})$/;

type Grade = 'pass' | 'warn' | 'fail';

function GradeIcon({ grade }: { grade: Grade }) {
  if (grade === 'pass') return <CheckCircleIcon size={16} />;
  if (grade === 'warn') return <AlertTriangleIcon size={16} />;
  return <AlertCircleIcon size={16} />;
}

function grantBarsOf(policy: PolicyPayload): BarItem[] {
  const decileGrants = policy.decile_grants;
  if (decileGrants && Object.keys(decileGrants).length > 0) {
    return Object.entries(decileGrants)
      .filter(([, v]) => Number(v) > 0)
      .map(([k, v]) => ({ key: k, name: `${k}분위`, value: Number(v), display: krw(Number(v)) }))
      .sort((a, b) => Number(a.key) - Number(b.key));
  }
  const incomeGrants = policy.income_grants;
  if (incomeGrants && Object.keys(incomeGrants).length > 0) {
    return Object.entries(incomeGrants)
      .filter(([, v]) => Number(v) > 0)
      .map(([k, v]) => ({ key: k, name: `소득 ${k}`, value: Number(v), display: krw(Number(v)) }))
      .sort((a, b) => INCOME_ORDER.indexOf(a.key) - INCOME_ORDER.indexOf(b.key));
  }
  return [];
}

function addDays(iso: string, n: number): string | null {
  const m = ISO_DAY.exec(iso);
  if (!m) return null;
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  if (Number.isNaN(d.getTime())) return null;
  d.setUTCDate(d.getUTCDate() + n);
  const p = (x: number) => String(x).padStart(2, '0');
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}`;
}

function parseCount(value: string): number | null {
  if (!/^\d+$/.test(value.trim())) return null;
  const n = Number(value.trim());
  return Number.isSafeInteger(n) ? n : null;
}

function errorOf(value: unknown): ApiErrorShape {
  if (value && typeof value === 'object' && 'message' in value) return value as ApiErrorShape;
  return { message: String(value) };
}

/** 비어 있는 새 정책 초안. 값을 지어내지 않고 **비워 둔다** */
function emptyDraft(id: string): PolicyPayload {
  return {
    id,
    name: '',
    type: 'grant',
    description: '',
    announce_date: null,
    effective_from: '',
    effective_until: '',
    target_districts: ['서울특별시'],
    benefit_categories: [],
    poi_restricted: false,
    grant_key: 'spend_decile',
    decile_grants: Object.fromEntries(DECILES.map((d) => [d, 0])),
    excluded_deciles: [],
    income_grants: {},
    excluded_income: [],
    notes: '웹 콘솔에서 작성한 정책',
  };
}

/* --- 화면 ------------------------------------------------------------------ */

export function NewRunScreen() {
  const [step, setStep] = useState(0);

  /* 서버 상태 */
  const [policies, setPolicies] = useState<PolicyIndexItem[] | null>(null);
  const [policiesError, setPoliciesError] = useState<ApiErrorShape | null>(null);
  const [lock, setLock] = useState<LockStatus | null>(null);
  const [lockError, setLockError] = useState<ApiErrorShape | null>(null);

  /* 선택 */
  const [choice, setChoice] = useState<string | null>(null);
  const [detail, setDetail] = useState<PolicyDetail | null>(null);
  const [draft, setDraft] = useState<PolicyPayload | null>(null);

  /* 검증 */
  const [validation, setValidation] = useState<PolicyValidation | null>(null);
  const [validating, setValidating] = useState(false);
  const [validationError, setValidationError] = useState<ApiErrorShape | null>(null);

  /* 실행 설정 */
  const [runId, setRunId] = useState('');
  const [startDay, setStartDay] = useState('');
  const [days, setDays] = useState('7');
  const [agents, setAgents] = useState('200');
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [attempted, setAttempted] = useState(false);

  /* 실행 */
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<ApiErrorShape | null>(null);
  const [launched, setLaunched] = useState<{ lock: Record<string, unknown> | undefined; injected: boolean } | null>(
    null,
  );

  const isControl = choice === CONTROL;
  const isNew = choice === NEW_POLICY;
  const item = choice && !isControl && !isNew ? policies?.find((p) => p.id === choice) : undefined;

  /* --- 서버에서 읽기 ------------------------------------------------------ */

  const loadPolicies = useCallback(() => {
    setPoliciesError(null);
    api
      .listPolicies()
      .then((list) => setPolicies(list.items))
      .catch((error) => {
        setPolicies([]);
        setPoliciesError(errorOf(error));
      });
  }, []);

  const loadLock = useCallback(() => {
    setLockError(null);
    api
      .getLock()
      .then(setLock)
      .catch((error) => setLockError(errorOf(error)));
  }, []);

  useEffect(loadPolicies, [loadPolicies]);
  useEffect(loadLock, [loadLock]);
  /* 실행 화면에서는 lock 을 계속 확인한다 — 다른 창에서 시작했을 수 있다 */
  useEffect(() => {
    if (step !== 3) return;
    const timer = window.setInterval(loadLock, 4000);
    return () => window.clearInterval(timer);
  }, [step, loadLock]);

  /* --- 파생값 ------------------------------------------------------------- */

  const activePolicy: PolicyPayload | null = isNew ? draft : (detail?.policy ?? null);
  const grantBars = useMemo<BarItem[]>(
    () => (activePolicy ? grantBarsOf(activePolicy) : []),
    [activePolicy],
  );

  const failing = validation ? validation.checks.filter((c) => c.grade === 'fail') : [];
  const warning = validation ? validation.checks.filter((c) => c.grade === 'warn') : [];
  const passing = validation ? validation.checks.filter((c) => c.grade === 'pass') : [];
  const lead: Grade = failing.length > 0 ? 'fail' : warning.length > 0 ? 'warn' : 'pass';

  const dayCount = parseCount(days);
  const agentCount = parseCount(agents);
  const startValid = ISO_DAY.test(startDay);

  const errors: Record<string, string> = {};
  if (!/^[A-Za-z0-9_-]{2,32}$/.test(runId.trim())) {
    errors.runId = '실행 이름은 영문·숫자·하이픈 2~32자여야 합니다 — 예: BASE_0810.';
  }
  if (!startValid) {
    errors.startDay = '시작일이 비어 있거나 형식이 어긋납니다 — 2025-07-21 처럼 입력하세요.';
  }
  if (dayCount === null || dayCount < 1 || dayCount > 365) {
    errors.days = '기간은 1일 이상 365일 이하의 정수여야 합니다.';
  }
  if (agentCount === null || agentCount < 1) {
    errors.agents = '대상자 수는 1명 이상의 정수여야 합니다.';
  }
  const show = (key: string) => (attempted || touched[key] ? errors[key] : undefined);
  const settingsOk = Object.keys(errors).length === 0;

  /* 정책이 있으면 검증을 통과해야 실행 단계로 간다. 대조군은 검증할 파일이 없다 */
  const needsValidation = Boolean(item) || isNew;
  const validationBlocks = needsValidation && (!validation || failing.length > 0);
  const canGoLaunch = settingsOk && !validationBlocks;

  const endDay = startValid && dayCount !== null ? addDays(startDay, dayCount - 1) : null;
  const locked = lock?.locked === true;

  const choiceName = isControl
    ? '무정책 (대조군)'
    : isNew
      ? (draft?.name?.trim() || '새 정책 (이름 없음)')
      : (detail?.policy.name ?? item?.name ?? '정책을 고르지 않았습니다');

  /* --- 초안 검증 ---------------------------------------------------------- */

  const draftErrors = useMemo(() => {
    if (!draft) return {} as Record<string, string>;
    const out: Record<string, string> = {};
    if (!draft.name.trim()) out.name = '정책 이름이 비어 있습니다 — 무엇을 하는 정책인지 적으세요.';
    if (!ISO_DAY.test(draft.effective_from)) out.effective_from = '시행 시작일이 필요합니다 — 2025-07-21 형식.';
    if (!ISO_DAY.test(draft.effective_until)) out.effective_until = '시행 종료일이 필요합니다 — 2025-11-30 형식.';
    if (
      ISO_DAY.test(draft.effective_from) &&
      ISO_DAY.test(draft.effective_until) &&
      draft.effective_until < draft.effective_from
    ) {
      out.effective_until = '종료일이 시작일보다 빠릅니다 — 날짜를 바꾸세요.';
    }
    if (draft.type === 'grant') {
      const total = Object.values(draft.decile_grants ?? {}).reduce((a, b) => a + Number(b || 0), 0);
      if (total <= 0) out.grants = '지원금 정책인데 지급액이 모두 0입니다 — 한 구간 이상 금액을 넣으세요.';
    }
    return out;
  }, [draft]);

  const draftOk = draft !== null && Object.keys(draftErrors).length === 0;

  /* --- 동작 -------------------------------------------------------------- */

  function resetDownstream() {
    setValidation(null);
    setValidationError(null);
    setAttempted(false);
    setTouched({});
    setLaunched(null);
    setLaunchError(null);
  }

  function pick(next: string) {
    setChoice(next);
    setDetail(null);
    setDraft(null);
    resetDownstream();

    if (next === CONTROL) {
      setStartDay('');
      return;
    }
    if (next === NEW_POLICY) {
      api
        .nextPolicyId()
        .then((r) => setDraft(emptyDraft(r.policy_id)))
        .catch(() => setDraft(emptyDraft('P900')));
      return;
    }
    api
      .getPolicy(next)
      .then((d) => {
        setDetail(d);
        setStartDay(d.policy.effective_from ?? '');
      })
      .catch((error) => setValidationError(errorOf(error)));
  }

  function runValidation() {
    setValidating(true);
    setValidationError(null);
    const request =
      isNew && draft
        ? api.validatePolicyDraft(draft)
        : item
          ? api.validatePolicy(item.id)
          : null;
    if (!request) {
      setValidating(false);
      return;
    }
    request
      .then(setValidation)
      .catch((error) => setValidationError(errorOf(error)))
      .finally(() => setValidating(false));
  }

  function next() {
    setAttempted(true);
    if (step === 0) {
      if (!choice) return;
      setAttempted(false);
      setStep(isNew ? 1 : 2);
      return;
    }
    if (step === 1) {
      if (!draftOk) return;
      setAttempted(false);
      setStep(2);
      return;
    }
    if (step === 2) {
      if (!canGoLaunch) return;
      setAttempted(false);
      setStep(3);
    }
  }

  function back(to: number) {
    setStep(to);
    setAttempted(false);
    setLaunchError(null);
  }

  function launch() {
    setLaunching(true);
    setLaunchError(null);
    api
      .startRun({
        run_id: runId.trim(),
        policy_id: isControl ? 'P000' : (activePolicy?.id ?? item?.id ?? ''),
        policy: isNew && draft ? draft : null,
        start_day: startDay,
        days: dayCount,
        agents: agentCount,
      })
      .then((result) => {
        setLaunched({ lock: result.lock, injected: Boolean(result.injected_policy) });
        loadLock();
      })
      .catch((error) => setLaunchError(errorOf(error)))
      .finally(() => setLaunching(false));
  }

  function patchDraft(patch: Partial<PolicyPayload>) {
    setDraft((current) => (current ? { ...current, ...patch } : current));
    setValidation(null);
  }

  function toggleCategory(name: string) {
    if (!draft) return;
    const current = draft.benefit_categories ?? [];
    patchDraft({
      benefit_categories: current.includes(name)
        ? current.filter((x) => x !== name)
        : [...current, name],
    });
  }

  /* --- 렌더 -------------------------------------------------------------- */

  const visibleSteps = isNew ? STEPS : STEPS.filter((s) => s.id !== 'compose');
  const stepIndex = visibleSteps.findIndex((s) => s.id === STEPS[step].id);

  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">새 시뮬레이션 만들기</h1>
          <p className="pagehead__purpose">
            적용할 정책을 고르거나 새로 만들고, 설정을 확인한 뒤 실행합니다. 마지막 단계 전까지는
            아무것도 시작되지 않습니다.
          </p>
        </div>
      </header>

      {/* 단계 표시기 */}
      <nav aria-label="진행 단계">
        <ol className="row" style={{ gap: 'var(--sp-5)' }}>
          {visibleSteps.map((s, i) => {
            const done = i < stepIndex;
            const current = i === stepIndex;
            return (
              <li
                key={s.id}
                className="row"
                aria-current={current ? 'step' : undefined}
                style={{
                  gap: 'var(--sp-2)',
                  fontSize: 'var(--fs-md)',
                  fontWeight: current ? 600 : 400,
                  color: current ? 'var(--fg)' : 'var(--fg-muted)',
                }}
              >
                {done ? (
                  <CheckCircleIcon size={16} />
                ) : (
                  <span className="num" aria-hidden="true">
                    {i + 1}
                  </span>
                )}
                <span>
                  <span className="visually-hidden">{`${i + 1}단계 `}</span>
                  {s.label}
                </span>
              </li>
            );
          })}
        </ol>
      </nav>

      {/* 서버 lock — 어느 단계에서도 보인다 */}
      {locked ? (
        <Callout tone="warn">
          지금 실행 중인 시뮬레이션이 있습니다 (
          <span className="num">{String(lock?.owner?.run_id ?? '알 수 없음')}</span> · 시작{' '}
          {dateTime(String(lock?.owner?.started_at ?? ''))}
          {lock?.stale ? ' · 프로세스가 살아 있지 않습니다' : ''}). 서버가 실행 lock 을 물리적으로
          쥐고 있어 새 실행은 시작되지 않습니다.
        </Callout>
      ) : null}
      {lockError ? (
        <Callout tone="warn">
          실행 lock 을 조회하지 못했습니다 ({lockError.message}). 중복 실행을 막을 수 있는지 확인되지
          않았으므로, 다른 창에서 시뮬레이션이 돌고 있지 않은지 직접 확인하세요.
        </Callout>
      ) : null}

      {/* ================= 1단계 — 정책 고르기 ================= */}
      {step === 0 ? (
        <>
          <section className="section">
            <div className="section__head">
              <h2 className="section__title">1단계. 정책 고르기</h2>
              <p className="section__note">
                {policies === null
                  ? '정책 목록을 불러오는 중입니다.'
                  : `등록된 정책 ${int(policies.length)}건. 새로 만들거나, 정책 없이 돌려 비교 기준을 만들 수도 있습니다.`}
              </p>
            </div>

            {policiesError ? (
              <ErrorState
                title="정책 목록을 불러오지 못했습니다"
                body={`${policiesError.message} — 콘솔 API 가 떠 있는지 확인하세요.`}
                detail={policiesError.detail}
                action={
                  <Button icon={<RefreshIcon size={16} />} onClick={loadPolicies}>
                    다시 시도
                  </Button>
                }
              />
            ) : null}

            {policies === null ? (
              <Card>
                <SkeletonText lines={4} />
              </Card>
            ) : (
              <div className="table-wrap">
                <table className="table table--lead">
                  <caption className="visually-hidden">적용할 정책 고르기</caption>
                  <thead>
                    <tr>
                      <th scope="col">정책</th>
                      <th scope="col">유형</th>
                      <th scope="col" className="col-md">
                        시행 기간
                      </th>
                      <th scope="col" className="col-lg">
                        지급 기준
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="is-clickable" data-selected={isNew}>
                      <td>
                        <button
                          type="button"
                          className="table__rowbtn"
                          onClick={() => pick(NEW_POLICY)}
                          aria-current={isNew ? 'true' : undefined}
                        >
                          <span style={{ fontWeight: isNew ? 600 : 400 }}>+ 새 정책 만들어 주입</span>
                          <span className="cell-sub">
                            지급 분위·기간·사용처를 직접 정하고, 검증을 통과하면 이 실행에 그대로 넣습니다
                          </span>
                        </button>
                      </td>
                      <td>새로 작성</td>
                      <td className="col-md">직접 입력</td>
                      <td className="col-lg">직접 입력</td>
                    </tr>

                    {policies.map((p) => {
                      const active = p.id === choice;
                      return (
                        <tr key={p.id} className="is-clickable" data-selected={active}>
                          <td>
                            <button
                              type="button"
                              className="table__rowbtn"
                              onClick={() => pick(p.id)}
                              aria-current={active ? 'true' : undefined}
                            >
                              <span style={{ fontWeight: active ? 600 : 400 }}>{p.name}</span>
                              <span className="cell-sub cell-sub--md">
                                {p.effective_from} ~ {p.effective_until}
                              </span>
                              <span className="cell-sub cell-sub--lg">
                                지급 기준 {grantKey(p.grant_key_effective).label}
                              </span>
                            </button>
                          </td>
                          <td>{policyType(p.type).label}</td>
                          <td className="col-md num">
                            {p.effective_from} ~<br />
                            {p.effective_until}
                          </td>
                          <td className="col-lg">{grantKey(p.grant_key_effective).label}</td>
                        </tr>
                      );
                    })}

                    <tr className="is-clickable" data-selected={isControl}>
                      <td>
                        <button
                          type="button"
                          className="table__rowbtn"
                          onClick={() => pick(CONTROL)}
                          aria-current={isControl ? 'true' : undefined}
                        >
                          <span style={{ fontWeight: isControl ? 600 : 400 }}>무정책 (대조군)</span>
                          <span className="cell-sub">
                            정책 없이 같은 조건으로 돌려 비교 기준을 만듭니다
                          </span>
                        </button>
                      </td>
                      <td>대조군</td>
                      <td className="col-md">지정 없음</td>
                      <td className="col-lg">지급 없음</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <div className="row-between">
            <p className="card__note">
              {choice ? `고른 것: ${choiceName}` : '적용할 정책을 고르거나 새로 만드세요.'}
            </p>
            <Button variant="primary" disabled={!choice} onClick={next}>
              {isNew ? '다음: 정책 작성' : '다음: 설정 확인'}
            </Button>
          </div>
        </>
      ) : null}

      {/* ================= 2단계 — 정책 작성 ================= */}
      {step === 1 && draft ? (
        <>
          <section className="section">
            <div className="section__head">
              <h2 className="section__title">2단계. 정책 작성</h2>
              <p className="section__note">
                서버가 비어 있는 ID <span className="num">{draft.id}</span> 를 골라 두었습니다. 저장은
                다음 단계의 검증을 통과한 뒤에만 이뤄집니다.
              </p>
            </div>

            <div className="grid">
              <Card className="c6" title="무엇을 하는 정책인가">
                <TextField
                  label="정책 이름"
                  value={draft.name}
                  onChange={(e) => patchDraft({ name: e.currentTarget.value })}
                  help="보고서와 목록에 그대로 나옵니다."
                  error={draftErrors.name}
                />
                <SelectField
                  label="유형"
                  value={draft.type}
                  onChange={(e) => patchDraft({ type: e.currentTarget.value })}
                  options={POLICY_TYPES}
                  help="지원금이 아니면 지급액 대신 방문·체류 변화로 효과가 나타납니다."
                />
                <TextAreaField
                  label="설명"
                  rows={3}
                  value={draft.description}
                  onChange={(e) => patchDraft({ description: e.currentTarget.value })}
                  help="대상자에게 보이는 안내문입니다. 효과의 방향을 유도하는 표현은 쓰지 마세요."
                />
              </Card>

              <Card className="c6" title="언제·어디서">
                <TextField
                  label="시행 시작일"
                  type="date"
                  value={draft.effective_from}
                  onChange={(e) => patchDraft({ effective_from: e.currentTarget.value })}
                  help="보고서의 사전/사후를 나누는 기준일이 됩니다."
                  error={draftErrors.effective_from}
                />
                <TextField
                  label="시행 종료일"
                  type="date"
                  value={draft.effective_until}
                  onChange={(e) => patchDraft({ effective_until: e.currentTarget.value })}
                  error={draftErrors.effective_until}
                />
                <TextField
                  label="대상 지역"
                  value={(draft.target_districts ?? []).join(', ')}
                  onChange={(e) =>
                    patchDraft({
                      target_districts: e.currentTarget.value
                        .split(',')
                        .map((x) => x.trim())
                        .filter(Boolean),
                    })
                  }
                  help="쉼표로 구분합니다. 서울 전체면 '서울특별시', 자치구 단위면 '강남구' 처럼 적습니다."
                />
                <label className="row" style={{ gap: 'var(--sp-2)' }}>
                  <input
                    type="checkbox"
                    checked={Boolean(draft.poi_restricted)}
                    onChange={(e) => patchDraft({ poi_restricted: e.currentTarget.checked })}
                  />
                  <span>쿠폰 가맹점에서만 사용 가능</span>
                </label>
              </Card>
            </div>

            <Card
              title="분위별 지급액"
              note="소비 10분위 기준입니다. 0 인 구간은 지급 대상이 아닙니다."
            >
              <div className="grid">
                {DECILES.map((d) => (
                  <div key={d} className="c3">
                  <TextField
                    label={`${d}분위`}
                    type="number"
                    inputMode="numeric"
                    min={0}
                    step={10000}
                    value={String(draft.decile_grants?.[d] ?? 0)}
                    onChange={(e) =>
                      patchDraft({
                        decile_grants: {
                          ...(draft.decile_grants ?? {}),
                          [d]: Number(e.currentTarget.value || 0),
                        },
                      })
                    }
                  />
                  </div>
                ))}
              </div>
              {draftErrors.grants ? (
                <p className="field__error">
                  <AlertCircleIcon size={14} />
                  <span>{draftErrors.grants}</span>
                </p>
              ) : null}
              {grantBars.length > 0 ? <BarList items={grantBars} /> : null}
            </Card>

            <Card
              title="대상 업종"
              note="비우면 모든 업종에서 쓸 수 있습니다. 고르면 보고서의 처치군이 이 업종들로 정해집니다."
            >
              <div className="row" style={{ flexWrap: 'wrap', gap: 'var(--sp-3)' }}>
                {L1_CATEGORIES.map((name) => {
                  const on = (draft.benefit_categories ?? []).includes(name);
                  return (
                    <label key={name} className="row" style={{ gap: 'var(--sp-2)' }}>
                      <input type="checkbox" checked={on} onChange={() => toggleCategory(name)} />
                      <span>{name}</span>
                    </label>
                  );
                })}
              </div>
            </Card>

            <Disclosure title="JSON 미리보기" meta={`${draft.id}.json`}>
              <pre className="code">{JSON.stringify(draft, null, 2)}</pre>
              <p className="card__note wrap">
                이 내용이 <span className="num">data/neo4j_load/policies/{draft.id}.json</span> 으로
                저장됩니다. 저장은 서버의 사전 점검(preflight)을 통과한 경우에만 이뤄집니다.
              </p>
            </Disclosure>
          </section>

          <div className="row-between">
            <Button icon={<ArrowLeftIcon size={18} />} onClick={() => back(0)}>
              뒤로: 정책 고르기
            </Button>
            <Button variant={draftOk ? 'primary' : 'secondary'} onClick={next}>
              다음: 설정 확인
            </Button>
          </div>
          {attempted && !draftOk ? (
            <p className="field__error">
              <AlertCircleIcon size={14} />
              <span>정책 내용에 고칠 곳이 있습니다 — 위 입력란의 안내를 확인하세요.</span>
            </p>
          ) : null}
        </>
      ) : null}

      {/* ================= 3단계 — 설정 확인 ================= */}
      {step === 2 ? (
        <>
          <section className="section">
            <div className="section__head">
              <h2 className="section__title">{isNew ? '3' : '2'}단계. 설정 확인</h2>
              <p className="section__note">
                고른 것: {choiceName}
                {activePolicy ? ` · ${policyType(activePolicy.type).label}` : ''}
                {activePolicy
                  ? ` · ${activePolicy.poi_restricted ? '쿠폰 가맹점에서만 사용' : '사용처 제한 없음'}`
                  : ''}
              </p>
            </div>

            <div className="grid">
              <Card className="c6" title="실행 이름과 기간">
                <TextField
                  label="실행 이름 (run id)"
                  value={runId}
                  onChange={(e) => setRunId(e.currentTarget.value)}
                  onBlur={() => setTouched((t) => ({ ...t, runId: true }))}
                  help="산출물 디렉터리와 보고서에 그대로 쓰입니다 — 예: BASE_0810."
                  error={show('runId')}
                />
                <TextField
                  label="시작일"
                  type="date"
                  value={startDay}
                  onChange={(e) => setStartDay(e.currentTarget.value)}
                  onBlur={() => setTouched((t) => ({ ...t, startDay: true }))}
                  help={
                    activePolicy?.effective_from
                      ? `고른 정책의 시행 시작일(${activePolicy.effective_from})을 기본값으로 넣었습니다.`
                      : '대조군은 기준이 될 정책이 없어 기본값을 넣지 않았습니다.'
                  }
                  error={show('startDay')}
                />
                <TextField
                  label="기간 (일)"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={365}
                  value={days}
                  onChange={(e) => setDays(e.currentTarget.value)}
                  onBlur={() => setTouched((t) => ({ ...t, days: true }))}
                  help={endDay ? `${startDay} 부터 ${endDay} 까지 ${int(dayCount)}일을 돕니다.` : undefined}
                  error={show('days')}
                />
                <TextField
                  label="하루 대상자 수 (명)"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  value={agents}
                  onChange={(e) => setAgents(e.currentTarget.value)}
                  onBlur={() => setTouched((t) => ({ ...t, agents: true }))}
                  help="기존 실행 BASE·FINAL 과 같은 200명을 기본값으로 넣었습니다."
                  error={show('agents')}
                />
              </Card>

              <Card
                className="c6"
                title="분위별 지급액"
                note={
                  activePolicy && grantBars.length > 0
                    ? `${grantKey(activePolicy.grant_key ?? 'spend_decile').label} ${grantBars.length}구간`
                    : undefined
                }
              >
                {grantBars.length > 0 ? (
                  <BarList items={grantBars} />
                ) : (
                  <EmptyState
                    fill
                    title="지급액이 없습니다"
                    body={
                      isControl
                        ? '무정책 대조군이라 대상자에게 지급하는 금액이 없습니다. 같은 조건에서 정책만 뺀 결과를 만듭니다.'
                        : `${activePolicy ? policyType(activePolicy.type).label : '이'} 정책이라 지급하는 금액이 없습니다. 효과는 소비 금액이 아니라 방문·체류 변화로 나타납니다.`
                    }
                  />
                )}
              </Card>
            </div>
          </section>

          <section className="section">
            <div className="section__head">
              <h2 className="section__title">검증</h2>
              <p className="section__note">
                {needsValidation
                  ? '서버의 사전 점검(policy_preflight.py)을 실제로 실행해 정책이 시뮬레이션에 그대로 배선되는지 확인합니다.'
                  : '대조군에는 검증할 정책 파일이 없습니다.'}
              </p>
            </div>
            <Card>
              {!needsValidation ? (
                <p className="card__note">
                  무정책 대조군이라 검증할 정책 파일이 없습니다. 기간과 대상자 수만 맞으면 다음 단계로
                  갈 수 있습니다.
                </p>
              ) : !validation ? (
                <div className="stack-sm">
                  <p className="card__note wrap">
                    사전 점검은 <strong>시뮬레이션을 시작하지 않습니다.</strong> 정책 표현이 올바른지,
                    지급 구간이 시뮬레이션 계층과 맞는지만 확인합니다.
                  </p>
                  <div>
                    <Button
                      variant="primary"
                      busy={validating}
                      busyLabel="검증하는 중"
                      onClick={runValidation}
                    >
                      검증 실행
                    </Button>
                  </div>
                  {validationError ? (
                    <ErrorState
                      title="검증을 실행하지 못했습니다"
                      body={validationError.message}
                      detail={validationError.detail}
                    />
                  ) : null}
                  {attempted ? (
                    <p className="field__error">
                      <AlertCircleIcon size={14} />
                      <span>검증을 먼저 실행해야 실행 단계로 갈 수 있습니다.</span>
                    </p>
                  ) : null}
                </div>
              ) : (
                <>
                  <div className="verdict">
                    {failing.length > 0 ? (
                      <AlertCircleIcon size={24} style={{ color: 'var(--danger)' }} />
                    ) : (
                      <CheckCircleIcon size={24} style={{ color: 'var(--ok)' }} />
                    )}
                    <span className="verdict__text">
                      {failing.length > 0
                        ? `오류 ${int(failing.length)}건 — 정책을 고친 뒤 다시 확인하세요`
                        : '이 정책은 바로 실행할 수 있습니다'}
                    </span>
                  </div>

                  <p className="tally">
                    <span>
                      통과 <span className="tally__n">{int(passing.length)}</span>
                    </span>
                    <span>
                      확인 필요 <span className="tally__n">{int(warning.length)}</span>
                    </span>
                    <span>
                      오류 <span className="tally__n">{int(failing.length)}</span>
                    </span>
                  </p>

                  {failing.length + warning.length === 0 ? (
                    <p className="card__note">확인이 필요한 항목이 없습니다.</p>
                  ) : (
                    <ul className="checks">
                      {failing.map((c) => (
                        <li
                          className={`check check--fail${lead === 'fail' ? ' check--lead' : ''}`}
                          key={`f-${c.message}`}
                        >
                          <GradeIcon grade="fail" />
                          <span className="wrap">{c.message}</span>
                        </li>
                      ))}
                      {warning.map((c) => (
                        <li
                          className={`check check--warn${lead === 'warn' ? ' check--lead' : ''}`}
                          key={`w-${c.message}`}
                        >
                          <GradeIcon grade="warn" />
                          <span className="wrap">{c.message}</span>
                        </li>
                      ))}
                    </ul>
                  )}

                  <p className="card__note wrap">
                    실행 명령 <span className="num">{validation.command.join(' ')}</span>
                  </p>

                  {!validation.db_wiring_checked ? (
                    <p className="card__note wrap">
                      데이터베이스 연결 정보가 없어 “정책이 실제로 대상자에게 보이는지”는 아직 확인하지
                      못했습니다. 이 항목은 통과가 아니라 <strong>미확인</strong>입니다.
                    </p>
                  ) : null}

                  <div className="row" style={{ gap: 'var(--sp-3)' }}>
                    <Button icon={<RefreshIcon size={16} />} busy={validating} onClick={runValidation}>
                      다시 검증
                    </Button>
                  </div>

                  <Disclosure title="프롬프트 미리보기" meta="대상자에게 보이는 문장">
                    <pre className="code">{validation.prompt_preview || '(없음)'}</pre>
                  </Disclosure>
                </>
              )}
            </Card>
          </section>

          <div className="section">
            <div className="row-between">
              <Button icon={<ArrowLeftIcon size={18} />} onClick={() => back(isNew ? 1 : 0)}>
                뒤로: {isNew ? '정책 작성' : '정책 고르기'}
              </Button>
              <Button variant={canGoLaunch ? 'primary' : 'secondary'} onClick={next}>
                다음: 실행
              </Button>
            </div>
            {attempted && !canGoLaunch ? (
              <p className="field__error">
                <AlertCircleIcon size={14} />
                <span>
                  {settingsOk
                    ? '검증을 통과해야 실행 단계로 갈 수 있습니다 — 위의 검증 결과를 확인하세요.'
                    : '설정에 고칠 곳이 있습니다 — 위 입력란의 안내를 확인하세요.'}
                </span>
              </p>
            ) : null}
          </div>
        </>
      ) : null}

      {/* ================= 4단계 — 실행 ================= */}
      {step === 3 ? (
        <>
          <section className="section">
            <div className="section__head">
              <h2 className="section__title">{isNew ? '4' : '3'}단계. 실행</h2>
              <p className="section__note">아래 설정으로 시뮬레이션을 시작합니다.</p>
            </div>
            <Card>
              <dl className="dl">
                <div className="dl__item">
                  <dt className="dl__k">실행 이름</dt>
                  <dd className="dl__v num">{runId}</dd>
                </div>
                <div className="dl__item">
                  <dt className="dl__k">적용 정책</dt>
                  <dd className="dl__v">
                    {choiceName}
                    {isNew ? (
                      <>
                        {' '}
                        <Badge tone="info">새 정책 주입</Badge>
                      </>
                    ) : null}
                  </dd>
                </div>
                <div className="dl__item">
                  <dt className="dl__k">기간</dt>
                  <dd className="dl__v num">
                    {startDay} ~ {endDay ?? '—'} ({int(dayCount)}일)
                  </dd>
                </div>
                <div className="dl__item">
                  <dt className="dl__k">하루 대상자 수</dt>
                  <dd className="dl__v num">{int(agentCount)}명</dd>
                </div>
                <div className="dl__item">
                  <dt className="dl__k">검증</dt>
                  <dd className="dl__v">
                    {needsValidation
                      ? `통과 ${int(passing.length)} · 확인 필요 ${int(warning.length)} · 오류 ${int(failing.length)}`
                      : '대조군이라 검증할 정책 파일이 없습니다'}
                  </dd>
                </div>
              </dl>
            </Card>
          </section>

          {/* 기준 B8 */}
          <section className="section">
            {locked ? (
              <Callout tone="warn">
                실행 lock 을 다른 실행이 쥐고 있어 시작할 수 없습니다. 서버가 막기 때문에 이 버튼을
                눌러도 기존 실행이 죽지 않습니다.
              </Callout>
            ) : (
              <Callout>
                실행 lock 이 비어 있습니다. 시작하면 서버가 lock 을 잡고, 그 뒤로는 다른 요청이
                물리적으로 실행을 시작할 수 없습니다.
              </Callout>
            )}
            <div>
              <Disclosure
                title="중복 실행으로 실행이 죽은 기록"
                meta={`${lockEvidence.timeline.length}줄`}
              >
                <p className="card__note wrap">{lockEvidence.note}</p>
                <pre className="code">{lockEvidence.timeline.map((t) => t.text).join('\n')}</pre>
                <p className="card__note wrap">
                  출처: <span className="num">{lockEvidence.source}</span>
                  {lockEvidence.killed_run ? ` · 죽은 실행 ${lockEvidence.killed_run.run_id}` : ''}
                </p>
              </Disclosure>
            </div>
          </section>

          <section className="section">
            <div className="row-between">
              <Button icon={<ArrowLeftIcon size={18} />} onClick={() => back(2)}>
                뒤로: 설정 확인
              </Button>
              <Button
                variant="primary"
                disabled={locked || Boolean(launched)}
                busy={launching}
                busyLabel="시작하는 중"
                onClick={launch}
              >
                실행 시작
              </Button>
            </div>

            <div role="status" aria-live="polite">
              {launched ? (
                <Card title="실행을 시작했습니다">
                  <p className="card__note wrap">
                    {launched.injected
                      ? '새 정책이 검증을 통과해 저장되었고, 그 정책으로 실행이 시작되었습니다.'
                      : '선택한 정책으로 실행이 시작되었습니다.'}{' '}
                    산출물이 쌓이면 실행 모니터에서 진행을 볼 수 있습니다.
                  </p>
                  <pre className="code">{JSON.stringify(launched.lock ?? {}, null, 2)}</pre>
                </Card>
              ) : null}

              {launchError ? (
                <ErrorState
                  title="실행을 시작하지 못했습니다"
                  body={launchError.message}
                  detail={launchError.detail}
                />
              ) : null}
            </div>

            <p className="card__note wrap">
              실행 명령은 서버 운영자가 <span className="num">SIM_RUN_COMMAND_JSON</span> 으로
              구성합니다. 콘솔은 임의 명령을 받지 않으며, 실행 파라미터는 환경변수로만 전달됩니다.
            </p>
          </section>
        </>
      ) : null}
    </div>
  );
}
