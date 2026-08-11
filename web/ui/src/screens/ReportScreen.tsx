/**
 * 최종 보고서 — 라우트 `/runs/:runId/report`.
 *
 * 이 화면은 두 가지 일을 한다. **그 둘을 한 화면에 욱여넣지 않는다.**
 *   1) 보고서를 만든다 — 엔진·기간·절을 고르고 job 을 띄우고 진행을 본다
 *   2) 만들어진 보고서를 읽는다 — 본문을 이 페이지에 그대로 편다
 * 아직 만든 적이 없으면 (1)만, 하나라도 있으면 (2)를 먼저 보여준다.
 *
 * **iframe 을 쓰지 않는다.** 페이지 안에 또 하나의 스크롤 영역이 생기면
 * 어느 쪽을 굴리는지 알 수 없다 (SKILL §5 `scroll-behavior`). 본문을 이 페이지에
 * 그대로 펴고, 보고서 CSS 는 `.reportdoc` 안으로 가둔다.
 *
 * **되는 척하지 않는다 (기준 B1).** API 가 없으면 없다고 적는다. 진행률을
 * 지어내지 않고, 서버가 준 job 상태와 로그만 보여준다.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRun } from '../app/RunContext';
import { Badge } from '../components/Badge';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Callout, EmptyState, ErrorState, SkeletonText } from '../components/Feedback';
import { SelectField, TextField } from '../components/Field';
import { AlertCircleIcon, CheckCircleIcon, DownloadIcon, RefreshIcon } from '../components/Icon';
import { api, artifactUrl } from '../lib/api';
import type { ApiErrorShape, LlmStatus, ReportCatalog, ReportJob } from '../lib/api';
import { dateTime, int } from '../lib/format';
import { loadReportDoc, SCOPE } from '../lib/reportDoc';
import type { ReportDoc } from '../lib/reportDoc';
import { READ_ONLY } from '../lib/runtime';

/** 콘솔은 라이트 테마 하나뿐이다. 보고서도 같은 테마로 고정해 두 문서가 섞이지 않게 한다 */
const DOC_THEME = 'light';

const POLL_MS = 1500;

/**
 * 스코프만으로는 안 되는 보정.
 * 보고서는 단독 문서로 만들어져 화면 전체를 자기 것으로 여긴다. 여기서는
 * 문서가 **페이지 안의 한 덩어리**여야 하므로 흐름에 눕힌다.
 */
const DOC_FIX = `
.reportdoc { max-width: 100%; }
.reportdoc .wrap { padding: 0 !important; max-width: 100% !important; }
.reportdoc [class*="sidebar"] {
  position: static !important;
  width: auto !important;
  height: auto !important;
  max-height: none !important;
  border-right: 0 !important;
  border-bottom: 1px solid var(--border) !important;
  margin-bottom: var(--sp-5) !important;
}
.reportdoc .layout {
  display: block !important;
  height: auto !important;
  min-height: 0 !important;
  overflow: visible !important;
}
/* 문서 안쪽에 또 스크롤이 생기지 않게 — 중첩 스크롤 금지 (§5) */
.reportdoc * { overflow-y: visible !important; }
/* 넓은 표만 예외로 가로 스크롤 허용. 잘리는 것보다 낫다 */
.reportdoc .tablewrap, .reportdoc table { overflow-x: auto; }
.reportdoc img, .reportdoc figure, .reportdoc canvas { max-width: 100%; }
.reportdoc svg { max-width: 100%; height: auto; }
/* 문서가 자기 테마 버튼을 들고 있어도 콘솔 안에서는 동작하지 않는다 (스크립트 제거됨) */
.reportdoc .themebtn { display: none !important; }
`;

/* --- 작은 상태 도우미 -------------------------------------------------------- */

type Load<T> =
  | { status: 'loading' }
  | { status: 'error'; error: ApiErrorShape }
  | { status: 'ready'; data: T };

function errorOf(value: unknown): ApiErrorShape {
  if (value && typeof value === 'object' && 'message' in value) return value as ApiErrorShape;
  return { message: String(value) };
}

const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;

function addDays(iso: string, n: number): string | null {
  if (!ISO_DAY.test(iso)) return null;
  const [y, m, d] = iso.split('-').map(Number);
  const date = new Date(Date.UTC(y, m - 1, d));
  if (Number.isNaN(date.getTime())) return null;
  date.setUTCDate(date.getUTCDate() + n);
  const p = (x: number) => String(x).padStart(2, '0');
  return `${date.getUTCFullYear()}-${p(date.getUTCMonth() + 1)}-${p(date.getUTCDate())}`;
}

const STAGE_LABEL: Record<string, string> = {
  queued: '대기',
  starting: '시작하는 중',
  running: '실행 중',
  scanning: '산출물 스캔',
  verifying: '일관성 검증',
  narrating: '해설 생성',
  rendering: '그림·표 렌더',
  writing: '파일 저장',
  conditions: '조건 요약',
  analysis: '분석 계산',
  interview: '인터뷰 분석',
  ready: '완료',
  failed: '실패',
};

/* --- 화면 ------------------------------------------------------------------- */

/** 목록에서 보고서를 고를 때 읽는 것은 파일 이름이 아니라 언제 만든 것인가다 */
function reportLabel(item: { created_at?: string }): string {
  const at = item.created_at ? new Date(item.created_at) : null;
  if (!at || Number.isNaN(at.getTime())) return '분석 보고서';
  const two = (n: number) => String(n).padStart(2, '0');
  return `${at.getFullYear()}-${two(at.getMonth() + 1)}-${two(at.getDate())} ${two(at.getHours())}:${two(at.getMinutes())} 생성`;
}

function periodLabel(item: { start?: string; days?: number }): string {
  return item.start && item.days ? `${item.start} 부터 ${item.days}일` : '—';
}

export function ReportScreen() {
  const run = useRun();
  const boundPolicy = run.policy.items[0]?.id ?? null;

  const [policyId, setPolicyId] = useState<string | null>(boundPolicy);
  const [policyOptions, setPolicyOptions] = useState<Array<{ value: string; label: string }>>([]);
  const [catalog, setCatalog] = useState<Load<ReportCatalog>>({ status: 'loading' });
  const [llm, setLlm] = useState<LlmStatus | null>(null);
  const [pinging, setPinging] = useState(false);

  /* 생성 요청 입력 */
  const [start, setStart] = useState(run.firstDay ?? '');
  const [days, setDays] = useState(String(run.daysPresent || 7));
  const [policyFrom, setPolicyFrom] = useState('');
  const [engine, setEngine] = useState<'v2' | 'dasol'>('v2');
  const [sections, setSections] = useState<string[] | null>(null);
  const [useLlm, setUseLlm] = useState(true);

  /* job */
  const [job, setJob] = useState<ReportJob | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<ApiErrorShape | null>(null);

  /* 열람 */
  const [openPath, setOpenPath] = useState<string | null>(null);
  const [doc, setDoc] = useState<Load<ReportDoc> | null>(null);
  const docRef = useRef<HTMLDivElement | null>(null);

  /* --- 정책 목록 --------------------------------------------------------- */
  useEffect(() => {
    let alive = true;
    api
      .listPolicies()
      .then((list) => {
        if (!alive) return;
        setPolicyOptions(list.items.map((p) => ({ value: p.id, label: `${p.id} · ${p.name}` })));
        setPolicyId((current) => current ?? list.items[0]?.id ?? null);
      })
      .catch(() => {
        /* 정책 목록이 없으면 catalog 요청이 그 사실을 더 정확하게 알려준다 */
      });
    return () => {
      alive = false;
    };
  }, []);

  /* --- 카탈로그 ---------------------------------------------------------- */
  const loadCatalog = useCallback(() => {
    if (!policyId) return;
    setCatalog({ status: 'loading' });
    api
      .reportCatalog(run.id, policyId)
      .then((data) => {
        setCatalog({ status: 'ready', data });
        setLlm(data.llm);
        setPolicyFrom((current) => current || data.policy.effective_from || '');
        setEngine(data.engine_v2.available ? 'v2' : 'dasol');
        setSections((current) =>
          current ?? data.v2_sections.filter((s) => s.applicable).map((s) => s.id),
        );
      })
      .catch((error) => setCatalog({ status: 'error', error: errorOf(error) }));
  }, [run.id, policyId]);

  useEffect(loadCatalog, [loadCatalog]);

  /* --- 진행 중인 job 이어받기 ------------------------------------------- */
  useEffect(() => {
    let alive = true;
    api
      .listReportJobs(run.id)
      .then((list) => {
        if (!alive) return;
        const latest = list.items[0];
        if (latest) setJob(latest);
        if (latest?.state === 'completed') setOpenPath(latest.output_path);
      })
      .catch(() => {
        /* 서버가 없으면 카탈로그 오류가 이미 그 사실을 말한다 */
      });
    return () => {
      alive = false;
    };
  }, [run.id]);

  /* --- job 폴링 ---------------------------------------------------------- */
  useEffect(() => {
    if (!job || (job.state !== 'running' && job.state !== 'queued')) return;
    let alive = true;
    const timer = window.setInterval(() => {
      api
        .getReportJob(job.job_id)
        .then((next) => {
          if (!alive) return;
          setJob(next);
          if (next.state === 'completed') {
            setOpenPath(next.output_path);
            loadCatalog();
          }
        })
        .catch(() => {
          /* 일시적인 실패는 다음 주기에 다시 시도한다 */
        });
    }, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [job, loadCatalog]);

  /* --- 보고서 본문 ------------------------------------------------------- */
  useEffect(() => {
    if (!openPath) {
      setDoc(null);
      return;
    }
    let alive = true;
    setDoc({ status: 'loading' });
    loadReportDoc(artifactUrl(openPath))
      .then((value) => alive && setDoc({ status: 'ready', data: value }))
      .catch((error) => alive && setDoc({ status: 'error', error: errorOf(error) }));
    return () => {
      alive = false;
    };
  }, [openPath]);

  /* 문서가 들어오면 테마를 콘솔과 맞춘다 */
  useEffect(() => {
    const node = docRef.current?.querySelector<HTMLElement>('.doc');
    if (node) node.setAttribute('data-theme', DOC_THEME);
  }, [doc]);

  /* --- 파생값 ------------------------------------------------------------ */
  const dayCount = /^\d+$/.test(days.trim()) ? Number(days.trim()) : null;
  const endDay = start && dayCount ? addDays(start, dayCount - 1) : null;
  const ready = catalog.status === 'ready' ? catalog.data : null;
  const lockedByRun = ready?.report_lock.locked ?? false;
  const engineInfo = ready?.engines.find((e) => e.id === engine) ?? null;

  const errors = useMemo(() => {
    const out: Record<string, string> = {};
    if (!ISO_DAY.test(start)) out.start = '시작일이 비어 있거나 형식이 어긋납니다 — 2025-07-21 처럼 입력하세요.';
    if (dayCount === null || dayCount < 1 || dayCount > 365) {
      out.days = '기간은 1일 이상 365일 이하의 정수여야 합니다.';
    }
    if (policyFrom && !ISO_DAY.test(policyFrom)) {
      out.policyFrom = '정책 시행일 형식이 어긋납니다 — 2025-07-28 처럼 입력하세요.';
    }
    return out;
  }, [start, dayCount, policyFrom]);

  const canSubmit =
    !READ_ONLY &&
    Boolean(policyId) &&
    Object.keys(errors).length === 0 &&
    Boolean(engineInfo?.available) &&
    !lockedByRun &&
    !starting &&
    job?.state !== 'running' &&
    job?.state !== 'queued';

  const selected = sections ?? [];
  const requiredIds = new Set(ready?.v2_required ?? []);

  function toggleSection(id: string) {
    if (requiredIds.has(id)) return;
    setSections((current) => {
      const base = current ?? [];
      return base.includes(id) ? base.filter((x) => x !== id) : [...base, id];
    });
  }

  function submit() {
    if (!policyId || dayCount === null) return;
    setStarting(true);
    setStartError(null);
    api
      .startReportJob({
        run_id: run.id,
        policy_id: policyId,
        start,
        days: dayCount,
        policy_from: policyFrom || null,
        analyses: engine === 'v2' ? selected : [],
        include_interview: false,
        engine,
        use_llm: useLlm,
      })
      .then((next) => {
        setJob(next);
        setOpenPath(null);
      })
      .catch((error) => setStartError(errorOf(error)))
      .finally(() => setStarting(false));
  }

  function ping() {
    setPinging(true);
    api
      .llmPing()
      .then(setLlm)
      .catch((error) => setLlm({ ...(llm as LlmStatus), reachable: false, error: errorOf(error).message }))
      .finally(() => setPinging(false));
  }

  /* --- 렌더 -------------------------------------------------------------- */

  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">보고서</h1>
          <p className="pagehead__purpose">
            이 실행의 산출물에서 직접 계산한 분석 보고서를 만들고 읽습니다. 모든 수치에는 어떤
            파일에서 나왔는지가 함께 적힙니다.
          </p>
        </div>
        {openPath ? (
          <div className="pagehead__actions">
            <Button
              icon={<DownloadIcon size={18} />}
              onClick={() => {
                const a = document.createElement('a');
                a.href = artifactUrl(openPath);
                a.download = `${run.id}_분석보고서.html`;
                document.body.appendChild(a);
                a.click();
                a.remove();
              }}
            >
              보고서 내려받기
            </Button>
          </div>
        ) : null}
      </header>

      {catalog.status === 'error' ? (
        <ErrorState
          title="보고서 API 에 연결하지 못했습니다"
          body={`${catalog.error.message} — 콘솔 API(python -m uvicorn web.api.app:app)가 떠 있는지 확인하세요. 서버 없이 보고서를 만들 수는 없습니다.`}
          detail={catalog.error.detail}
          action={
            <Button icon={<RefreshIcon size={16} />} onClick={loadCatalog}>
              다시 시도
            </Button>
          }
        />
      ) : null}

      {catalog.status === 'loading' ? (
        <Card title="상태를 확인하는 중">
          <SkeletonText lines={3} />
        </Card>
      ) : null}

      {ready ? (
        <>
          {!READ_ONLY ? (
          <>
          {/* ---------- 1단계. 무엇으로 만드는가 ---------- */}
          <section className="section">
            <div className="section__head">
              <h2 className="section__title">1단계. 대상과 기간</h2>
              <p className="section__note">
                실행 {run.id} · 기록된 일자 {int(ready.run.days_present.length)}일 · 상태{' '}
                {ready.run.status === 'completed' ? '완료' : '중단됨'}
              </p>
            </div>
            <div className="grid">
              <Card className="c6" title="분석 대상">
                <SelectField
                  label="정책"
                  value={policyId ?? ''}
                  onChange={(e) => setPolicyId(e.currentTarget.value)}
                  options={
                    policyOptions.length > 0
                      ? policyOptions
                      : [{ value: ready.policy.id, label: `${ready.policy.id} · ${ready.policy.name}` }]
                  }
                  help={
                    boundPolicy
                      ? `이 실행의 결제 기록에 남은 정책은 ${boundPolicy} 입니다.`
                      : '이 실행의 결제 기록에서 적용 정책을 찾지 못했습니다 — 직접 고르세요.'
                  }
                />
                <TextField
                  label="정책 시행일"
                  type="date"
                  value={policyFrom}
                  onChange={(e) => setPolicyFrom(e.currentTarget.value)}
                  help="이 날짜를 기준으로 사전/사후를 나눕니다. 비우면 정책 파일의 시행일을 씁니다. 이 값이 없으면 이중차분을 계산할 수 없습니다."
                  error={errors.policyFrom}
                />
              </Card>
              <Card className="c6" title="분석 기간">
                <TextField
                  label="시작일"
                  type="date"
                  value={start}
                  onChange={(e) => setStart(e.currentTarget.value)}
                  help={
                    ready.run.days_present.length > 0
                      ? `이 실행에 기록이 있는 구간: ${ready.run.days_present[0]} ~ ${
                          ready.run.days_present[ready.run.days_present.length - 1]
                        }`
                      : '이 실행에는 일자 기록이 없습니다.'
                  }
                  error={errors.start}
                />
                <TextField
                  label="기간 (일)"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={365}
                  value={days}
                  onChange={(e) => setDays(e.currentTarget.value)}
                  help={endDay ? `${start} 부터 ${endDay} 까지 ${int(dayCount)}일을 봅니다.` : undefined}
                  error={errors.days}
                />
              </Card>
            </div>
          </section>

          {/* ---------- 2단계. 무엇을 담는가 ---------- */}
          <section className="section">
            <div className="section__head">
              <h2 className="section__title">2단계. 보고서 구성</h2>
              <p className="section__note">
                보고서에 담을 내용을 고릅니다. 계산할 수 없는 항목은 이유와 함께 잠깁니다 — 화면이 임의로
                이중차분을 강제하지 않습니다.
              </p>
            </div>

            <Card
                title="담을 절"
                note={`선택 ${int(selected.length)}개 · 항상 포함 ${int(ready.v2_required.length)}개`}
                flush
              >
                <div className="table-wrap">
                  <table className="table">
                    <caption className="visually-hidden">보고서에 담을 절 고르기</caption>
                    <thead>
                      <tr>
                        <th scope="col">담기</th>
                        <th scope="col">절</th>
                        <th scope="col">내용</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ready.v2_sections.map((item) => {
                        const on = selected.includes(item.id) || requiredIds.has(item.id);
                        return (
                          <tr key={item.id} data-selected={on}>
                            <td>
                              <input
                                type="checkbox"
                                checked={on}
                                disabled={!item.applicable || requiredIds.has(item.id)}
                                onChange={() => toggleSection(item.id)}
                                aria-label={`${item.label} 담기`}
                              />
                            </td>
                            <td>
                              <strong>{item.label}</strong>
                              {requiredIds.has(item.id) ? (
                                <span className="cell-sub">항상 포함 — 근거 없는 보고서를 만들지 않습니다</span>
                              ) : null}
                            </td>
                            <td className="wrap">
                              {item.applicable ? (
                                item.description
                              ) : (
                                <span className="check check--warn">
                                  <AlertCircleIcon size={14} />
                                  <span className="wrap">{item.disabled_reason}</span>
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
            </Card>

            <Card title="해설 LLM" note="숫자는 계산 결과에서만 옵니다. 해설 문장만 모델이 씁니다.">
              <dl className="dl">
                <div className="dl__item">
                  <dt className="dl__k">제공자</dt>
                  <dd className="dl__v">
                    {llm?.provider ?? '—'}
                    {llm?.model ? ` · ${llm.model}` : ''}{' '}
                    {llm?.configured ? <Badge tone="ok">설정됨</Badge> : <Badge tone="warn">미설정</Badge>}
                  </dd>
                </div>
                {llm?.reason ? (
                  <div className="dl__item">
                    <dt className="dl__k">안내</dt>
                    <dd className="dl__v wrap">{llm.reason}</dd>
                  </div>
                ) : null}
                {llm?.reachable !== undefined ? (
                  <div className="dl__item">
                    <dt className="dl__k">연결 확인</dt>
                    <dd className="dl__v">
                      {llm.reachable ? (
                        <>
                          <CheckCircleIcon size={16} /> 응답 확인 ({int(llm.latency_ms ?? null)}ms)
                        </>
                      ) : (
                        <>
                          <AlertCircleIcon size={16} /> {llm.error ?? '응답 없음'}
                        </>
                      )}
                    </dd>
                  </div>
                ) : null}
              </dl>
              {!READ_ONLY ? (
                <div className="row" style={{ gap: 'var(--sp-3)', marginTop: 'var(--sp-3)' }}>
                  <Button busy={pinging} busyLabel="확인하는 중" onClick={ping}>
                    연결 확인
                  </Button>
                  <label className="row" style={{ gap: 'var(--sp-2)' }}>
                    <input
                      type="checkbox"
                      checked={useLlm}
                      onChange={(e) => setUseLlm(e.currentTarget.checked)}
                    />
                    <span>이 보고서에 LLM 해설 사용</span>
                  </label>
                </div>
              ) : null}
              {!llm?.configured ? (
                <p className="card__note wrap">
                  해설 모델이 연결되지 않아도 보고서는 그대로 만들어집니다. 해설 문장만
                  정해진 서술로 대체됩니다.
                </p>
              ) : null}
            </Card>
          </section>

          {/* ---------- 3단계. 생성 ---------- */}
          <section className="section">
            <div className="section__head">
              <h2 className="section__title">3단계. 생성</h2>
              <p className="section__note">
                보고서 생성은 한 번에 하나만 돕니다. 실행 중인 시뮬레이션이 있으면 시작하지 않습니다.
              </p>
            </div>

            {lockedByRun ? (
              <Callout tone="warn">
                이미 다른 보고서 생성 job 이 실행 중입니다. 끝난 뒤에 다시 시도하세요.
              </Callout>
            ) : null}

            {ready.snapshot.ready === false ? (
              <Callout tone="warn">
                이 실행으로는 보고서를 만들 수 없습니다.
              </Callout>
            ) : null}

            <div className="row-between">
              <p className="card__note wrap">
                {engineInfo?.available
                  ? `${start} 부터 ${int(dayCount)}일을 분석합니다.`
                  : (engineInfo?.reason ?? '사용할 수 있는 엔진이 없습니다.')}
              </p>
              <Button variant="primary" disabled={!canSubmit} busy={starting} busyLabel="시작하는 중" onClick={submit}>
                보고서 생성
              </Button>
            </div>

            {startError ? (
              <ErrorState
                title="보고서 생성을 시작하지 못했습니다"
                body={startError.message}
                detail={startError.detail}
              />
            ) : null}

            {job ? <JobPanel job={job} onOpen={() => setOpenPath(job.output_path)} /> : null}
          </section>
          </>
          ) : (
            <Callout>읽기 전용으로 배포되어 새 보고서를 생성할 수 없습니다. 아래의 기존 보고서를 열람할 수 있습니다.</Callout>
          )}

          {/* ---------- 이전 보고서 ---------- */}
          {ready.report_artifacts.length > 0 ? (
            <section className="section">
              <div className="section__head">
                <h2 className="section__title">만들어진 보고서</h2>
                <p className="section__note">
                  {int(ready.report_artifacts.length)}건 · 클릭하면 아래에 본문이 펼쳐집니다.
                </p>
              </div>
              <div className="table-wrap">
                <table className="table">
                  <caption className="visually-hidden">만들어진 보고서 목록</caption>
                  <thead>
                    <tr>
                      <th scope="col">보고서</th>
                      <th scope="col" className="col-md">
                        분석 기간
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {ready.report_artifacts.map((item) => (
                      <tr key={item.path} className="is-clickable" data-selected={openPath === item.path}>
                        <td>
                          <button
                            type="button"
                            className="table__rowbtn"
                            onClick={() => setOpenPath(item.path)}
                            aria-current={openPath === item.path ? 'true' : undefined}
                          >
                            {reportLabel(item)}
                          </button>
                        </td>
                        <td className="col-md num">{periodLabel(item)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}
        </>
      ) : null}

      {/* ---------- 본문 ---------- */}
      {openPath ? (
        <section className="section">
          <div className="section__head">
            <h2 className="section__title">보고서 본문</h2>
            
          </div>
          {doc?.status === 'loading' ? <SkeletonText lines={6} /> : null}
          {doc?.status === 'error' ? (
            <ErrorState
              title="보고서 본문을 불러오지 못했습니다"
              body={doc.error.message}
              detail={doc.error.detail}
            />
          ) : null}
          {doc?.status === 'ready' ? (
            <>
              {/* 보고서 자체 스타일 — 셀렉터가 전부 .reportdoc 안으로 갇혀 있다 */}
              <style>{doc.data.css + DOC_FIX}</style>
              <div
                ref={docRef}
                className={SCOPE}
                /* 문서는 DOMParser 로 파싱하며 script·link 를 이미 제거했다 */
                dangerouslySetInnerHTML={{ __html: doc.data.html }}
              />
            </>
          ) : null}
        </section>
      ) : ready && ready.report_artifacts.length === 0 && !job ? (
        <EmptyState
          title="아직 만들어진 보고서가 없습니다"
          body="위에서 기간과 담을 절을 고른 뒤 보고서를 생성하세요. 생성이 끝나면 본문이 이 자리에 펼쳐집니다."
        />
      ) : null}
    </div>
  );
}

/* --- job 패널 ---------------------------------------------------------------- */

function JobPanel({ job, onOpen }: { job: ReportJob; onOpen: () => void }) {
  const running = job.state === 'running' || job.state === 'queued';
  const tone = job.state === 'failed' ? 'danger' : job.state === 'completed' ? 'ok' : 'info';
  return (
    <Card
      title={`생성 작업 ${job.job_id}`}
      aside={
        <Badge tone={tone}>
          {job.state === 'completed' ? '완료' : job.state === 'failed' ? '실패' : '진행 중'}
        </Badge>
      }
    >
      <dl className="dl">
        <div className="dl__item">
          <dt className="dl__k">단계</dt>
          <dd className="dl__v">{STAGE_LABEL[job.stage] ?? job.stage}</dd>
        </div>
        <div className="dl__item">
          <dt className="dl__k">엔진</dt>
          <dd className="dl__v">
            {job.engine === 'v2' ? '상세 분석 보고서 (v2)' : '기존 DASOL 엔진'}
            {job.use_llm === false ? ' · LLM 해설 없음' : ''}
          </dd>
        </div>
        <div className="dl__item">
          <dt className="dl__k">시작</dt>
          <dd className="dl__v num">{dateTime(job.started_at)}</dd>
        </div>
        {job.finished_at ? (
          <div className="dl__item">
            <dt className="dl__k">종료</dt>
            <dd className="dl__v num">{dateTime(job.finished_at)}</dd>
          </div>
        ) : null}
        {job.consistent !== undefined ? (
          <div className="dl__item">
            <dt className="dl__k">일관성 검증</dt>
            <dd className="dl__v">
              {job.consistent ? (
                <>
                  <CheckCircleIcon size={16} /> 모든 항등식 일치
                </>
              ) : (
                <>
                  <AlertCircleIcon size={16} /> 어긋난 항등식 있음 — 보고서 마지막 절 확인
                </>
              )}
            </dd>
          </div>
        ) : null}
      </dl>

      {job.error ? <Callout tone="warn">{job.error}</Callout> : null}

      {job.state === 'completed' ? (
        <Button variant="primary" onClick={onOpen}>
          이 보고서 열기
        </Button>
      ) : null}

      {running ? (
        <p className="card__note" role="status" aria-live="polite">
          서버가 보고서를 만들고 있습니다. 이 화면은 {POLL_MS / 1000}초마다 상태를 다시 물어봅니다 —
          진행률을 지어내지 않고 서버가 알려준 단계만 표시합니다.
        </p>
      ) : null}
    </Card>
  );
}
