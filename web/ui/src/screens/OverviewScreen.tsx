/**
 * 실행 개요 — `/runs/:runId` (설계도 §2, §4).
 * 단일 목적: 이 실행이 무엇인지 한 장으로 파악한다. 상세는 각 기능 화면으로 보낸다.
 *
 * 1차 블록 4개: 실행 조건 · 핵심 지표 · 진행 상황 · 다음 행동
 *
 * run 은 **셸이 소유한다.** 이 화면에는 실행을 고르는 장치가 없고, `useRun()` 이
 * 이미 정해진 실행을 준다 (설계도 §5).
 *
 * 중단된 실행(BASE7500 같은)은 **있는 것만** 보여준다. 모르는 값에 0 을 넣지 않고
 * "알 수 없음"과 그 이유를 함께 적는다 (CONTRACT §3.3, §4.1-5 — "중단"이지 "실패"가 아니다).
 */
import { Link } from 'react-router-dom';
import { PHASE_LABEL, useRun } from '../app/RunContext';
import type { FeatureKey, RunContextValue } from '../app/RunContext';
import { Stat } from '../components/Card';
import { Callout } from '../components/Feedback';
import { ActivityIcon, BarChartIcon, MapIcon } from '../components/Icon';
import { Meter } from '../components/Meter';
import { dateTime, dec, int, krw, percent } from '../lib/format';

/**
 * 이 실행에 적용된 정책 한 줄.
 *
 * 실행 산출물에는 정책 id 필드가 따로 없다. 결제 기록에 남은 정책 지갑 사용분이
 * 유일한 근거라서(`RunContext.readPolicy`), 기록 자체가 없으면 **"무정책"이 아니라
 * "알 수 없음"** 이다. 없는 것과 확인할 수 없는 것을 같게 적지 않는다.
 */
function policyLine(run: RunContextValue): { value: string; note: string } {
  if (!run.policy.known) {
    return {
      value: '알 수 없음',
      note: `결제 기록이 없어 확인할 수 없습니다. ${run.bundle.events.reason ?? '기록 파일 없음'}`,
    };
  }
  if (run.policy.items.length === 0) {
    return {
      value: '무정책(대조군)',
      note: '결제 기록에 정책 지갑에서 나간 금액이 없습니다.',
    };
  }
  return {
    value: run.policy.items.map((p) => (p.name ? `${p.name} (${p.id})` : p.id)).join(' · '),
    note: '결제 기록에 남은 정책 지갑 사용분에서 확인한 값입니다.',
  };
}

/** 다음 행동. 표시 순서는 고정이고, 못 쓰는 것은 숨기지 않고 비활성 + 이유를 적는다 (설계도 §4) */
const ACTIONS: Array<{ key: FeatureKey; label: string; Icon: typeof MapIcon }> = [
  { key: 'results', label: '결과 보기', Icon: BarChartIcon },
  { key: 'visualize', label: '지도에서 보기', Icon: MapIcon },
  { key: 'monitor', label: '모니터 보기', Icon: ActivityIcon },
];

/** primary 는 화면에 하나뿐이다 (§8). 쓸 수 있는 것 중 앞선 것 하나만 고른다 */
const PRIMARY_ORDER: FeatureKey[] = ['results', 'monitor', 'visualize'];

export function OverviewScreen() {
  const run = useRun();
  const { index, bundle } = run;
  const days = bundle.days.items;
  const totals = bundle.events.totals;
  const policy = policyLine(run);

  const completed = run.phase === 'completed';
  const planned = run.daysPlanned;
  const present = run.daysPresent;
  const dayProgress = planned === null || planned <= 0 ? null : Math.min(1, present / planned);
  const lastDay = days.length > 0 ? days[days.length - 1] : null;
  const processed = days.reduce((sum, d) => sum + d.agents_ok, 0);
  const period = run.firstDay && run.lastDay ? `${run.firstDay} ~ ${run.lastDay}` : null;

  const primaryKey = PRIMARY_ORDER.find((key) => run.can(key)) ?? null;
  const blocked = run.features.filter((f) => ACTIONS.some((a) => a.key === f.key) && !f.available);

  /** 보던 일자를 그대로 들고 지도로 넘어간다 (스펙 §9 state-preservation) */
  function actionTo(key: FeatureKey): string {
    const base = run.path(key === 'results' ? 'results' : key === 'monitor' ? 'monitor' : 'visualize');
    return key === 'visualize' ? `${base}?day=${bundle.focusDay}` : base;
  }

  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">실행 {run.id}</h1>
          <p className="pagehead__purpose">
            이 실행이 어떤 조건으로 돌았고 지금 어디까지 왔는지 한 장으로 봅니다.
          </p>
        </div>
      </header>

      {/* 중단된 실행을 숨기지 않는다. 무엇이 없는지 먼저 말한다 (설계도 §3) */}
      {run.phase === 'stopped' ? (
        <Callout tone="warn">
          이 실행은 끝나기 전에 멈췄습니다. 실패한 것이 아니라 남은 일자가 만들어지지 않은
          것이며, 아래에는 실제로 기록이 남아 있는 값만 표시합니다.
        </Callout>
      ) : null}

      {/* 1차 블록 1 — 실행 조건 */}
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">실행 조건</h2>
        </div>
        <dl className="dl">
          <div className="dl__item">
            <dt className="dl__k">상태</dt>
            <dd className="dl__v">{PHASE_LABEL[run.phase]}</dd>
          </div>

          <div className="dl__item">
            <dt className="dl__k">기간</dt>
            <dd className="dl__v">
              {period === null ? (
                <>
                  알 수 없음
                  <span className="cell-sub">기록이 남은 일자가 없습니다.</span>
                </>
              ) : (
                <>
                  <span className="num">{period}</span>
                  <span className="cell-sub">
                    {planned === null
                      ? `${int(present)}일 기록됨 · 계획 일수 알 수 없음`
                      : `${int(present)}일 기록됨 / 계획 ${int(planned)}일`}
                  </span>
                </>
              )}
            </dd>
          </div>

          <div className="dl__item">
            <dt className="dl__k">하루 대상자</dt>
            <dd className="dl__v">
              {run.agentsTarget === null ? (
                <>
                  알 수 없음
                  <span className="cell-sub">
                    요약 파일이 없어 목표 인원을 확인할 수 없습니다. 실제로 처리된 인원은 아래
                    진행 상황에 있습니다.
                  </span>
                </>
              ) : (
                <>
                  <span className="num">{int(run.agentsTarget)}</span>명
                </>
              )}
            </dd>
          </div>

          <div className="dl__item">
            <dt className="dl__k">적용 정책</dt>
            <dd className="dl__v wrap">
              {policy.value}
              <span className="cell-sub">{policy.note}</span>
            </dd>
          </div>
        </dl>
      </section>

      {/* 1차 블록 2 — 핵심 지표. 결과 화면의 상위 지표를 그대로 쓴다 (같은 숫자를 다르게 찍지 않는다) */}
      <div className="grid statrow">
        <Stat
          className="c3"
          label="총 결제액"
          value={totals ? krw(totals.amt) : null}
          hint={totals ? `${int(totals.events)}건` : undefined}
          unknownReason="결제 기록 없음"
        />
        <Stat
          className="c3"
          label="정책 지갑 결제액"
          value={totals ? krw(totals.policy_paid) : null}
          hint={
            totals && totals.amt > 0
              ? `총 결제액의 ${percent(totals.policy_paid / totals.amt, 1)}`
              : undefined
          }
          unknownReason="결제 기록 없음"
        />
        <Stat
          className="c3"
          label="정책이 만든 추가 소비"
          value={totals ? krw(totals.extra_spent) : null}
          hint="값이 비어 있는 건이 많아 하한값입니다"
          unknownReason="결제 기록 없음"
        />
        <Stat
          className="c3"
          label="처리한 대상자"
          value={days.length > 0 ? int(processed) : null}
          unit="명"
          hint={days.length > 0 ? `${int(days.length)}일 합계` : undefined}
          unknownReason="기록이 남은 일자가 없습니다"
        />
      </div>

      {/* 1차 블록 3 — 진행 상황 */}
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">진행 상황</h2>
        </div>

        {completed ? (
          <p className="section__note">
            계획한 {int(planned ?? present)}일이 모두 기록됐습니다.
            {index.completed_at ? ` ${dateTime(index.completed_at)} 종료.` : ''}
          </p>
        ) : dayProgress !== null ? (
          <>
            <div className="progress__head">
              <span className="progress__value">{dec(dayProgress * 100, 0)}%</span>
              <span className="section__note">
                {int(present)} / {int(planned)}일 기록됨
              </span>
            </div>
            <Meter ratio={dayProgress} label="일자 진행률" className="progress__track" />
          </>
        ) : (
          /* 분모를 모르면 막대를 그리지 않는다 (CONTRACT §3.3) */
          <Stat
            label="진행률"
            value={null}
            unknownReason="계획 일수가 기록에 없어 진행률을 계산할 수 없습니다."
          />
        )}

        {!completed && lastDay ? (
          <p className="section__note">
            남아 있는 기록은 <span className="num">{lastDay.day}</span> 까지이고, 그날은{' '}
            {lastDay.day_complete ? '마무리됐습니다' : '아직 마무리되지 않았습니다'}. 그때까지{' '}
            {int(processed)}명이 처리됐습니다.
          </p>
        ) : null}
      </section>

      {/* 1차 블록 4 — 다음 행동 */}
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">다음 행동</h2>
          <p className="section__note">이 실행을 어느 방향으로 더 볼지 고르세요.</p>
        </div>
        <div className="row">
          {ACTIONS.map(({ key, label, Icon }) => {
            const usable = run.can(key);
            const variant = key === primaryKey ? 'btn--primary' : 'btn--secondary';
            if (!usable) {
              const why = run.features.find((f) => f.key === key)?.reason;
              return (
                <button
                  key={key}
                  type="button"
                  className="btn btn--secondary"
                  disabled
                  /* 왜 못 쓰는지가 이름에 함께 읽혀야 한다 — 아래 안내 줄은 눈으로 보는 쪽 몫이다 */
                  aria-label={why ? `${label} — ${why}` : label}
                >
                  <Icon size={18} />
                  <span>{label}</span>
                </button>
              );
            }
            return (
              <Link key={key} className={`btn ${variant}`} to={actionTo(key)}>
                <Icon size={18} />
                <span>{label}</span>
              </Link>
            );
          })}
        </div>
        {/* 못 쓰는 것은 숨기지 않고 이유를 적는다 (설계도 §4 empty-nav-state) */}
        {blocked.length > 0 ? (
          <p className="section__note">
            {blocked.map((f) => `${f.label}: ${f.reason ?? '사용할 수 없음'}`).join(' · ')}
          </p>
        ) : null}
      </section>
    </div>
  );
}
