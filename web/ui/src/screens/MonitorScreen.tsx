/**
 * 실행 모니터 — 스펙 §7.
 * 단일 목적: 지금 잘 돌고 있는지 본다.
 *
 * 기본 노출  : 진행률 · 핵심 지표 4개 · 일자별 추이   (1차 블록 3개)
 * 접어두는 것: 단계별 소요 분해 · 느린 사례 · 응답 오류 원문 · 실행 계획 근거
 *
 * **run 은 셸이 소유한다** (설계도 §5). 화면 안의 run 선택기를 걷어냈고, `useRun()` 이
 * 준 실행을 그대로 그린다. 어떤 실행을 보는 중인지는 사이드바가 항상 말한다.
 *
 * 모르는 값을 0 으로 그리지 않는다. 목표 인원을 모르면 진행 바 자체를 그리지 않는다
 * (CONTRACT §3.3 — progress_ratio 가 null 이면 0%도 100%도 아니다).
 *
 * §7b 적용: 지표·진행률·표에서 카드 상자를 걷어냈다. 뱃지는 "중단됨" 하나만 남기고
 * 나머지는 글자로 말한다. 막대 색은 한 계열만 쓴다.
 */
import { useMemo, useState } from 'react';
import { useRun } from '../app/RunContext';
import type { RunContextValue } from '../app/RunContext';
import { Disclosure } from '../components/Disclosure';
import { Button } from '../components/Button';
import { Stat } from '../components/Card';
import { Callout, EmptyState } from '../components/Feedback';
import { AlertTriangleIcon } from '../components/Icon';
import { BarList, Meter } from '../components/Meter';
import type { BarItem } from '../components/Meter';
import { dec, duration, int } from '../lib/format';
import { errorType, timingPath } from '../lib/labels';

const DEFAULT_ROWS = 8;

export function MonitorScreen() {
  const run = useRun();
  // key 를 걸어 두면 실행을 바꿀 때 "전체 일자 보기" 같은 화면 안 상태가 따라오지 않는다
  return <MonitorView key={run.id} run={run} />;
}

function MonitorView({ run }: { run: RunContextValue }) {
  const [showAllDays, setShowAllDays] = useState(false);

  const { index: indexItem, bundle } = run;
  const days = bundle.days.items;
  const plan = bundle.detail.plan;

  const totals = useMemo(() => {
    let ok = 0;
    let err = 0;
    let elapsed = 0;
    let elapsedKnown = false;
    for (const d of days) {
      ok += d.agents_ok;
      err += d.agents_error;
      if (d.elapsed_sec !== null) {
        elapsed += d.elapsed_sec;
        elapsedKnown = true;
      }
    }
    return { ok, err, elapsed, elapsedKnown };
  }, [days]);

  const plannedDays = plan.planned_days;
  const dayProgress = plannedDays === null ? null : Math.min(1, days.length / plannedDays);

  const visibleDays = showAllDays ? days : days.slice(0, DEFAULT_ROWS);

  const bottleneckRows = bundle.bottlenecks.available
    ? (bundle.bottlenecks.bottleneck_rank ?? [])
    : (bundle.bottlenecks.fallback_rank ?? []);

  const bottleneckBars: BarItem[] = bottleneckRows.slice(0, 8).map((row, i) => {
    const path = String(row.path);
    const total = Number(row.total_sec);
    const named = timingPath(path);
    return {
      key: `${path}-${i}`,
      name: named.detail ? `${named.label} · ${named.detail}` : named.label,
      value: total,
      display: duration(total),
    };
  });

  const errorTypeBars: BarItem[] = Object.entries(bundle.failures.by_error_type ?? {}).map(
    ([type, n]) => {
      const t = errorType(type);
      return {
        key: type,
        name: t.known ? t.label : type,
        value: n,
        display: `${int(n)}건`,
      };
    },
  );

  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">실행 모니터</h1>
          <p className="pagehead__purpose">
            시뮬레이션이 지금 어디까지 왔고, 문제가 생긴 곳이 있는지 확인합니다.
          </p>
        </div>
      </header>

      {/* 1차 블록 1 — 진행률. 숫자 자체가 제목이라 카드도 제목도 두지 않는다 (§7b) */}
      <section className="section">
        {dayProgress === null ? (
          <>
            <Stat
              label="완료한 일자"
              value={null}
              unknownReason={`${days.length}일치가 기록돼 있지만, 계획 일수를 알 수 없어 진행률을 계산할 수 없습니다.`}
            />
            <Callout tone="warn">
              이 실행에는 요약 파일이 없습니다. 실행이 끝나기 전에 중단됐다는 뜻이며, 실패한 것은
              아닙니다. 아래 “실행 계획 근거”에서 로그에 남은 값을 확인할 수 있습니다.
            </Callout>
          </>
        ) : (
          <>
            <div className="progress__head">
              <span className="progress__value">{dec(dayProgress * 100, 0)}%</span>
              <span className="section__note">
                {int(days.length)} / {int(plannedDays)}일 완료
              </span>
            </div>
            <Meter ratio={dayProgress} label="일자 진행률" className="progress__track" />
            <p className="section__note">
              시작 {plan.start_day} · 동시 처리 {int(plan.workers)}개 ·{' '}
              {indexItem.completed_at ? `${indexItem.completed_at.slice(0, 10)} 종료` : '진행 중'}
            </p>
          </>
        )}
      </section>

      {/* 1차 블록 2 — 핵심 지표 4개 */}
      <div className="grid statrow">
        <Stat
          className="c3"
          label="처리한 대상자"
          value={int(totals.ok)}
          unit="명"
          hint={`${int(days.length)}일 합계`}
        />
        <Stat
          className="c3"
          label="처리 실패한 대상자"
          value={int(totals.err)}
          unit="명"
          hint={totals.err === 0 ? '실패 없음' : '해당 일자를 상세에서 확인하세요'}
        />
        <Stat
          className="c3"
          label="응답 오류"
          value={bundle.failures.total === null ? null : int(bundle.failures.total)}
          unit="건"
          hint="재시도로 복구된 건을 포함합니다"
          unknownReason="오류 기록 파일이 없습니다"
        />
        <Stat
          className="c3"
          label="총 소요 시간"
          value={totals.elapsedKnown ? duration(totals.elapsed) : null}
          hint={totals.elapsedKnown ? `일 평균 ${duration(totals.elapsed / days.length)}` : undefined}
          unknownReason="일자별 소요 기록이 아직 없습니다"
        />
      </div>

      {/* 1차 블록 3 — 일자별 추이 */}
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">일자별 추이</h2>
          <p className="section__note">
            하루가 끝나야 소요 시간이 기록됩니다. 아직 끝나지 않은 날은 “진행 중”으로 표시됩니다.
          </p>
        </div>
        <div className="table-wrap">
          <table className="table table--lead">
            <caption className="visually-hidden">일자별 처리 인원과 소요 시간</caption>
            <thead>
              <tr>
                <th scope="col">일자</th>
                <th scope="col" className="n">
                  처리 인원
                </th>
                <th scope="col" className="n col-md">
                  실패
                </th>
                <th scope="col" className="n col-lg">
                  소요 시간
                </th>
              </tr>
            </thead>
            <tbody>
              {visibleDays.map((d) => (
                <tr key={d.day}>
                  <th scope="row" style={{ fontWeight: 400 }}>
                    <span className="num">{d.day}</span>
                    {!d.day_complete ? <span className="cell-sub">진행 중</span> : null}
                    <span className="cell-sub cell-sub--lg">
                      소요 {d.elapsed_sec === null ? '기록 없음' : duration(d.elapsed_sec)}
                    </span>
                  </th>
                  <td className="n">
                    {int(d.agents_ok)}
                    {d.progress_ratio !== null ? (
                      <span className="cell-sub">목표 대비 {dec(d.progress_ratio * 100, 0)}%</span>
                    ) : (
                      <span className="cell-sub">목표 인원 알 수 없음</span>
                    )}
                    <span className="cell-sub cell-sub--md">
                      실패 {d.agents_error === 0 ? '없음' : `${int(d.agents_error)}명`}
                    </span>
                  </td>
                  <td className="n col-md">
                    {d.agents_error === 0 ? '—' : int(d.agents_error)}
                    {d.checkpoint_failed_count === null ? (
                      <span className="cell-sub">기록 미완료</span>
                    ) : null}
                  </td>
                  <td className="n col-lg">{d.elapsed_sec === null ? '—' : duration(d.elapsed_sec)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {days.length > DEFAULT_ROWS ? (
          <div>
            <Button variant="ghost" onClick={() => setShowAllDays((v) => !v)}>
              {showAllDays ? '최근 8일만 보기' : `전체 ${days.length}일 보기`}
            </Button>
          </div>
        ) : null}
      </section>

      {/* 접어두는 것 — 기본 닫힘 */}
      <section className="section">
        <h2 className="section__title">상세 보기</h2>
        <div>
          <Disclosure title="단계별 소요 분해" meta={bundle.focusDay}>
            {bottleneckBars.length === 0 ? (
              <EmptyState
                title="단계별 기록이 없습니다"
                body={bundle.bottlenecks.reason ?? '해당 일자의 소요 기록 파일을 찾지 못했습니다.'}
              />
            ) : (
              <>
                {bundle.bottlenecks.degraded ? (
                  <Callout tone="warn">
                    {bundle.bottlenecks.degraded_note ??
                      '일부 단계만 다시 계산했습니다. 하위 단계는 확인할 수 없습니다.'}
                  </Callout>
                ) : null}
                <BarList items={bottleneckBars} />
                <p className="card__note">
                  {bundle.focusDay} 하루 동안 각 단계가 차지한 총 시간입니다.
                </p>
              </>
            )}
          </Disclosure>

          <Disclosure
            title="오래 걸린 사례"
            meta={bundle.slow.total === null ? '기록 없음' : `${int(bundle.slow.total)}건`}
          >
            {!bundle.slow.available ? (
              <EmptyState
                title="느린 사례 기록이 없습니다"
                body={`하루가 끝나야 만들어지는 파일입니다. 이 실행은 그 전에 멈췄습니다. (${
                  bundle.slow.reason ?? '기록 파일 없음'
                })`}
              />
            ) : (
              <>
                <p className="card__note">
                  {Object.entries(bundle.slow.phase_counts ?? {})
                    .map(([phase, n]) => {
                      const named = timingPath(
                        `phase.t_${phase === 'dawn' ? 'dawn' : phase === 'stage1' ? 's1' : 's2'}`,
                      );
                      return `${named.label} ${int(n)}건`;
                    })
                    .join(' · ')}
                </p>
                <div className="table-wrap">
                  <table className="table table--lead">
                    <caption className="visually-hidden">오래 걸린 대상자 상위 목록</caption>
                    <thead>
                      <tr>
                        <th scope="col">대상자</th>
                        <th scope="col" className="n">
                          가장 오래 걸린 단계
                        </th>
                        <th scope="col" className="n col-md">
                          소요
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {bundle.slow.items.slice(0, 8).map((it) => {
                        const entries = Object.entries(it.slow);
                        const worst = entries.reduce((a, b) => (b[1] > a[1] ? b : a), entries[0]!);
                        const named = timingPath(
                          `phase.t_${worst[0] === 'dawn' ? 'dawn' : worst[0] === 'stage1' ? 's1' : 's2'}`,
                        );
                        return (
                          <tr key={it.aid}>
                            <td className="wrap num">{it.aid}</td>
                            <td className="n">
                              {named.label}
                              <span className="cell-sub cell-sub--md">{duration(worst[1])}</span>
                            </td>
                            <td className="n col-md">{duration(worst[1])}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="card__note">
                  기준이 되는 임계값은 기록 파일에 남지 않아 알 수 없습니다. 순위만 참고하세요.
                </p>
              </>
            )}
          </Disclosure>

          <Disclosure
            title="응답 오류 원문"
            meta={bundle.failures.total === null ? '기록 없음' : `${int(bundle.failures.total)}건`}
          >
            <p className="card__note">
              대상자가 실패한 건수가 아닙니다. 다시 시도해서 복구된 건도 여기에 남습니다.
            </p>
            {errorTypeBars.length > 0 ? <BarList items={errorTypeBars} /> : null}
            <ul className="checks">
              {bundle.failures.items.slice(0, 3).map((f, i) => (
                <li className="check check--warn" key={`${f.aid}-${f.attempt}-${i}`}>
                  <AlertTriangleIcon size={16} />
                  <div className="stack-sm">
                    <span className="wrap">
                      {errorType(f.error_type).label} · {f.day} · {f.attempt + 1}번째 시도
                    </span>
                    <pre className="code">{f.error}</pre>
                  </div>
                </li>
              ))}
            </ul>
          </Disclosure>

          <Disclosure title="실행 계획 근거">
            <dl className="dl">
              <div className="dl__item">
                <dt className="dl__k">정본 (요약 파일)</dt>
                <dd className="dl__v">
                  {plan.source === null
                    ? '없음 — 실행이 끝나지 않아 요약 파일이 만들어지지 않았습니다'
                    : `${plan.source} · ${int(plan.planned_days)}일 · 하루 ${int(plan.agents_target)}명`}
                </dd>
              </div>
              <div className="dl__item">
                <dt className="dl__k">참고 (실행 로그)</dt>
                <dd className="dl__v wrap">
                  {bundle.detail.log_hint === null
                    ? '없음'
                    : `${String(bundle.detail.log_hint.source_file)} · ${int(
                        Number(bundle.detail.log_hint.planned_days),
                      )}일 · 하루 ${int(Number(bundle.detail.log_hint.agents_target))}명`}
                </dd>
              </div>
            </dl>
            <Callout>
              로그는 참고용입니다. 재시작으로 덮여쓰였을 수 있어, 요약 파일이 있으면 항상 그쪽이
              맞습니다.
            </Callout>
          </Disclosure>
        </div>
      </section>
    </div>
  );
}
