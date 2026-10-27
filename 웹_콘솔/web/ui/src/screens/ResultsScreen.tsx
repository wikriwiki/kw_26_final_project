/**
 * 결과 — 스펙 §7.
 * 단일 목적: 정책 효과를 해석한다.
 *
 * 기본 노출  : 핵심 지표 4개 · 일별 결제 추이 · 업종별
 * 정책 실행  : 정책 효과(같은 시민·같은 날) · 누가 반응했나(집단별) · 실측 연구와 비교 — 정책 담당자가 볼 것만
 * 접어두는 것: 원자료 표 · 내보내기 · 출처
 *
 * **run 은 셸이 소유한다** (설계도 §5). 화면 안의 "실행 선택" 칩 그룹을 걷어냈고,
 * `useRun()` 이 준 실행을 그대로 그린다. 실행 이름·상태·기간은 사이드바가 항상 말한다.
 *
 * 예전 자리의 빈 "지도 미리보기" 상자는 아무것도 보여주지 않으면서 화면의 가장 큰 칸을
 * 차지했다. 그 자리에 정책 효과가 모양으로 보이는 일별 결제 추이를 둔다. 지도로 가는 길은
 * 화면 머리의 버튼이 맡고, 지금 보고 있는 실행과 일자를 그대로 들고 넘어간다.
 *
 * 결제 기록의 행정동 값이 3개 실행 전부에서 비어 있어(CONTRACT §2.8) 지역별 **결제 집계**는
 * 만들 수 없다. 그 사실은 업종 카드에 적어 두고, 지도는 결제 집계가 아니라 이동·방문을
 * 보는 것이라는 점을 미리보기 설명에 분명히 적는다.
 */
import { Link, useNavigate } from 'react-router-dom';
import { useRun } from '../app/RunContext';
import { Button } from '../components/Button';
import { Card, Stat } from '../components/Card';
import { RunEffectStat } from '../components/RunEffectStat';
import { DailyStack } from '../components/DailyStack';
import { Disclosure } from '../components/Disclosure';
import { Callout, EmptyState } from '../components/Feedback';
import { ErrorState, SkeletonCard } from '../components/Feedback';
import { MapIcon, UsersIcon } from '../components/Icon';
import { BarList } from '../components/Meter';
import type { BarItem } from '../components/Meter';
import { rememberScroll, useScrollMemory } from '../app/useViewState';
import { loadPolicyDetail, useLoad } from '../lib/runData';
import { int, krw, krwAxis, percent } from '../lib/format';
import { experimentsApi, useResource } from '../lib/experiments';
import { filtersQuery } from './experiment/ExperimentResults';
import { GroupEffectCard, PolicyEffect, ValidationCard } from './experiment/PolicyPanels';

/** 정책 실행 — 같은 시민을 정책 있음·없음으로 돌린 비교가 곧 결과다 */
function PolicyRunResults({ experimentId }: { experimentId: string }) {
  const run = useRun();
  const navigate = useNavigate();
  useScrollMemory('results');
  const detail = useResource(() => experimentsApi.detail(experimentId), `experiment:${experimentId}`);
  const comparison = useResource(() => experimentsApi.comparison(experimentId), `comparison:${experimentId}`);
  const experiment = detail.data;
  const effective = experiment?.window.effective_date;
  const marker = effective ? { x: effective, label: '정책 시행' } : undefined;

  function openMap() {
    rememberScroll('results');
    navigate(`${run.path('visualize')}${run.bundle.focusDay ? `?day=${run.bundle.focusDay}` : ''}`, { state: { from: 'results' } });
  }

  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">{experiment?.title ?? '결과'}</h1>
          {experiment?.summary ? <p className="pagehead__purpose">{experiment.summary}</p> : null}
        </div>
        <div className="pagehead__actions">
          <Button variant="secondary" icon={<MapIcon size={18} />} onClick={openMap}>
            3D 지도
          </Button>
          <Link className="btn btn--primary" to={run.path('agents')}>
            <UsersIcon size={18} />
            <span>시민 인터뷰</span>
          </Link>
        </div>
      </header>
      {comparison.error ? <ErrorState title="비교를 계산하지 못했습니다" body={comparison.error.message} /> : null}
      {!comparison.data ? (
        comparison.error ? null : <SkeletonCard rows={6} />
      ) : (
        <>
          <PolicyEffect comparison={comparison.data} marker={marker} />
          <GroupEffectCard comparison={comparison.data} askHref={(filters) => `${run.path('agents')}?${filtersQuery(filters)}`} />
          {experiment?.validation?.rows?.length ? <ValidationCard validation={experiment.validation} /> : null}
        </>
      )}
    </div>
  );
}


export function ResultsScreen() {
  const experimentId = useRun().index.meta?.experiment_id ?? null;
  return experimentId ? <PolicyRunResults experimentId={experimentId} /> : <RunResults />;
}

function RunResults() {
  const { index: indexItem, bundle, path, policy } = useRun();
  const navigate = useNavigate();
  useScrollMemory('results');

  const events = bundle.events;
  const totals = events.totals;
  const bound = policy.items[0]?.id ?? null;
  const boundDetail = useLoad(bound ? `policy:${bound}` : null, () => loadPolicyDetail(bound as string));
  const effectiveFrom = boundDetail.status === 'ready' ? (boundDetail.data.policy?.effective_from ?? null) : null;
  const daily = (events.by_day ?? []).map((d) => ({ day: d.day, total: d.amt, part: d.policy_paid }));

  const industryBars: BarItem[] = (events.by_l1 ?? [])
    .slice()
    .sort((a, b) => b.amt - a.amt)
    .slice(0, 8)
    .map((row) => ({
      key: row.l1,
      name: row.l1,
      value: row.amt,
      display: krw(row.amt),
    }));

  const hasWallet = (totals?.policy_paid ?? 0) > 0;

  const couponShare =
    totals && totals.events > 0 ? totals.coupon_eligible_events / totals.events : null;

  /** 보던 일자를 들고 시각화로. from 을 남겨야 "← 결과로" 가 진짜 뒤로 갈 수 있다 */
  function openMap() {
    rememberScroll('results');
    navigate(`${path('visualize')}${bundle.focusDay ? `?day=${bundle.focusDay}` : ''}`, { state: { from: 'results' } });
  }

  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">결과</h1>
          <p className="pagehead__purpose">
            정책이 소비를 어디서 얼마나 바꿨는지
          </p>
        </div>
        <div className="pagehead__actions">
          <Button variant="secondary" icon={<MapIcon size={18} />} onClick={openMap}>
            3D 지도에서 보기
          </Button>
        </div>
      </header>

      {!events.available ? (
        <Callout tone="warn">
          결제 내역이 없어 업종별 분석이 없습니다.
        </Callout>
      ) : null}

      {/* 1차 블록 1 — 핵심 지표 4개. 상자 없이 정렬로 묶는다 */}
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
          label="지원금 결제액"
          value={totals ? (totals.policy_paid > 0 ? krw(totals.policy_paid) : '해당 없음') : null}
          hint={
            totals && totals.amt > 0
              ? totals.policy_paid > 0
                ? `총 결제액의 ${percent(totals.policy_paid / totals.amt, 1)}`
                : '결제할 때 지원금에서 나가는 돈이 없는 정책'
              : undefined
          }
          unknownReason="결제 기록 없음"
        />
        <RunEffectStat
          className="c3"
          experimentId={indexItem.meta?.experiment_id ?? null}
          extraSpent={totals ? totals.extra_spent : null}
          extraKnown={!(events.null_only_fields ?? []).includes('ex')}
        />
        {totals && totals.policy_paid > 0 ? (
          <Stat
            className="c3"
            label="지원금 사용처 결제 비율"
            value={couponShare === null ? null : percent(couponShare, 1)}
            hint={`${int(totals.coupon_eligible_events)}건 / ${int(totals.events)}건`}
            unknownReason="결제 기록 없음"
          />
        ) : null}
      </div>

      {/* 1차 블록 2 — 지도 미리보기(시각화 주 진입) + 업종별 */}
      <div className="grid">
        <Card className="c7" title="일별 결제 추이">
          {daily.length > 0 ? (
            <>
              <DailyStack
                points={daily}
                format={krw}
                tickFormat={krwAxis}
                totalLabel="총 결제"
                partLabel="지원금"
                marker={effectiveFrom ? { day: effectiveFrom, label: '정책 시행' } : null}
                title="일별 총결제액과 그중 지원금 결제"
              />
            </>
          ) : (
            <EmptyState
              title="일별 결제 기록이 없습니다"
              body={events.reason ?? '결제 기록 파일이 없습니다.'}
            />
          )}
        </Card>

        <Card className="c5" title="업종별 결제액">
          {industryBars.length > 0 ? (
            <>
              <BarList items={industryBars} />
              <p className="card__note">
                결제액 상위 8개 업종
              </p>
            </>
          ) : (
            <EmptyState
              title="업종별 집계가 없습니다"
              body={events.reason ?? '결제 기록 파일이 없습니다.'}
            />
          )}
        </Card>
      </div>

      {/* 접어두는 것 — 기본 닫힘 */}
      <section className="section">
        <h2 className="section__title">더 보기</h2>
        <div>
          <Disclosure
            title="일자별 결제"
            meta={events.by_day && events.by_day.length > 0 ? `${events.by_day.length}일` : '기록 없음'}
          >
            {events.by_day && events.by_day.length > 0 ? (
              <div className="table-wrap">
                <table className="table table--lead">
                  <caption className="visually-hidden">일자별 결제 집계</caption>
                  <thead>
                    <tr>
                      <th scope="col">일자</th>
                      <th scope="col" className="n">
                        건수
                      </th>
                      <th scope="col" className="n col-md">
                        결제액
                      </th>
                      {hasWallet ? (
                        <th scope="col" className="n col-lg">
                          지원금
                        </th>
                      ) : null}
                    </tr>
                  </thead>
                  <tbody>
                    {events.by_day.map((row) => (
                      <tr key={row.day}>
                        <th scope="row" className="num" style={{ fontWeight: 400 }}>
                          {row.day}
                        </th>
                        <td className="n">
                          {int(row.events)}
                          <span className="cell-sub cell-sub--md">{krw(row.amt)}</span>
                        </td>
                        <td className="n col-md">{krw(row.amt)}</td>
                        {hasWallet ? <td className="n col-lg">{krw(row.policy_paid)}</td> : null}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState
                title="일자별 집계가 없습니다"
                body={events.reason ?? '결제 기록 파일이 없습니다.'}
              />
            )}
          </Disclosure>

        </div>
      </section>
    </div>
  );
}
