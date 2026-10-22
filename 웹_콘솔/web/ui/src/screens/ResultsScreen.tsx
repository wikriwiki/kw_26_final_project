/**
 * 결과 — 스펙 §7.
 * 단일 목적: 정책 효과를 해석한다.
 *
 * 기본 노출  : 핵심 지표 4개 · 일별 결제 추이 · 업종별 · 소비 분위별 비교
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
import { useNavigate } from 'react-router-dom';
import { useRun } from '../app/RunContext';
import { Button } from '../components/Button';
import { Card, Stat } from '../components/Card';
import { RunEffectStat } from '../components/RunEffectStat';
import { DailyStack } from '../components/DailyStack';
import { Disclosure } from '../components/Disclosure';
import { Callout, EmptyState } from '../components/Feedback';
import { MapIcon } from '../components/Icon';
import { BarList } from '../components/Meter';
import type { BarItem } from '../components/Meter';
import { rememberScroll, useScrollMemory } from '../app/useViewState';
import { loadPolicyDetail, useLoad, type DecileBucket } from '../lib/runData';
import { int, krw, krwAxis, percent } from '../lib/format';
import { decile } from '../lib/labels';


export function ResultsScreen() {
  const { index: indexItem, bundle, path, policy } = useRun();
  const navigate = useNavigate();
  useScrollMemory('results');

  const events = bundle.events;
  const totals = events.totals;
  /**
   * 분위별 **1인당 지급액**은 정책이 정한 값이다 — 실행 집계가 아니다.
   * 쿠폰은 시행 첫날 한 번 지급되므로 그날치 집계는 이후 모든 날에 0 이고,
   * 그 0 을 '지급액'이라 적으면 정책이 아무것도 주지 않은 것처럼 읽힌다.
   */
  const bound = policy.items[0]?.id ?? null;
  const boundDetail = useLoad(bound ? `policy:${bound}` : null, () => loadPolicyDetail(bound as string));
  const grants = (
    boundDetail.status === 'ready' ? (boundDetail.data.policy?.decile_grants ?? {}) : {}
  ) as Record<string, number>;
  const effectiveFrom = boundDetail.status === 'ready' ? (boundDetail.data.policy?.effective_from ?? null) : null;
  const daily = (events.by_day ?? []).map((d) => ({ day: d.day, total: d.amt, part: d.policy_paid }));

  // 분위가 없는 대상자는 보여주지 않는다. 읽는 사람이 판단에 쓸 수 없는 줄이다
  const deciles = ((bundle.dayAggregate?.by_spend_decile ?? []) as unknown as DecileBucket[]).filter(
    (row) => row.spend_decile !== null,
  );

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

  // 지원금이 없는 정책(거리두기·캐시백)은 지급액 칸을 두지 않는다
  const hasGrant = Object.values(grants).some((v) => v > 0);
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

      {/* 1차 블록 3 — 소비 분위별 비교 */}
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">소비 분위별 비교</h2>
          <p className="section__note">
            {bundle.focusDay} 하루
          </p>
        </div>

        <div className="table-wrap">
          <table className="table table--lead">
            <caption className="visually-hidden">소비 분위별 1인당 지급액과 소비액</caption>
            <thead>
              <tr>
                <th scope="col">소비 분위</th>
                <th scope="col" className="n">
                  시민
                </th>
                {hasGrant ? (
                  <th scope="col" className="n col-md">
                    1인당 지급액
                  </th>
                ) : null}
                <th scope="col" className="n col-lg">
                  당일 소비액
                </th>
              </tr>
            </thead>
            <tbody>
              {deciles.map((row) => (
                <tr key={String(row.spend_decile)}>
                  <th scope="row">
                    {decile(row.spend_decile)}
                  </th>
                  <td className="n">
                    {int(row.agents)}명
                    {hasGrant ? (
                      <span className="cell-sub cell-sub--md">
                        지급 {krw(grants[String(row.spend_decile)] ?? 0)}
                      </span>
                    ) : null}
                    <span className="cell-sub cell-sub--lg">
                      소비 {krw(row.cm_today_total_incl_online)}
                    </span>
                  </td>
                  {hasGrant ? <td className="n col-md">{krw(grants[String(row.spend_decile)] ?? 0)}</td> : null}
                  <td className="n col-lg">{krw(row.cm_today_total_incl_online)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="section__note">
          시민 {int(deciles.reduce((s, r) => s + r.agents, 0))}명
        </p>
      </section>

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
