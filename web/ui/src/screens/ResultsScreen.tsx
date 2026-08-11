/**
 * 결과 — 스펙 §7.
 * 단일 목적: 정책 효과를 해석한다.
 *
 * 기본 노출  : 핵심 지표 4개 · 지도 미리보기 · 업종별 · 소비 분위별 비교
 * 접어두는 것: 원자료 표 · 내보내기 · 출처
 *
 * **run 은 셸이 소유한다** (설계도 §5). 화면 안의 "실행 선택" 칩 그룹을 걷어냈고,
 * `useRun()` 이 준 실행을 그대로 그린다. 실행 이름·상태·기간은 사이드바가 항상 말한다.
 *
 * 지도 미리보기가 시각화 화면의 **주 진입 경로**다 (§7 "시각화 페이지 진입 흐름").
 * 미리보기 자체가 버튼이고, 옆에 "지도에서 열기"도 함께 둔다 — 정확히 조준하지 않아도
 * 갈 수 있어야 한다. 지금 보고 있는 실행과 일자를 그대로 들고 넘어간다.
 *
 * 결제 기록의 행정동 값이 3개 실행 전부에서 비어 있어(CONTRACT §2.8) 지역별 **결제 집계**는
 * 만들 수 없다. 그 사실은 업종 카드에 적어 두고, 지도는 결제 집계가 아니라 이동·방문을
 * 보는 것이라는 점을 미리보기 설명에 분명히 적는다.
 */
import { useNavigate } from 'react-router-dom';
import { useRun } from '../app/RunContext';
import { Button } from '../components/Button';
import { Card, Stat } from '../components/Card';
import { Disclosure } from '../components/Disclosure';
import { Callout, EmptyState } from '../components/Feedback';
import { DownloadIcon, MapIcon } from '../components/Icon';
import { BarList } from '../components/Meter';
import type { BarItem } from '../components/Meter';
import { rememberScroll, useScrollMemory } from '../app/useViewState';
import type { DecileBucket } from '../lib/fixtures';
import { dec, int, krw, percent } from '../lib/format';
import { decile } from '../lib/labels';
import { policyDetails } from '../lib/fixtures';

const ARTIFACT_LABEL: Record<string, string> = {
  summary_json: '실행 요약',
  events_jsonl: '결제 기록',
  poi_summary_json: '매장 집계',
  stage1_failures_jsonl: '응답 오류 기록',
  timing_dir: '단계별 소요',
  checkpoints_dir: '진행 체크포인트',
  metrics_dir: '대상자별 기록',
};

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
  const grants = (() => {
    const bound = policy.items[0]?.id;
    const detail = bound ? policyDetails[bound] : undefined;
    return (detail?.policy?.decile_grants ?? {}) as Record<string, number>;
  })();

  // 분위가 없는 대상자는 보여주지 않는다. 읽는 사람이 판단에 쓸 수 없는 줄이다
  const deciles = (bundle.dayAggregate.by_spend_decile as unknown as DecileBucket[]).filter(
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

  const couponShare =
    totals && totals.events > 0 ? totals.coupon_eligible_events / totals.events : null;

  /** 보던 일자를 들고 시각화로. from 을 남겨야 "← 결과로" 가 진짜 뒤로 갈 수 있다 */
  function openMap() {
    rememberScroll('results');
    navigate(`${path('visualize')}?day=${bundle.focusDay}`, { state: { from: 'results' } });
  }

  return (
    <div className="stack">
      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">결과</h1>
          <p className="pagehead__purpose">
            정책이 실제 소비를 어디서 얼마나 바꿨는지 확인합니다.
          </p>
        </div>
      </header>

      {!events.available ? (
        <Callout tone="warn">
          이 실행에는 결제 내역이 없어 업종별 분석을 볼 수 없습니다. 아래 소비 분위별 비교는
          그대로 보실 수 있습니다.
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
          label="쿠폰 사용처 결제 비율"
          value={couponShare === null ? null : percent(couponShare, 1)}
          hint={
            totals ? `${int(totals.coupon_eligible_events)}건 / ${int(totals.events)}건` : undefined
          }
          unknownReason="결제 기록 없음"
        />
      </div>

      {/* 1차 블록 2 — 지도 미리보기(시각화 주 진입) + 업종별 */}
      <div className="grid">
        <Card className="c6">
          <button type="button" className="mappreview" onClick={openMap}>
            {/*
              실제 썸네일이 생기면 이 자리에 <img> 가 들어간다.
              없는 지도를 그린 척하지 않으려고 지금은 무엇이 열리는지만 적는다.
            */}
            <span className="mappreview__surface">
              <span className="mappreview__label">3D 지도 미리보기</span>
              <span className="mappreview__meta num">{bundle.focusDay}</span>
            </span>
          </button>
          <p className="card__note">
            대상자가 하루 동안 어디로 움직이고 어디에 머물렀는지 지도 위에서 봅니다. 결제 금액이
            아니라 이동과 방문을 보는 화면입니다.
          </p>
          <div className="row">
            <Button variant="primary" icon={<MapIcon size={18} />} onClick={openMap}>
              지도에서 열기
            </Button>
          </div>
        </Card>

        <Card className="c6" title="업종별 결제액">
          {industryBars.length > 0 ? (
            <>
              <BarList items={industryBars} />
              <p className="card__note">
                결제액 상위 8개 업종입니다. 지역(행정동) 단위 집계는 결제 기록의 행정동 값이 전부
                비어 있어 아직 만들 수 없습니다 — 매장 정보에 행정동 코드가 채워지면 이 카드에
                지역별 결제액이 함께 표시됩니다.
              </p>
            </>
          ) : (
            <EmptyState
              title="업종별 집계가 없습니다"
              body={events.reason ?? '이 실행에는 결제 기록 파일이 만들어지지 않았습니다.'}
            />
          )}
        </Card>
      </div>

      {/* 1차 블록 3 — 소비 분위별 비교 */}
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">소비 분위별 비교</h2>
          <p className="section__note">
            {bundle.focusDay} 하루 기준입니다. 소비가 적은 쪽(1분위)일수록 지급액이 큽니다.
          </p>
        </div>

        <div className="table-wrap">
          <table className="table table--lead">
            <caption className="visually-hidden">소비 분위별 1인당 지급액과 소비액</caption>
            <thead>
              <tr>
                <th scope="col">소비 분위</th>
                <th scope="col" className="n">
                  대상자
                </th>
                <th scope="col" className="n col-md">
                  1인당 지급액
                </th>
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
                    <span className="cell-sub cell-sub--md">
                      지급 {krw(grants[String(row.spend_decile)] ?? 0)}
                    </span>
                    <span className="cell-sub cell-sub--lg">
                      소비 {krw(row.cm_today_total_incl_online)}
                    </span>
                  </td>
                  <td className="n col-md">{krw(grants[String(row.spend_decile)] ?? 0)}</td>
                  <td className="n col-lg">{krw(row.cm_today_total_incl_online)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="section__note">
          대상자 {int(deciles.reduce((s, r) => s + r.agents, 0))}명
        </p>
      </section>

      {/* 접어두는 것 — 기본 닫힘 */}
      <section className="section">
        <h2 className="section__title">상세 보기</h2>
        <div>
          <Disclosure
            title="일자별 원자료"
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
                      <th scope="col" className="n col-lg">
                        정책 지갑
                      </th>
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
                        <td className="n col-lg">{krw(row.policy_paid)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState
                title="일자별 집계가 없습니다"
                body={events.reason ?? '결제 기록 파일이 만들어지지 않았습니다.'}
              />
            )}
          </Disclosure>

          <Disclosure title="내보내기">
            <p className="card__note">
              이 실행에 남아 있는 원자료입니다. 없는 파일은 실행이 그 단계까지 가지 못했다는
              뜻입니다.
            </p>
            <dl className="dl">
              {Object.entries(indexItem.artifacts).map(([key, present]) => (
                <div className="dl__item" key={key}>
                  <dt className="dl__k">{ARTIFACT_LABEL[key] ?? key}</dt>
                  <dd className="dl__v">{present ? '있음' : '없음'}</dd>
                </div>
              ))}
            </dl>
            <div className="row">
              <Button variant="secondary" icon={<DownloadIcon size={18} />} disabled>
                원자료 내려받기
              </Button>
              <span className="card__note">서버에 연결하면 사용할 수 있습니다.</span>
            </div>
          </Disclosure>

          <Disclosure title="이 숫자의 출처">
            <dl className="dl">
              <div className="dl__item">
                <dt className="dl__k">결제 기록</dt>
                <dd className="dl__v wrap">{events.source ?? '없음'}</dd>
              </div>
              <div className="dl__item">
                <dt className="dl__k">분위별 집계</dt>
                <dd className="dl__v wrap">{bundle.dayAggregate.source_file}</dd>
              </div>
              <div className="dl__item">
                <dt className="dl__k">쿠폰 사용 가능 매장</dt>
                <dd className="dl__v">
                  {events.poi_summary
                    ? `${int(events.poi_summary.poi_eligible)}곳 / 전체 ${int(
                        events.poi_summary.poi_total,
                      )}곳 (${dec((events.poi_summary.poi_eligible / events.poi_summary.poi_total) * 100, 1)}%)`
                    : '알 수 없음'}
                </dd>
              </div>
              <div className="dl__item">
                <dt className="dl__k">쓸 수 없는 값</dt>
                <dd className="dl__v">
                  {events.null_only_fields && events.null_only_fields.length > 0
                    ? `${events.null_only_fields.join(', ')} — 전부 비어 있어 집계에 쓰지 않았습니다`
                    : '없음'}
                </dd>
              </div>
            </dl>
          </Disclosure>
        </div>
      </section>
    </div>
  );
}
