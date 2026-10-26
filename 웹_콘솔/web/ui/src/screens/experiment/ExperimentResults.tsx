/**
 * 실험 결과 — 일별 추이와 정책 OFF/ON 비교.
 *
 * 공통 사전 기간은 두 팔이 같은 기록을 쓴다. 그래서 회색 한 줄로 그리고, 시행일부터 OFF(실선)·ON(점선)이 갈라진다.
 * 비교 표는 실험 코드(`no_smoking_zone.py score`)와 같은 묶음·결측 규칙의 **미리보기**다. 두 팔이 끝나기 전에는 중간값이다.
 */
import { useMemo } from 'react';
import { Badge } from '../../components/Badge';
import { Card, Stat } from '../../components/Card';
import { EffectBars } from '../../components/EffectBars';
import { EmptyState, ErrorState, SkeletonCard } from '../../components/Feedback';
import { LineChart } from '../../components/LineChart';
import type { LineSeries } from '../../components/LineChart';
import { dec, int, krw } from '../../lib/format';
import { experimentsApi, SMOKING_TEXT, useResource } from '../../lib/experiments';
import type { Comparison, DailyPoint, DailySeries, ExperimentSummary, Validation } from '../../lib/experiments';
import { useExperiment } from './ExperimentWorkspace';

type Metric = (point: DailyPoint) => number | null;

const TONE = { shared_pre: 'neutral', control: 'primary', treatment: 'alt', single: 'primary' } as const;

function chartSeries(series: DailySeries[], metric: Metric, cohort: number): LineSeries[] {
  return series
    .filter((s) => s.points.length)
    .map((s) => ({
      key: s.arm_id,
      label: s.label,
      tone: TONE[s.role],
      dashed: s.role === 'treatment',
      // 진행 중인 날(마지막 줄이 덜 쓰였거나 대상자가 덜 찬 날)은 하루 전체 값이 아니라서 선을 잇지 않는다
      points: s.points.map((p) => ({
        x: p.day,
        y: p.partial_tail || (cohort > 0 && p.unique_agents < cohort) ? null : metric(p),
      })),
    }));
}

const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? null : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
const ciText = (ci: [number, number] | null) =>
  ci ? `95% 신뢰구간 ${pct(ci[0])} ~ ${pct(ci[1])}` : undefined;

/** 정책 실험 — 같은 시민·같은 날의 정책 있음 − 없음 */
const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;

/** 실측 연구와 비교 — 같은 업종 묶음·정의로 다시 센 시뮬 값과 공개 연구의 추정치 */
function ValidationCard({ validation }: { validation: Validation }) {
  return (
    <Card title="실측 연구와 비교" className="c12">
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">항목</th>
              <th scope="col" className="n">시뮬</th>
              <th scope="col" className="n">실측</th>
              <th scope="col">방향</th>
              <th scope="col">출처</th>
            </tr>
          </thead>
          <tbody>
            {validation.rows.map((row) => (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                <td className="n num">
                  {signed(row.sim_pct)}
                  {row.sim_ci95_pct ? (
                    <span className="cell-sub">
                      {signed(row.sim_ci95_pct[0])} ~ {signed(row.sim_ci95_pct[1])}
                    </span>
                  ) : null}
                </td>
                <td className="n num">{signed(row.truth_pct)}</td>
                <td>{Math.sign(row.sim_pct) === Math.sign(row.truth_pct) ? '같음' : '반대'}</td>
                <td className="cell-sub wrap">{row.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="card__note">{validation.basis}</p>
      <p className="card__note">{validation.note}</p>
    </Card>
  );
}

function PolicyEffect({ comparison, marker }: { comparison: Comparison; marker?: { x: string; label: string } }) {
  if (!comparison.available || !comparison.rows) {
    return (
      <Card title="정책 효과" className="c12">
        <p className="card__note">{comparison.reason ?? '비교할 날이 없습니다.'}</p>
      </Card>
    );
  }
  const total = comparison.rows.find((r) => r.key === 'total');
  const shop = comparison.rows.find((r) => r.key === 'shop');
  const categories = comparison.rows.filter((r) => r.group === 'category');
  const daily = comparison.daily ?? [];
  // 캐시백(다음 달 적립)·거리두기는 결제 때 지갑에서 나가는 돈이 없다. 0원으로 보이지 않게 비운다
  const wallet = comparison.wallet && comparison.wallet.on_total > 0 ? comparison.wallet : null;
  return (
    <section className="section">
      <div className="section__head">
        <h2 className="section__title">정책 효과</h2>
        <p className="section__note">
          {int(comparison.agents ?? 0)}명 · {int(comparison.days?.length ?? 0)}일 · 같은 시민·같은 날 비교
        </p>
      </div>
      <div className="grid statrow">
        <Stat className="c4" label="총소비 (온라인 포함)" value={pct(total?.effect)} hint={ciText(total?.ci95 ?? null)} unknownReason="기록 없음" />
        <Stat className="c4" label="가게 소비 (온라인 제외)" value={pct(shop?.effect)} hint={ciText(shop?.ci95 ?? null)} unknownReason="기록 없음" />
        <Stat
          className="c4"
          label="지원금 사용 (1인 하루)"
          value={wallet ? krw(wallet.per_agent_day) : '해당 없음'}
          hint={wallet ? `기간 합계 ${krw(wallet.on_total)}` : '결제 때 쓰는 지원금이 없는 정책'}
        />
      </div>
      <div className="grid">
        <Card className="c7" title="업종별 효과">
          <EffectBars
            items={categories.map((r) => ({
              key: r.key,
              label: r.label,
              effect: r.effect,
              ci: r.ci95,
              significant: r.significant,
              sub: `1인 하루 ${krw(r.off_per_agent_day)} → ${krw(r.on_per_agent_day)}`,
            }))}
          />
          <p className="card__note">옅은 막대는 95% 신뢰구간이 0을 포함합니다.</p>
        </Card>
        <Card className="c5">
          <LineChart
            title="날짜별 총소비 효과"
            series={[
              {
                key: 'effect',
                label: '정책 있음 − 없음',
                tone: 'primary',
                points: daily.map((d) => ({ x: d.day, y: d.effect === null ? null : d.effect * 100 })),
              },
            ]}
            format={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}
            marker={marker}
          />
          <p className="card__note">95% 신뢰구간: 시민 단위 부트스트랩 500회</p>
        </Card>
      </div>
    </section>
  );
}

function groupLabel(group: string, experiment: ExperimentSummary): string {
  const facilities = experiment.analysis.facilities ?? {};
  const districts = experiment.analysis.districts ?? {};
  const [facility, detail] = group.split(':');
  const head = facilities[facility] ?? facility;
  if (!detail) return head;
  return `${head} · ${districts[detail] ?? SMOKING_TEXT[detail] ?? detail}`;
}

function ComparisonCard({ comparison, experiment }: { comparison: Comparison; experiment: ExperimentSummary }) {
  if (!comparison.available || !comparison.comparisons || !comparison.post) {
    const pre = comparison.pre;
    return (
      <Card title="정책 있음·없음 비교" aside={<Badge>대기</Badge>} className="c12">
        <p className="card__note">{comparison.reason ?? '비교할 수 있는 날이 아직 없습니다.'}</p>
        {pre && Object.keys(pre.totals).length ? (
          <div className="table-wrap">
            <table className="table">
              <caption className="table__caption">정책 전 {int(pre.days)}일 · 대상 시설 이용</caption>
              <thead>
                <tr>
                  <th scope="col">묶음</th>
                  <th scope="col" className="n">결제 건수</th>
                  <th scope="col" className="n">매출</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(pre.totals).map(([group, value]) => (
                  <tr key={group}>
                    <th scope="row">{groupLabel(group, experiment)}</th>
                    <td className="n num">{int(value.payment_count)}</td>
                    <td className="n num">{krw(value.revenue_krw)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Card>
    );
  }
  const rows = comparison.comparisons.filter((row) => row.metric === 'revenue_krw');
  const counts = new Map(comparison.comparisons.filter((row) => row.metric === 'payment_count').map((row) => [row.group, row]));
  return (
    <Card
      title="정책 있음·없음 비교"
      aside={<Badge tone={comparison.status === 'complete_preview' ? 'ok' : 'warn'}>{comparison.status === 'complete_preview' ? '완료' : '중간 집계'}</Badge>}
      note={`같은 사람·같은 날 ${int(comparison.post.observed_paired_agent_days)}건 · ${comparison.post.days.length}일`}
      className="c12"
    >
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">묶음</th>
              <th scope="col" className="n">정책 없음</th>
              <th scope="col" className="n">정책 있음</th>
              <th scope="col" className="n">ON − OFF</th>
              <th scope="col" className="n">변화율</th>
              <th scope="col" className="n">결제 건수</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const count = counts.get(row.group);
              return (
                <tr key={row.group}>
                  <th scope="row">{groupLabel(row.group, experiment)}</th>
                  <td className="n num">{krw(row.off)}</td>
                  <td className="n num">{krw(row.on)}</td>
                  <td className="n num">{krw(row.on_minus_off)}</td>
                  <td className="n num">{row.percent_change === null ? '—' : `${dec(row.percent_change, 1)}%`}</td>
                  <td className="n num">{count ? `${int(count.off)} → ${int(count.on)}` : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="card__note">{comparison.rules}</p>
    </Card>
  );
}

export function ExperimentResults() {
  const { experiment } = useExperiment();
  const id = experiment.experiment_id;
  const daily = useResource(() => experimentsApi.daily(id), id);
  const comparison = useResource(() => experimentsApi.comparison(id), id);
  const series = useMemo(() => daily.data?.series ?? [], [daily.data]);
  const marker = experiment.window.effective_date ? { x: experiment.window.effective_date, label: '정책 시행' } : undefined;
  const hasTarget = experiment.kind === 'no_smoking_zone';
  const paired = experiment.kind === 'policy_paired';
  const cohort = experiment.cohort_size;

  if (daily.loading && !daily.data) return <SkeletonCard rows={6} />;
  if (daily.error) return <ErrorState title="추이를 불러오지 못했습니다" body={daily.error.message} />;
  if (!series.some((s) => s.points.length)) {
    return <EmptyState title="실행 기록이 없습니다" body="개요에서 기록을 받으면 나타납니다." />;
  }

  return (
    <div className="stack">
      {paired && comparison.data ? <PolicyEffect comparison={comparison.data} marker={marker} /> : null}
      {paired && experiment.validation?.rows?.length ? <ValidationCard validation={experiment.validation} /> : null}
      <div className="grid">
        {hasTarget ? (
          <>
            <Card className="c6">
              <LineChart
                title="정책 대상 시설 매출 (하루)"
                series={chartSeries(series, (p) => p.target?.total.revenue_krw ?? null, cohort)}
                format={(v) => krw(v)}
                marker={marker}
              />
            </Card>
            <Card className="c6">
              <LineChart
                title="정책 대상 시설 결제 건수 (하루)"
                series={chartSeries(series, (p) => p.target?.total.payment_count ?? null, cohort)}
                format={(v) => int(v)}
                marker={marker}
              />
            </Card>
          </>
        ) : null}
        <Card className="c6">
          <LineChart
            title="오프라인 소비 합계 (하루)"
            series={chartSeries(series, (p) => p.totals.spend_offline ?? null, cohort)}
            format={(v) => krw(v)}
            marker={marker}
          />
        </Card>
        <Card className="c6">
          <LineChart
            title="평균 기분 (0~1)"
            series={chartSeries(series, (p) => p.means.mood ?? null, cohort)}
            format={(v) => dec(v, 2)}
            marker={marker}
          />
        </Card>
        {series.some((s) => s.points.some((p) => (p.status.skipped ?? 0) > 0)) ? (
        <Card className="c6">
          <LineChart
            title="기록이 빠진 사람 (하루)"
            series={chartSeries(series, (p) => p.status.skipped ?? 0, cohort)}
            format={(v) => int(v)}
            marker={marker}
          />
        </Card>
        ) : null}
        <Card className="c6">
          <LineChart
            title="결제 영수증 수 (하루)"
            series={chartSeries(series, (p) => p.totals.receipt_count ?? null, cohort)}
            format={(v) => int(v)}
            marker={marker}
          />
        </Card>
      </div>
      {comparison.data && !paired ? <ComparisonCard comparison={comparison.data} experiment={experiment} /> : null}
      {comparison.error ? <ErrorState title="비교를 계산하지 못했습니다" body={comparison.error.message} /> : null}
    </div>
  );
}
