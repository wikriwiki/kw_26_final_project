/**
 * 실험 결과 — 일별 추이와 정책 OFF/ON 비교.
 *
 * 공통 사전 기간은 두 팔이 같은 기록을 쓴다. 그래서 회색 한 줄로 그리고, 시행일부터 OFF(실선)·ON(점선)이 갈라진다.
 * 비교 표는 실험 코드(`no_smoking_zone.py score`)와 같은 묶음·결측 규칙의 **미리보기**다. 두 팔이 끝나기 전에는 중간값이다.
 */
import { useMemo } from 'react';
import { Badge } from '../../components/Badge';
import { Card } from '../../components/Card';
import { EmptyState, ErrorState, SkeletonCard } from '../../components/Feedback';
import { LineChart } from '../../components/LineChart';
import type { LineSeries } from '../../components/LineChart';
import { dec, int, krw } from '../../lib/format';
import { experimentsApi, SMOKING_TEXT, useResource } from '../../lib/experiments';
import type { Comparison, DailyPoint, DailySeries, ExperimentSummary, GroupFilters } from '../../lib/experiments';
import { useExperiment } from './ExperimentWorkspace';
import { GroupEffectCard, PolicyEffect, ValidationCard } from './PolicyPanels';

/** 집단 질의 탭으로 그 집단을 골라 둔 채 넘어간다 */
export function filtersQuery(filters: GroupFilters): string {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, list]) => (list ?? []).forEach((v) => params.append(key, v)));
  return params.toString();
}

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
      {paired && comparison.data ? (
        <GroupEffectCard comparison={comparison.data} askHref={(filters) => `/experiments/${encodeURIComponent(id)}/query?${filtersQuery(filters)}`} />
      ) : null}
      {/* 정책 실험이 아닌 실험(금연구역)은 정책 있음·없음 비교 표가 핵심 결과다 — 맨 위에 둔다 */}
      {comparison.data && !paired ? <ComparisonCard comparison={comparison.data} experiment={experiment} /> : null}
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
        {/* 평균 기분·기록이 빠진 사람 줄은 정책 판단에 쓰이지 않아 그리지 않는다. 소비액·결제 건수는 둔다 */}
        <Card className="c6">
          <LineChart
            title="결제 건수 (하루)"
            series={chartSeries(series, (p) => p.totals.receipt_count ?? null, cohort)}
            format={(v) => int(v)}
            marker={marker}
          />
        </Card>
      </div>
      {comparison.error ? <ErrorState title="비교를 계산하지 못했습니다" body={comparison.error.message} /> : null}
      {/* 사후 검증은 정책 시행 전에 보는 결과(효과·집단·추이) 다음, 맨 끝에 둔다 */}
      {experiment.validation?.rows?.length ? <ValidationCard validation={experiment.validation} /> : null}
    </div>
  );
}
