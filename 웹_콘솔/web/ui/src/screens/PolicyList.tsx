/**
 * 홈의 정책 목록 — 정책 본런마다 한 줄. 실행(지도·시민·결과)과 실험(정책 있음·없음 비교)을 한 줄에 묶는다.
 *
 * 예전에는 같은 정책이 '실험'과 '실행' 두 목록에 따로 나왔다. 실행 폴더의 이름표(run_meta)가
 * 어느 실험에서 왔는지 알려 주므로 한 줄로 합친다. 효과 값은 실험 비교에서 그대로 가져온다.
 */
import { Link } from 'react-router-dom';
import type { RunListItem } from '../app/RunContext';
import { experimentsApi, useResource } from '../lib/experiments';
import type { ExperimentSummary } from '../lib/experiments';
import { int } from '../lib/format';

const pct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;

function Effect({ experimentId }: { experimentId: string }) {
  const comparison = useResource(() => experimentsApi.comparison(experimentId), `comparison:${experimentId}`);
  const total = comparison.data?.rows?.find((r) => r.key === 'total') ?? null;
  if (comparison.loading) return <span className="polrow__effect polrow__effect--wait">총소비 …</span>;
  if (!total || total.effect === null) return <span className="polrow__effect">총소비 —</span>;
  return (
    <span className="polrow__effect">
      총소비 <b className="num">{pct(total.effect)}</b>
      <small>{total.significant ? '유의' : '유의하지 않음'}</small>
    </span>
  );
}

/** 실행으로 세우지 않은 실험(금연구역)의 대표 효과 — 정책 대상 시설 매출의 정책 있음·없음 변화율 */
function FacilityEffect({ experiment }: { experiment: ExperimentSummary }) {
  const comparison = useResource(() => experimentsApi.comparison(experiment.experiment_id), `comparison:${experiment.experiment_id}`);
  const facilities = experiment.analysis.facilities ?? {};
  const first = Object.keys(facilities)[0];
  const row = comparison.data?.comparisons?.find((c) => c.group === first && c.metric === 'revenue_krw');
  if (comparison.loading) return <span className="polrow__effect polrow__effect--wait">매출 …</span>;
  if (!row || row.percent_change === null) return <span className="polrow__effect">매출 —</span>;
  return (
    <span className="polrow__effect">
      {facilities[first]} 매출 <b className="num">{pct(row.percent_change / 100)}</b>
      <small>시드 1개</small>
    </span>
  );
}

/** 기록을 다 받았고 실행과 이어지지 않은 실험 */
export function finishedExperiments(items: ExperimentSummary[], linked: string[]): ExperimentSummary[] {
  return items.filter(
    (e) => !linked.includes(e.experiment_id) && e.arms.length > 0 && e.arms.every((a) => a.phase === 'complete'),
  );
}

export function PolicyList({ runs }: { runs: RunListItem[] }) {
  const list = useResource(() => experimentsApi.list(), 'experiments');
  const linked = runs.flatMap((r) => (r.index.meta ? [r.index.meta.experiment_id] : []));
  const extra = finishedExperiments(list.data?.items ?? [], linked);
  const rows = runs
    .filter((run) => run.index.meta)
    .sort((a, b) => (b.firstDay ?? '').localeCompare(a.firstDay ?? ''));
  if (rows.length === 0 && extra.length === 0) return null;
  return (
    <section className="section" aria-labelledby="policies-title">
      <header className="section__head">
        <h2 className="section__title" id="policies-title">
          정책
        </h2>
        <span className="section__note num">{int(rows.length + extra.length)}개</span>
      </header>
      <ul className="pollist">
        {rows.map((run) => {
          const meta = run.index.meta!;
          const base = `/runs/${encodeURIComponent(run.id)}`;
          return (
            <li key={run.id} className="polrow">
              <Link className="polrow__main" to={`${base}/results`}>
                <span className="polrow__title">{meta.title ?? run.id}</span>
                <span className="polrow__meta num">
                  {run.firstDay} ~ {run.lastDay?.slice(5)} · {int(run.daysPresent)}일 · {int(run.agentsTarget ?? 0)}명
                </span>
              </Link>
              <Effect experimentId={meta.experiment_id} />
              <nav className="polrow__links" aria-label={`${meta.title ?? run.id} 화면`}>
                <Link to={`${base}/results`}>결과</Link>
                <Link to={`${base}/visualize`}>지도</Link>
                <Link to={`${base}/agents`}>시민 인터뷰</Link>
              </nav>
            </li>
          );
        })}
        {extra.map((experiment) => {
          const base = `/experiments/${encodeURIComponent(experiment.experiment_id)}`;
          return (
            <li key={experiment.experiment_id} className="polrow">
              <Link className="polrow__main" to={`${base}/results`}>
                <span className="polrow__title">{experiment.title}</span>
                <span className="polrow__meta num">
                  {experiment.window.start} · {int(experiment.window.days ?? 0)}일 · {int(experiment.cohort_size)}명
                </span>
              </Link>
              <FacilityEffect experiment={experiment} />
              <nav className="polrow__links" aria-label={`${experiment.title} 화면`}>
                <Link to={`${base}/results`}>결과</Link>
                <Link to={`${base}/query`}>시민 인터뷰</Link>
              </nav>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
