/**
 * 홈의 정책 목록 — 정책 본런마다 한 줄. 실행(지도·시민·결과)과 실험(정책 있음·없음 비교)을 한 줄에 묶는다.
 *
 * 예전에는 같은 정책이 '실험'과 '실행' 두 목록에 따로 나왔다. 실행 폴더의 이름표(run_meta)가
 * 어느 실험에서 왔는지 알려 주므로 한 줄로 합친다. 효과 값은 실험 비교에서 그대로 가져온다.
 */
import { Link } from 'react-router-dom';
import type { RunListItem } from '../app/RunContext';
import { experimentsApi, useResource } from '../lib/experiments';
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

export function PolicyList({ runs }: { runs: RunListItem[] }) {
  const rows = runs
    .filter((run) => run.index.meta)
    .sort((a, b) => (b.firstDay ?? '').localeCompare(a.firstDay ?? ''));
  if (rows.length === 0) return null;
  return (
    <section className="section" aria-labelledby="policies-title">
      <header className="section__head">
        <h2 className="section__title" id="policies-title">
          정책
        </h2>
        <span className="section__note num">{int(rows.length)}개</span>
      </header>
      <ul className="pollist">
        {rows.map((run) => {
          const meta = run.index.meta!;
          const base = `/runs/${encodeURIComponent(run.id)}`;
          return (
            <li key={run.id} className="polrow">
              <Link className="polrow__main" to={base}>
                <span className="polrow__title">{meta.title ?? run.id}</span>
                <span className="polrow__meta num">
                  {run.firstDay} ~ {run.lastDay?.slice(5)} · {int(run.daysPresent)}일 · {int(run.agentsTarget ?? 0)}명
                </span>
              </Link>
              <Effect experimentId={meta.experiment_id} />
              <nav className="polrow__links" aria-label={`${meta.title ?? run.id} 화면`}>
                <Link to={`${base}/visualize`}>지도</Link>
                <Link to={`${base}/agents`}>시민</Link>
                <Link to={`/experiments/${encodeURIComponent(meta.experiment_id)}/results`}>정책 있음·없음 비교</Link>
              </nav>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
