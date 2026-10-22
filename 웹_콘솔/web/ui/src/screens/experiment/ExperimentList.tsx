/**
 * 홈의 실험 목록 — 레지스트리에 등록된 실험과 팔마다의 진행.
 *
 * API 가 없으면(픽스처만으로 띄운 개발 화면) 이 묶음을 그리지 않는다. 없는 실험을 그린 척하지 않는다.
 */
import { Link } from 'react-router-dom';
import { Badge } from '../../components/Badge';
import { Meter } from '../../components/Meter';
import { int } from '../../lib/format';
import { experimentsApi, PHASE_TEXT, useResource } from '../../lib/experiments';

/** 기록을 다 받은 실험을 먼저. 같은 묶음 안에서는 레지스트리 순서를 지킨다 */
function sortByReady<T extends { arms: Array<{ phase: string }> }>(items: T[]): T[] {
  const ready = (e: T) => (e.arms.length > 0 && e.arms.every((a) => a.phase === 'complete') ? 0 : 1);
  return items.map((e, i) => ({ e, i })).sort((a, b) => ready(a.e) - ready(b.e) || a.i - b.i).map((x) => x.e);
}

export function ExperimentList({ exclude = [] }: { exclude?: string[] }) {
  const list = useResource(() => experimentsApi.list(), 'experiments');
  const items = (list.data?.items ?? []).filter((e) => !exclude.includes(e.experiment_id));
  if (list.loading || list.error || items.length === 0) return null;
  return (
    <section className="section" aria-labelledby="experiments-title">
      <header className="section__head">
        <h2 className="section__title" id="experiments-title">
          다른 실험
        </h2>
      </header>
      <ul className="explist">
        {sortByReady(items).map((experiment) => (
          <li key={experiment.experiment_id}>
            <Link className="exprow" to={`/experiments/${experiment.experiment_id}`}>
              <span className="exprow__title">{experiment.title}</span>
              <span className="exprow__meta">
                {experiment.window.start ? `${experiment.window.start}부터 ${int(experiment.window.days)}일 · ` : ''}
                {int(experiment.cohort_size)}명{experiment.source_branch ? ` · ${experiment.source_branch}` : ''}
              </span>
              <span className="exprow__arms">
                {experiment.arms.map((arm) => (
                  <span key={arm.arm_id} className="exprow__arm">
                    <span className="exprow__armhead">
                      <span>{arm.label}</span>
                      <Badge tone={arm.phase === 'complete' ? 'ok' : arm.phase === 'running' ? 'info' : 'neutral'} plain>
                        {PHASE_TEXT[arm.phase]}
                      </Badge>
                    </span>
                    <Meter ratio={arm.days ? arm.days_complete / arm.days : 0} label={`${arm.label} 완료 일수`} />
                    <span className="num">
                      {int(arm.days_complete)} / {int(arm.days)}일
                    </span>
                  </span>
                ))}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
