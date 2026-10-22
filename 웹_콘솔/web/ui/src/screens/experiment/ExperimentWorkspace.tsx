/**
 * 실험 작업공간 — `/experiments/:experimentId/*`.
 *
 * 실행(run) 하나가 아니라 **실험** 하나를 연다. 실험은 여러 팔을 갖고(예: 공통 사전 → OFF/ON),
 * 각 팔은 실험 서버에서 읽기 전용으로 받아 온 실제 기록이다.
 * 탭은 넷: 개요(진행·동기화) · 결과(추이·비교) · 대상자(기록·인터뷰) · 집단 질의.
 */
import { Link, NavLink, Outlet, useOutletContext, useParams } from 'react-router-dom';
import { ErrorState, SkeletonCard } from '../../components/Feedback';
import { Button } from '../../components/Button';
import { ArrowLeftIcon } from '../../components/Icon';
import { experimentsApi, useResource } from '../../lib/experiments';
import type { ExperimentSummary } from '../../lib/experiments';

export interface ExperimentContext {
  experiment: ExperimentSummary;
  reload: () => void;
}

export function useExperiment(): ExperimentContext {
  return useOutletContext<ExperimentContext>();
}

const TABS = [
  { to: '', label: '개요', end: true },
  { to: 'results', label: '결과', end: false },
  { to: 'agents', label: '시민', end: false },
  { to: 'query', label: '집단 질의', end: false },
];

export function ExperimentWorkspace() {
  const { experimentId = '' } = useParams();
  const detail = useResource(() => experimentsApi.detail(experimentId), experimentId);

  return (
    <main id="main" className="home expws">
      <div className="stack">
        <Link className="expws__back" to="/">
          <ArrowLeftIcon size={16} />
          <span>목록으로</span>
        </Link>
        <header className="pagehead">
          <div className="pagehead__text">
            <h1 className="pagehead__title">{detail.data?.title ?? experimentId}</h1>
            {detail.data?.summary ? <p className="pagehead__purpose">{detail.data.summary}</p> : null}
          </div>
        </header>
        <nav className="exptabs" aria-label="실험 화면">
          {TABS.map((tab) => (
            <NavLink
              key={tab.label}
              to={tab.to}
              end={tab.end}
              className={({ isActive }) => (isActive ? 'exptabs__tab is-active' : 'exptabs__tab')}
            >
              {tab.label}
            </NavLink>
          ))}
        </nav>
        {detail.loading && !detail.data ? (
          <SkeletonCard rows={4} />
        ) : detail.error || !detail.data ? (
          <ErrorState
            title="실험을 불러오지 못했습니다"
            body={detail.error?.message ?? '알 수 없는 오류'}
            detail={detail.error?.detail}
            action={
              <Button variant="secondary" onClick={detail.reload}>
                다시 시도
              </Button>
            }
          />
        ) : (
          <Outlet context={{ experiment: detail.data, reload: detail.reload } satisfies ExperimentContext} />
        )}
      </div>
    </main>
  );
}
