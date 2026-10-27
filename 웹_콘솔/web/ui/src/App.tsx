/**
 * 라우팅 — `docs/DESIGN_IA_RUN_FIRST.md` §2.
 *
 * 정보구조가 "기능 → 실행"에서 "실행 → 기능"으로 뒤집혔다.
 *   `/`               시뮬레이션 목록 (셸 없음 — 아직 run 컨텍스트가 없다)
 *   `/new`            새 시뮬레이션 만들기 (셸 없음)
 *   `/runs/:runId/*`  그 실행의 개요·모니터·결과·시각화·정책 (run 셸)
 *
 * 구 경로는 죽이지 않고 새 구조로 넘긴다. 이미 공유된 링크가 있기 때문이다 (스펙 §9 deep-linking).
 */
import { lazy, Suspense, useEffect, type ComponentType, type ReactNode } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom';
import { AppShell } from './app/AppShell';
import { loadRunSummary, RunProvider } from './app/RunContext';
import { RouteTransition } from './app/RouteTransition';
import { HomeScreen, UnknownRunScreen } from './screens/HomeScreen';
import { NotFoundScreen } from './screens/NotFoundScreen';
import { ErrorState } from './components/Feedback';
import { isNotFound, useLoad } from './lib/runData';
import { CAN_EDIT } from './lib/runtime';

/**
 * 화면은 처음 열 때 받는다. 목록 화면 하나 보려고 보고서·정책 편집기·3D 임베드 코드까지
 * 내려받지 않게 한다. 이름 붙은 export 를 default 로 바꿔 끼우는 얇은 감싸개다.
 */
function screen<K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K) {
  return lazy(() => load().then((m) => ({ default: m[name] })));
}
const AgentScreen = screen(() => import('./screens/AgentScreen'), 'AgentScreen');
const KitScreen = screen(() => import('./screens/KitScreen'), 'KitScreen');
const MonitorScreen = screen(() => import('./screens/MonitorScreen'), 'MonitorScreen');
const NewRunScreen = screen(() => import('./screens/NewRunScreen'), 'NewRunScreen');
const OverviewScreen = screen(() => import('./screens/OverviewScreen'), 'OverviewScreen');
const PolicyScreen = screen(() => import('./screens/PolicyScreen'), 'PolicyScreen');
const ReportScreen = screen(() => import('./screens/ReportScreen'), 'ReportScreen');
const ResultsScreen = screen(() => import('./screens/ResultsScreen'), 'ResultsScreen');
const VisualizationScreen = screen(() => import('./screens/VisualizationScreen'), 'VisualizationScreen');
const ExperimentAgents = screen(() => import('./screens/experiment/ExperimentAgents'), 'ExperimentAgents');
const ExperimentOverview = screen(() => import('./screens/experiment/ExperimentOverview'), 'ExperimentOverview');
const ExperimentQuery = screen(() => import('./screens/experiment/ExperimentQuery'), 'ExperimentQuery');
const ExperimentResults = screen(() => import('./screens/experiment/ExperimentResults'), 'ExperimentResults');
const ExperimentWorkspace = screen(() => import('./screens/experiment/ExperimentWorkspace'), 'ExperimentWorkspace');

/**
 * 첫 화면을 그린 뒤 브라우저가 한가할 때 나머지 화면 코드를 미리 받아 둔다.
 * 나눠 받기로 첫 화면은 가벼워졌지만, 다른 화면을 처음 열 때 코드 한 번을 더 기다리게 됐다.
 * 미리 받아 두면 그 기다림이 사라진다. 이미 받은 모듈은 브라우저가 다시 받지 않는다.
 */
const SCREEN_MODULES = [
  () => import('./screens/OverviewScreen'),
  () => import('./screens/ResultsScreen'),
  () => import('./screens/VisualizationScreen'),
  () => import('./screens/AgentScreen'),
  () => import('./screens/MonitorScreen'),
  () => import('./screens/PolicyScreen'),
  () => import('./screens/ReportScreen'),
  () => import('./screens/experiment/ExperimentWorkspace'),
  () => import('./screens/experiment/ExperimentOverview'),
  () => import('./screens/experiment/ExperimentResults'),
  () => import('./screens/experiment/ExperimentAgents'),
  () => import('./screens/experiment/ExperimentQuery'),
];

function usePrefetchScreens() {
  useEffect(() => {
    const idle = (window as Window & { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback;
    const run = () => SCREEN_MODULES.forEach((load) => void load().catch(() => undefined));
    const handle = idle ? idle(run) : window.setTimeout(run, 1200);
    return () => {
      if (!idle) window.clearTimeout(handle);
    };
  }, []);
}

/** 화면 코드를 받는 동안 자리만 잡아 둔다. 문구를 띄우지 않아야 빠른 망에서 깜빡임이 없다 */
function ScreenFallback() {
  return <div className="screen-fallback" aria-busy="true" />;
}

// 셸 스타일은 화면 스타일 뒤에 온다 — 같은 특이도면 나중 규칙이 이긴다
import './styles/shell.css';

/** 셸 밖 화면에도 본문 폭과 여백은 필요하다 (셸 안에서는 `.page` 가 맡던 몫) */
function BareScreen({ children }: { children: ReactNode }) {
  return (
    <main id="main" className="home">
      <RouteTransition>{children}</RouteTransition>
    </main>
  );
}

/**
 * 주소의 run 을 서버에서 받아 컨텍스트로 올린다. 서버가 모른다고 하면 셸을 그리지 않는다.
 * 받는 동안은 본문 자리만 잡아 둔다 — 셸은 실행이 정해져야 성립한다.
 */
function RunWorkspace() {
  const { runId = '' } = useParams();
  const summary = useLoad(runId ? `summary:${runId}` : null, () => loadRunSummary(runId));
  if (!runId || isNotFound(summary)) return <UnknownRunScreen runId={runId} />;
  if (summary.status === 'error') {
    return (
      <BareScreen>
        <ErrorState
          title="실행을 불러오지 못했습니다"
          body={summary.error.message}
          detail={summary.error.detail}
          action={
            <button type="button" className="btn" onClick={summary.retry}>
              다시 시도
            </button>
          }
        />
      </BareScreen>
    );
  }
  if (summary.status !== 'ready') return <ScreenFallback />;
  return (
    <RunProvider summary={summary.data}>
      <AppShell />
    </RunProvider>
  );
}

/**
 * 구 경로 → 새 경로. `?run=` 을 경로 조각으로 올리고 나머지 질의는 그대로 넘긴다.
 *   `/results?run=X`         → `/runs/X/results`
 *   `/visualize?run=X&day=D` → `/runs/X/visualize?day=D`
 * run 을 모르면 목록으로 보낸다 — 아무 실행이나 골라 보여주지 않는다.
 *
 * `replace` 로 넘겨 히스토리에 구 주소를 남기지 않는다. 뒤로 가기가 리다이렉트에
 * 걸려 제자리로 튕기는 일을 막는다. state 는 그대로 실어 보낸다 (§9 back-behavior).
 */
function LegacyRunRedirect({ feature }: { feature: string }) {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  const raw = params.get('run');
  if (!raw) return <Navigate to="/" replace />;

  params.delete('run');
  const query = params.toString();
  return (
    <Navigate to={`/runs/${raw}/${feature}${query ? `?${query}` : ''}`} replace state={location.state} />
  );
}

export function App() {
  usePrefetchScreens();
  return (
    <BrowserRouter>
      <Suspense fallback={<ScreenFallback />}>
      <Routes>
        {/* 개발 전용 — 앱 셸 밖에 둬서 사용자 내비게이션에 나타나지 않는다 */}
        <Route path="/__kit" element={!CAN_EDIT ? <Navigate to="/" replace /> : <KitScreen />} />

        {/* 실행을 고르는 자리. 셸이 없다 */}
        <Route
          path="/"
          element={
            <RouteTransition>
              <HomeScreen />
            </RouteTransition>
          }
        />
        <Route
          path="/new"
          element={
            !CAN_EDIT ? (
              <Navigate to="/" replace />
            ) : (
              <BareScreen>
                <NewRunScreen />
              </BareScreen>
            )
          }
        />

        {/* 작업공간 — run 이 정해진 다음의 모든 화면.
            전환은 본문에만 건다. 셸(사이드바)은 제자리에 있어야 "같은 작업공간
            안에서 화면만 바뀌었다"로 읽힌다 (§7 excessive-motion) */}
        <Route path="/runs/:runId" element={<RunWorkspace />}>
          <Route
            index
            element={
              <RouteTransition>
                <OverviewScreen />
              </RouteTransition>
            }
          />
          <Route
            path="monitor"
            element={
              <RouteTransition>
                <MonitorScreen />
              </RouteTransition>
            }
          />
          <Route
            path="results"
            element={
              <RouteTransition>
                <ResultsScreen />
              </RouteTransition>
            }
          />
          {/* 지도는 무거운 임베드를 안고 있다. 떠오르는 동작 없이 불투명도만 */}
          <Route
            path="visualize"
            element={
              <RouteTransition motion="fade">
                <VisualizationScreen />
              </RouteTransition>
            }
          />
          <Route
            path="agents"
            element={
              <RouteTransition>
                <AgentScreen />
              </RouteTransition>
            }
          />
          {/* 보고서도 iframe 임베드다. 떠오르는 동작 없이 불투명도만 */}
          <Route
            path="report"
            element={
              <RouteTransition motion="fade">
                <ReportScreen />
              </RouteTransition>
            }
          />
          <Route
            path="policy"
            element={
              <RouteTransition>
                <PolicyScreen />
              </RouteTransition>
            }
          />
          <Route
            path="*"
            element={
              <RouteTransition>
                <NotFoundScreen />
              </RouteTransition>
            }
          />
        </Route>

        {/* 실험 작업공간 — 여러 팔을 가진 실험 하나. 실행(run) 셸과 따로 둔다:
            run 셸은 아직 픽스처 기반이고, 실험은 API 의 실제 기록을 읽는다 */}
        <Route path="/experiments/:experimentId" element={<ExperimentWorkspace />}>
          <Route
            index
            element={
              <RouteTransition>
                <ExperimentOverview />
              </RouteTransition>
            }
          />
          <Route
            path="results"
            element={
              <RouteTransition>
                <ExperimentResults />
              </RouteTransition>
            }
          />
          <Route
            path="agents"
            element={
              <RouteTransition>
                <ExperimentAgents />
              </RouteTransition>
            }
          />
          <Route
            path="query"
            element={
              <RouteTransition>
                <ExperimentQuery />
              </RouteTransition>
            }
          />
          <Route path="*" element={<NotFoundScreen />} />
        </Route>

        {/* 구 경로 — 공유된 링크가 죽으면 안 된다 */}
        <Route path="/results" element={<LegacyRunRedirect feature="results" />} />
        <Route path="/visualize" element={<LegacyRunRedirect feature="visualize" />} />
        {/* 모니터는 run 을 담고 있지 않았다. 고를 수 있는 곳으로 보낸다 */}
        <Route path="/monitor" element={<Navigate to="/" replace />} />
        {/* 실험 목록은 홈에 있다 */}
        <Route path="/experiments" element={<Navigate to="/" replace />} />

        <Route
          path="*"
          element={
            <BareScreen>
              <NotFoundScreen />
            </BareScreen>
          }
        />
      </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
