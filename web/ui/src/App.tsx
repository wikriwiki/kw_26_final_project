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
import type { ReactNode } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom';
import { AppShell } from './app/AppShell';
import { RunProvider } from './app/RunContext';
import { RouteTransition } from './app/RouteTransition';
import { AgentScreen } from './screens/AgentScreen';
import { HomeScreen, UnknownRunScreen } from './screens/HomeScreen';
import { KitScreen } from './screens/KitScreen';
import { MonitorScreen } from './screens/MonitorScreen';
import { NewRunScreen } from './screens/NewRunScreen';
import { OverviewScreen } from './screens/OverviewScreen';
import { PolicyScreen } from './screens/PolicyScreen';
import { ReportScreen } from './screens/ReportScreen';
import { ResultsScreen } from './screens/ResultsScreen';
import { VisualizationScreen } from './screens/VisualizationScreen';
import { NotFoundScreen } from './screens/NotFoundScreen';
import { isRunId } from './lib/fixtures';
import { READ_ONLY } from './lib/runtime';

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

/** 주소의 run 을 컨텍스트로 올린다. 모르는 실행이면 셸을 그리지 않는다 */
function RunWorkspace() {
  const { runId } = useParams();
  if (!runId || !isRunId(runId)) return <UnknownRunScreen runId={runId ?? ''} />;
  return (
    <RunProvider runId={runId}>
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
  if (!raw || !isRunId(raw)) return <Navigate to="/" replace />;

  params.delete('run');
  const query = params.toString();
  return (
    <Navigate to={`/runs/${raw}/${feature}${query ? `?${query}` : ''}`} replace state={location.state} />
  );
}

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* 개발 전용 — 앱 셸 밖에 둬서 사용자 내비게이션에 나타나지 않는다 */}
        <Route path="/__kit" element={READ_ONLY ? <Navigate to="/" replace /> : <KitScreen />} />

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
            READ_ONLY ? (
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

        {/* 구 경로 — 공유된 링크가 죽으면 안 된다 */}
        <Route path="/results" element={<LegacyRunRedirect feature="results" />} />
        <Route path="/visualize" element={<LegacyRunRedirect feature="visualize" />} />
        {/* 모니터는 run 을 담고 있지 않았다. 고를 수 있는 곳으로 보낸다 */}
        <Route path="/monitor" element={<Navigate to="/" replace />} />

        <Route
          path="*"
          element={
            <BareScreen>
              <NotFoundScreen />
            </BareScreen>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}
