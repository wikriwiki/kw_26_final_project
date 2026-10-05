/**
 * run 셸 — `docs/DESIGN_IA_RUN_FIRST.md` §4, 스펙 §6.
 *
 * 레일(56px)은 항상 남고, 드로어(240px)가 transform 으로 그 위를 덮으며 확장된다.
 * 두 겹의 아이콘 중심 x 가 같아서 시각적으로는 "레일이 늘어난 것"으로 보인다.
 * 열려 있는 쪽만 탭 순서에 들어가도록 tabIndex 를 교대로 준다.
 *
 * 사이드바가 이제 세 층이다. 위에서부터
 *   1. ← 모든 시뮬레이션   목록으로 돌아가기
 *   2. 현재 실행 + 전환    어느 화면에 있든 "무엇을 보는 중인지" 보인다
 *   3. 기능 내비          그 실행의 개요·모니터·결과·시각화·정책
 * 레일의 슬롯 순서도 같다 — 접었다 펴도 항목이 자리를 옮기지 않는다.
 */
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { Link, NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import {
  ActivityIcon,
  ArrowLeftIcon,
  BarChartIcon,
  ChevronDownIcon,
  InboxIcon,
  LayersIcon,
  MapIcon,
  MenuIcon,
  SlidersIcon,
  UsersIcon,
  XIcon,
} from '../components/Icon';
import { PHASE_LABEL, useRun, useRunSummaries } from './RunContext';
import type { FeatureKey, RunSummary } from './RunContext';
import { int } from '../lib/format';
import { useNav } from './useNav';

const FEATURE_ICON: Record<FeatureKey, typeof SlidersIcon> = {
  overview: LayersIcon,
  monitor: ActivityIcon,
  results: BarChartIcon,
  visualize: MapIcon,
  agents: UsersIcon,
  report: InboxIcon,
  policy: SlidersIcon,
};

/** 지도가 주인공인 화면 — 여백 없이 전체 폭을 쓰고, 사이드바는 레일로 접어 둔다 (스펙 §7) */
const BLEED_SEGMENT = 'visualize';

/** 실행을 한 줄로: "완료 · 7일 · 200명". 모르는 값은 적지 않는다 */
export function runMetaLine(run: RunSummary): string {
  const parts = [PHASE_LABEL[run.phase], `${int(run.daysPresent)}일`];
  if (run.agentsTarget !== null) parts.push(`${int(run.agentsTarget)}명`);
  return parts.join(' · ');
}

/** 주소에서 지금 보고 있는 기능 조각을 뽑는다. `/runs/BASE/results` → `results` */
function featureSegment(pathname: string): string {
  const m = /^\/runs\/[^/]+\/([^/]+)/.exec(pathname);
  return m?.[1] ?? '';
}

export function AppShell() {
  const { expanded, push, toggle, close, suppress } = useNav();
  const location = useLocation();
  const navigate = useNavigate();
  const run = useRun();
  const allRuns = useRunSummaries();

  const openerRef = useRef<HTMLButtonElement>(null);
  const closerRef = useRef<HTMLButtonElement>(null);
  const switchRef = useRef<HTMLButtonElement>(null);
  const lastKey = useRef(location.key);
  const wasExpanded = useRef(expanded);

  const segment = featureSegment(location.pathname);
  const bleed = segment === BLEED_SEGMENT;
  const menuId = useId();
  const [menuOpen, setMenuOpen] = useState(false);

  // 시각화 화면은 레일 상태로 진입한다. 저장된 선호값은 건드리지 않아 나가면 되돌아온다
  useEffect(() => {
    suppress(bleed);
  }, [bleed, suppress]);

  /** 닫을 때는 aria-hidden 이 걸리기 전에 포커스를 레일로 먼저 빼낸다 */
  const closeAndRestoreFocus = useCallback(() => {
    openerRef.current?.focus();
    setMenuOpen(false);
    close();
  }, [close]);

  const onToggle = useCallback(() => {
    if (expanded) closeAndRestoreFocus();
    else toggle();
  }, [expanded, closeAndRestoreFocus, toggle]);

  // 열렸을 때만 드로어 안으로 포커스를 옮긴다 (닫힘은 위에서 미리 처리)
  useEffect(() => {
    const opened = expanded && !wasExpanded.current;
    wasExpanded.current = expanded;
    if (opened) closerRef.current?.focus();
  }, [expanded]);

  // overlay 모드에서 "화면을 옮기면" 드로어를 닫는다.
  // 주소가 실제로 바뀐 경우만 — 첫 렌더와 StrictMode 재실행에서는 저장된 상태를 존중한다
  useEffect(() => {
    if (lastKey.current === location.key) return;
    lastKey.current = location.key;
    setMenuOpen(false);
    if (!push) close();
  }, [location.key, push, close]);

  // 전환 목록은 Esc 와 바깥 클릭으로 닫는다. 포커스는 열었던 버튼으로 되돌린다
  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      e.stopPropagation();
      switchRef.current?.focus();
      setMenuOpen(false);
    };
    const onDown = (e: MouseEvent) => {
      const el = e.target as Node;
      if (switchRef.current?.contains(el)) return;
      if (document.getElementById(menuId)?.contains(el)) return;
      setMenuOpen(false);
    };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('mousedown', onDown);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('mousedown', onDown);
    };
  }, [menuOpen, menuId]);

  /**
   * 실행을 바꿔도 **보던 기능 화면을 유지한다** (설계도 §4, 스펙 §9 state-preservation).
   * 다만 그 실행에 없는 기능이면 개요로 내려앉는다 — 빈 화면을 그리지 않는다.
   * 일자(`?day=`)는 실행마다 다르므로 들고 가지 않는다.
   */
  const switchTo = useCallback(
    (next: RunSummary) => {
      setMenuOpen(false);
      const target = next.features.find((f) => f.segment === segment && f.segment !== '');
      const keep = target?.available ? target.segment : '';
      navigate(keep ? `/runs/${next.id}/${keep}` : `/runs/${next.id}`);
    },
    [navigate, segment],
  );

  /** 레일에서 실행 표시를 누르면 드로어를 펴면서 전환 목록까지 연다 */
  const openSwitcher = useCallback(() => {
    toggle();
    setMenuOpen(true);
  }, [toggle]);

  const railTab = expanded ? -1 : 0;
  const drawerTab = expanded ? 0 : -1;
  const initial = run.id.slice(0, 1);

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        본문으로 건너뛰기
      </a>

      {/* 상주 레일 — 완전 숨김이 아니라 항상 도달 가능하다 */}
      <nav className="rail" aria-label="주요 메뉴">
        <button
          ref={openerRef}
          type="button"
          className="rail__btn"
          onClick={onToggle}
          aria-label={expanded ? '메뉴 접기' : '메뉴 펼치기'}
          aria-expanded={expanded}
          aria-controls="nav-drawer"
        >
          <MenuIcon size={20} />
        </button>

        <Link to="/" className="rail__btn" aria-label="모든 시뮬레이션" title="모든 시뮬레이션" tabIndex={railTab}>
          <ArrowLeftIcon size={20} />
        </Link>

        {/* 접힘 상태에서도 현재 실행이 보인다 (설계도 §7) */}
        <button
          type="button"
          className="rail__btn rail__run"
          onClick={openSwitcher}
          aria-label={`현재 실행 ${run.id} — 다른 실행으로 전환`}
          title={`${run.id} · ${runMetaLine(run)}`}
          tabIndex={railTab}
        >
          <span aria-hidden="true">{initial}</span>
        </button>

        <span className="rail__sep" />

        <ul className="rail__nav">
          {run.features.map(({ key, segment: seg, label, available, reason }) => {
            const Icon = FEATURE_ICON[key];
            if (!available) {
              return (
                <li key={key}>
                  <button
                    type="button"
                    className="rail__btn rail__btn--off"
                    aria-disabled="true"
                    aria-label={`${label} — ${reason}`}
                    title={`${label} — ${reason}`}
                    tabIndex={railTab}
                  >
                    <Icon size={20} />
                  </button>
                </li>
              );
            }
            return (
              <li key={key}>
                <NavLink
                  to={run.path(seg)}
                  end={seg === ''}
                  className="rail__btn"
                  aria-label={label}
                  title={label}
                  tabIndex={railTab}
                >
                  <Icon size={20} />
                </NavLink>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* 확장 드로어 */}
      <div className="drawer" id="nav-drawer" aria-hidden={!expanded}>
        <div className="drawer__head">
          <button
            ref={closerRef}
            type="button"
            className="drawer__close"
            onClick={closeAndRestoreFocus}
            aria-label="메뉴 접기"
            tabIndex={drawerTab}
          >
            <XIcon size={20} />
          </button>
          <span className="drawer__brand">정책 시뮬레이션 콘솔</span>
        </div>

        {/* 1층 — 목록으로 돌아가기. 항상 맨 위 */}
        <Link to="/" className="shellback" tabIndex={drawerTab}>
          <span className="shellback__icon">
            <ArrowLeftIcon size={20} />
          </span>
          <span>모든 시뮬레이션</span>
        </Link>

        {/* 2층 — 현재 실행 + 전환. 목록까지 가지 않고 바꿀 수 있다 */}
        <div className="runswitch">
          <button
            ref={switchRef}
            type="button"
            className="runswitch__btn"
            onClick={() => setMenuOpen((v) => !v)}
            aria-expanded={menuOpen}
            aria-controls={menuId}
            aria-haspopup="true"
            tabIndex={drawerTab}
          >
            <span className="runswitch__mark" aria-hidden="true">
              {initial}
            </span>
            <span className="runswitch__text">
              <span className="runswitch__id num">{run.id}</span>
              <span className="runswitch__meta">{runMetaLine(run)}</span>
            </span>
            <ChevronDownIcon size={18} className="runswitch__chev" />
          </button>

          {menuOpen ? (
            <div className="runmenu" id={menuId}>
              <p className="runmenu__cap">실행 전환</p>
              <ul>
                {allRuns.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className="runmenu__item"
                      aria-current={item.id === run.id ? 'true' : undefined}
                      onClick={() => switchTo(item)}
                    >
                      <span className="runmenu__id num">{item.id}</span>
                      <span className="runmenu__meta">{runMetaLine(item)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>

        <span className="shell__sep" />

        {/* 3층 — 기능 내비. 없는 기능은 숨기지 않고 비활성 + 이유 */}
        <nav aria-label="이 실행의 화면">
          <ul className="drawer__nav">
            {run.features.map(({ key, segment: seg, label, available, reason }) => {
              const Icon = FEATURE_ICON[key];
              if (!available) {
                return (
                  <li key={key}>
                    <button
                      type="button"
                      className="drawer__link drawer__link--off"
                      aria-disabled="true"
                      tabIndex={drawerTab}
                    >
                      <span className="drawer__icon">
                        <Icon size={20} />
                      </span>
                      <span className="drawer__label">
                        <span>{label}</span>
                        <span className="drawer__why">{reason}</span>
                      </span>
                    </button>
                  </li>
                );
              }
              return (
                <li key={key}>
                  <NavLink
                    to={run.path(seg)}
                    end={seg === ''}
                    className="drawer__link"
                    tabIndex={drawerTab}
                  >
                    <span className="drawer__icon">
                      <Icon size={20} />
                    </span>
                    <span className="drawer__label">{label}</span>
                  </NavLink>
                </li>
              );
            })}
          </ul>
        </nav>

        <p className="drawer__foot">서울 정책 시뮬레이션</p>
      </div>

      {/* 1024px 미만에서만 반응하는 스크림 */}
      <button
        type="button"
        className="scrim"
        onClick={closeAndRestoreFocus}
        tabIndex={-1}
        aria-hidden="true"
      />

      <div className="app__main">
        <main id="main" className={bleed ? 'page page--bleed' : 'page'}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
