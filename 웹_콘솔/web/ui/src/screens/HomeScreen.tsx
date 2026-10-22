/**
 * 시뮬레이션 목록 — `docs/DESIGN_IA_RUN_FIRST.md` §3.
 *
 * **단일 목적: 어떤 실행을 볼지 고른다.** 그 외의 것을 두지 않는다.
 * 실행이 명사고 기능이 동사다. 여기서 명사를 고르고 나면 셸이 동사를 맡는다.
 *
 * 셸(사이드바)이 없는 화면이다 — 아직 run 컨텍스트가 없으므로 고를 것도 없다.
 */
import { Link } from 'react-router-dom';
import { CityDots, type Residents } from '../components/CityDots';
import { api, request } from '../lib/api';
import { useLoad } from '../lib/runData';
import { Badge } from '../components/Badge';
import { EmptyState } from '../components/Feedback';
import { Meter } from '../components/Meter';
import { PHASE_LABEL, useRunList, useRunPolicy } from '../app/RunContext';
import type { PolicyBinding, RunListItem, RunPhase } from '../app/RunContext';
import { EMPTY, int, shortTime } from '../lib/format';
import { READ_ONLY } from '../lib/runtime';
import { ExperimentList } from './experiment/ExperimentList';
import { PolicyList } from './PolicyList';

/** 진행중 → 완료 → 중단. 비어 있는 묶음은 그리지 않는다 */
const GROUP_ORDER: RunPhase[] = ['running', 'completed', 'stopped'];

/** "2025-07-21 ~ 07-27". 하루짜리면 그 하루만 */
function periodText(run: RunListItem): string {
  const { firstDay, lastDay } = run;
  if (!firstDay) return EMPTY;
  if (!lastDay || lastDay === firstDay) return firstDay;
  return `${firstDay} ~ ${lastDay.slice(5)}`;
}

/** 기간(일 수) · 대상자 · 적용 정책. 모르는 값은 "알 수 없음"이라 적고 지어내지 않는다 */
function metaText(run: RunListItem): string {
  const parts: string[] = [periodText(run)];

  if (run.daysPlanned === null) parts.push(`${int(run.daysPresent)}일 기록됨`);
  else if (run.daysPlanned === run.daysPresent) parts.push(`${int(run.daysPresent)}일`);
  else parts.push(`${int(run.daysPresent)}일 / 계획 ${int(run.daysPlanned)}일`);

  parts.push(run.agentsTarget === null ? '대상자 수 알 수 없음' : `${int(run.agentsTarget)}명`);
  return parts.join(' · ');
}

/**
 * 적용 정책. 세 경우를 섞지 않는다.
 * 기록이 없는 것과 정책을 쓰지 않은 것은 다른 사실이다.
 */
function policyText(policy: PolicyBinding): string {
  if (!policy.known) return '정책 기록 없음';
  if (policy.title) return policy.title;
  if (policy.items.length === 0) return '무정책';
  return policy.items.map((p) => p.name ?? p.id).join(', ');
}

/** 적용 정책은 실행마다 결제 요약을 받아야 알 수 있다. 줄은 먼저 그리고 이 칸만 뒤에 채운다 */
function RunPolicy({ id }: { id: string }) {
  const policy = useRunPolicy(id);
  if (policy.status === 'ready') return <span className="runrow__policy">{policyText(policy.data)}</span>;
  if (policy.status === 'error') return <span className="runrow__policy">정책 기록을 읽지 못함</span>;
  return <span className="runrow__policy runrow__policy--pending" aria-busy="true" />;
}

/** 중단된 실행은 숨기지 않는다. 어디까지 있는지 적는다 (설계도 §3) */
function stoppedNote(run: RunListItem): string {
  const upto = run.lastDay ? `${run.lastDay} 까지의 자료가 남아 있습니다` : '남아 있는 일자 기록이 없습니다';
  return `완료 기록이 없습니다. ${upto}.`;
}

function RunRow({ run }: { run: RunListItem }) {
  const showProgress = run.phase === 'running' && run.daysPlanned !== null && run.daysPlanned > 0;

  return (
    <li className="runlist__li">
      <Link className="runrow" to={`/runs/${run.id}`}>
        <span className="runrow__head">
          <span className="runrow__name num">{run.id}</span>
          {/* §7b — 정상 완료에는 뱃지를 달지 않는다. 이상할 때만 표시한다 */}
          {run.phase === 'stopped' ? <Badge tone="warn">중단됨</Badge> : null}
        </span>

        <span className="runrow__meta">
          <span className="runrow__line">
            <span className="runrow__facts">{metaText(run)}</span>
            <RunPolicy id={run.id} />
          </span>
          {run.phase === 'stopped' ? <span className="runrow__why">{stoppedNote(run)}</span> : null}
          {/* 분모를 모르면 막대를 그리지 않는다 — 0% 로 그리면 거짓말이 된다 */}
          {showProgress ? (
            <span className="runrow__progress">
              <Meter
                ratio={run.daysPresent / (run.daysPlanned as number)}
                label={`${run.id} 진행률`}
              />
              <span className="runrow__pct num">
                {int(run.daysPresent)} / {int(run.daysPlanned)}일
              </span>
            </span>
          ) : null}
          {run.phase === 'running' && !showProgress ? (
            <span className="runrow__why">계획 일 수를 몰라 진행률을 계산할 수 없습니다.</span>
          ) : null}
        </span>

        <span className="runrow__time num">{shortTime(run.updatedAt)}</span>
      </Link>
    </li>
  );
}

/** 첫 화면 지도의 시민 — 3D 지도 팩 가운데 첫 팩. 팩이 없으면 지도를 그리지 않는다 */
async function loadResidents(): Promise<(Residents & { label: string | null }) | null> {
  const packs = await api.listVizPacks();
  const first = packs.items.find((p) => !('error' in p)) ?? null;
  if (!first) return null;
  return request<Residents & { label: string | null }>(`/viz/packs/${encodeURIComponent(first.pack_id)}/residents`);
}

function HomeHero(_props: { runs: RunListItem[] }) {
  const residents = useLoad('home-residents', loadResidents);
  const map = residents.status === 'ready' ? residents.data : null;
  return (
    <section className={`hero${map ? '' : ' hero--solo'}`}>
      <div className="hero__text">
        <p className="kicker">서울 상권정책 시뮬레이션</p>
        <h1 className="hero__title">
          정책을 넣고,
          <br />
          시민의 하루를 본다.
        </h1>
        <p className="hero__lead">
          같은 시민에게 정책이 있을 때와 없을 때, 소비가 어떻게 다른지 비교합니다.
        </p>
        {!READ_ONLY ? (
          <div className="hero__actions">
            <Link className="btn btn--primary" to="/new">
              새 시뮬레이션 만들기
            </Link>
          </div>
        ) : null}
      </div>
      {map ? (
        <figure className="hero__map">
          <CityDots data={map} label={`시민 ${int(map.count)}명의 집 위치`} />
        </figure>
      ) : null}
    </section>
  );
}

export function HomeScreen() {
  const list = useRunList();
  const runs = list.status === 'ready' ? list.data : [];
  // 정책 본런에서 세운 실행은 위 '정책' 목록에 나온다. 여기에는 그 밖의 실행만
  const others = runs.filter((r) => !r.index.meta);
  const linked = runs.flatMap((r) => (r.index.meta ? [r.index.meta.experiment_id] : []));
  const groups = GROUP_ORDER.map((phase) => ({
    phase,
    items: others.filter((r) => r.phase === phase),
  })).filter((g) => g.items.length > 0);

  return (
    <main id="main" className="home">
      <div className="stack">
        <HomeHero runs={runs} />

        <PolicyList runs={runs} />

        {/* 실행으로 세우지 않은 실험(원격에서 받는 실험 등). API 가 없으면 그리지 않는다 */}
        <ExperimentList exclude={linked} />

        {list.status === 'loading' || list.status === 'idle' ? (
          <div className="screen-fallback" aria-busy="true" />
        ) : list.status === 'error' ? (
          <EmptyState
            title="시뮬레이션 목록을 불러오지 못했습니다"
            body={list.error.message}
            action={
              <button type="button" className="btn" onClick={list.retry}>
                다시 시도
              </button>
            }
          />
        ) : groups.length === 0 && runs.length === 0 ? (
          <EmptyState
            title="아직 실행한 시뮬레이션이 없습니다"
            body={
              READ_ONLY
                ? '조회할 수 있는 시뮬레이션 결과가 없습니다.'
                : '정책과 기간을 정해 첫 시뮬레이션을 만드세요.'
            }
            action={
              !READ_ONLY ? (
                <Link className="btn btn--primary" to="/new">
                  새 시뮬레이션 만들기
                </Link>
              ) : undefined
            }
          />
        ) : (
          groups.map(({ phase, items }) => (
            <section className="section" key={phase}>
              <div className="section__head">
                <h2 className="section__title">
                  <span className="kicker">실행</span>
                  {PHASE_LABEL[phase]}
                </h2>
                <span className="section__note num">{int(items.length)}건</span>
              </div>
              <ul className="runlist">
                {items.map((run) => (
                  <RunRow key={run.id} run={run} />
                ))}
              </ul>
            </section>
          ))
        )}
      </div>
    </main>
  );
}

/**
 * 주소에 없는 실행을 가리킬 때. 셸을 그리지 않는다 — 셸은 실행이 있어야 성립한다.
 * 공유받은 링크가 죽었을 때 사용자가 다음에 할 일을 적는다 (스펙 §8 empty-states).
 */
export function UnknownRunScreen({ runId }: { runId: string }) {
  return (
    <main id="main" className="home">
      <div className="stack">
        <header className="pagehead">
          <div className="pagehead__text">
            <h1 className="pagehead__title">그 실행을 찾지 못했습니다</h1>
            <p className="pagehead__purpose">
              주소가 가리키는 실행(<span className="num">{runId}</span>)이 목록에 없습니다. 이름이
              바뀌었거나 아직 불러오지 않았을 수 있습니다.
            </p>
          </div>
        </header>
        <EmptyState
          title="목록에서 다시 고르세요"
          body="이 실행은 더 이상 없습니다."
          action={
            <Link className="btn btn--primary" to="/">
              모든 시뮬레이션
            </Link>
          }
        />
      </div>
    </main>
  );
}
