/**
 * 시뮬레이션 목록 — `docs/DESIGN_IA_RUN_FIRST.md` §3.
 *
 * **단일 목적: 어떤 실행을 볼지 고른다.** 그 외의 것을 두지 않는다.
 * 실행이 명사고 기능이 동사다. 여기서 명사를 고르고 나면 셸이 동사를 맡는다.
 *
 * 셸(사이드바)이 없는 화면이다 — 아직 run 컨텍스트가 없으므로 고를 것도 없다.
 */
import { Link } from 'react-router-dom';
import { Badge } from '../components/Badge';
import { EmptyState } from '../components/Feedback';
import { Meter } from '../components/Meter';
import { PHASE_LABEL, useRunSummaries } from '../app/RunContext';
import type { RunPhase, RunSummary } from '../app/RunContext';
import { EMPTY, int, shortTime } from '../lib/format';

/** 진행중 → 완료 → 중단. 비어 있는 묶음은 그리지 않는다 */
const GROUP_ORDER: RunPhase[] = ['running', 'completed', 'stopped'];

/** "2025-07-21 ~ 07-27". 하루짜리면 그 하루만 */
function periodText(run: RunSummary): string {
  const { firstDay, lastDay } = run;
  if (!firstDay) return EMPTY;
  if (!lastDay || lastDay === firstDay) return firstDay;
  return `${firstDay} ~ ${lastDay.slice(5)}`;
}

/** 기간(일 수) · 대상자 · 적용 정책. 모르는 값은 "알 수 없음"이라 적고 지어내지 않는다 */
function metaText(run: RunSummary): string {
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
function policyText(run: RunSummary): string {
  if (!run.policy.known) return '정책 기록 없음';
  if (run.policy.items.length === 0) return '무정책';
  return run.policy.items.map((p) => p.name ?? p.id).join(', ');
}

/** 중단된 실행은 숨기지 않는다. 어디까지 있는지 적는다 (설계도 §3) */
function stoppedNote(run: RunSummary): string {
  const upto = run.lastDay ? `${run.lastDay} 까지의 자료가 남아 있습니다` : '남아 있는 일자 기록이 없습니다';
  return `완료 기록이 없습니다. ${upto}.`;
}

function RunRow({ run }: { run: RunSummary }) {
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
            <span className="runrow__policy">{policyText(run)}</span>
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

export function HomeScreen() {
  const runs = useRunSummaries();
  const groups = GROUP_ORDER.map((phase) => ({
    phase,
    items: runs.filter((r) => r.phase === phase),
  })).filter((g) => g.items.length > 0);

  return (
    <main id="main" className="home">
      <div className="stack">
        <header className="pagehead">
          <div className="pagehead__text">
            <h1 className="pagehead__title">시뮬레이션</h1>
            <p className="pagehead__purpose">확인할 시뮬레이션을 고르세요.</p>
          </div>
          <div className="pagehead__actions">
            {/* 이 화면의 유일한 primary */}
            <Link className="btn btn--primary" to="/new">
              새 시뮬레이션 만들기
            </Link>
          </div>
        </header>

        {groups.length === 0 ? (
          <EmptyState
            title="아직 실행한 시뮬레이션이 없습니다"
            body="정책을 고르고 기간과 대상자를 정하면 첫 시뮬레이션을 만들 수 있습니다."
            action={
              <Link className="btn btn--primary" to="/new">
                새 시뮬레이션 만들기
              </Link>
            }
          />
        ) : (
          groups.map(({ phase, items }) => (
            <section className="section" key={phase}>
              <div className="section__head">
                <h2 className="section__title">{PHASE_LABEL[phase]}</h2>
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
          body="지금 남아 있는 실행은 시뮬레이션 목록에서 확인할 수 있습니다."
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
