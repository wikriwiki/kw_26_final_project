/**
 * 대상자 문답 — 라우트 `/runs/:runId/agents`.
 *
 * **단일 목적: 대상자 한 명이 그날 무엇을 했고 왜 그랬는지 확인한다.**
 * 왼쪽에서 한 명을 찾고, 오른쪽에서 그 사람의 기록에 대해 묻는다.
 *
 * ## 지어내지 않는다
 *
 * LLM 이 연결돼 있지 않다. 그래서 이 화면은 **문장을 만들지 않는다.**
 * 미리 정해 둔 질문마다 답이 나오는 기록이 정해져 있고, 화면은 그 기록을 옮겨 적은 뒤
 * 어느 파일의 어느 항목인지 함께 보인다. 자유 입력은 그 질문들로 이어 주기만 하고,
 * 이어지지 않으면 "답할 기록이 없습니다"라고 말한다. 답을 만드는 일은 `lib/agentData.ts` 가 한다.
 *
 * ## 1,825명을 다 그리지 않는다
 *
 * 목록은 가상 스크롤이다. 보이는 만큼만(약 10줄) DOM 에 올린다 (SKILL §3 `virtualize-lists`).
 * 기록도 마찬가지다 — 목록용 요약(341KB)만 먼저 받고, 한 사람의 활동·기억(25KB)은
 * 고를 때 받는다. 원본 110MB 를 통째로 들이지 않는다.
 *
 * ## 스타일
 *
 * `src/styles/` 는 다른 조각의 소유라 이 화면 전용 규칙은 아래 `SCREEN_CSS` 에 함께 둔다.
 * 토큰·컴포넌트 클래스는 전부 기존 것을 그대로 쓴다 — 새 색도, 새 라디우스도 만들지 않는다.
 * (정리할 때 이 블록을 `styles/screens.css` 로 옮기면 된다.)
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { SelectField, TextField } from '../components/Field';
import { Callout, EmptyState, ErrorState, SkeletonText } from '../components/Feedback';
import { BarList } from '../components/Meter';
import { int } from '../lib/format';
import { askAgent } from '../lib/interview';
import {
  EMPTY_FILTER,
  QUESTIONS,
  SEX_LABEL,
  answer,
  facets,
  filterRoster,
  hourBand,
  loadAgent,
  loadRoster,
  matchQuestion,
  rosterLine,
} from '../lib/agentData';
import type {
  AgentDetail,
  Answer,
  AnswerBlock,
  Question,
  Roster,
  RosterFilter,
  RosterItem,
} from '../lib/agentData';

/* --- 이 화면 전용 규칙 ----------------------------------------------------- */

const SCREEN_CSS = `
.ag-side { align-self: start; }
@media (min-width: 1024px) {
  .ag-side { position: sticky; top: var(--sp-4); }
}

/* 목록은 세로로만 흐른다.
   세로만 auto 로 두면 가로가 visible → auto 로 계산돼 가로 스크롤 컨테이너가 된다.
   그래서 가로를 hidden 으로 못 박는다 (styles/components.css 의 .code 와 같은 어법) */
.ag-list {
  flex: 1 1 auto;
  min-height: 288px;
  max-height: 60vh;
  overflow: hidden auto;
  border-top: var(--hairline);
  border-bottom: var(--hairline);
  margin-inline: calc(-1 * var(--sp-4));
  padding-inline: var(--sp-4);
}
.ag-list:focus-visible { outline-offset: -2px; }

.ag-vp { position: relative; width: 100%; }

.ag-row {
  position: absolute;
  left: 0;
  right: 0;
  height: 64px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 2px;
  padding: var(--sp-2) var(--sp-3);
  border-left: 3px solid transparent;
  border-radius: var(--radius);
  cursor: pointer;
}
.ag-row:hover { background: var(--surface-sunken); }
/* 고른 줄은 배경면 대신 좌측 3px 인디케이터로 — 사이드바·표와 같은 어법 (§7b).
   면을 깔면 그 위의 12px 보조 글자 대비가 7:1 아래로 내려간다 */
.ag-row[aria-selected="true"] { border-left-color: var(--primary); }
.ag-row[aria-selected="true"] .ag-row__id { font-weight: var(--fw-semibold); }
.ag-row__id {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: var(--fs-sm);
  color: var(--fg);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ag-row__meta {
  font-size: var(--fs-caption);
  color: var(--fg-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 문답 한 묶음 — 면이나 말풍선 대신 왼쪽 헤어라인과 정렬로 묶는다 (§7b) */
.ag-turn { display: flex; flex-direction: column; gap: var(--sp-2); }
.ag-turn + .ag-turn { border-top: var(--hairline); padding-top: var(--sp-5); }
.ag-turn__q {
  font-size: var(--fs-body);
  font-weight: var(--fw-semibold);
  color: var(--fg);
}
.ag-turn__when { font-size: var(--fs-caption); color: var(--fg-muted); }
.ag-turn__a {
  display: flex;
  flex-direction: column;
  gap: var(--sp-3);
  border-left: 2px solid var(--border-strong);
  padding-left: var(--sp-3);
}
.ag-turn__src {
  font-size: var(--fs-caption);
  color: var(--fg-muted);
  font-family: var(--font-mono);
  line-height: var(--lh-body);
}

.ag-log { display: flex; flex-direction: column; gap: var(--sp-3); }
.ag-log__it {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 2px;
}
@media (min-width: 768px) {
  .ag-log__it { grid-template-columns: 64px minmax(0, 1fr); gap: 2px var(--sp-3); }
  .ag-log__lead { grid-row: 1 / span 3; text-align: right; }
}
.ag-log__lead {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: var(--fs-sm);
  color: var(--fg-muted);
}
.ag-log__title { font-size: var(--fs-md); font-weight: var(--fw-medium); color: var(--fg); }
.ag-log__meta { font-size: var(--fs-caption); color: var(--fg-muted); }
.ag-log__body {
  font-size: var(--fs-md);
  color: var(--fg-muted);
  line-height: var(--lh-body);
  white-space: pre-line;
}

.ag-ask { display: flex; flex-direction: column; gap: var(--sp-3); }
.ag-ask__cap {
  font-size: var(--fs-caption);
  font-weight: var(--fw-medium);
  color: var(--fg-muted);
  letter-spacing: 0.02em;
}
.ag-free { display: flex; flex-wrap: wrap; align-items: flex-end; gap: var(--sp-2); }
.ag-free > .field { flex: 1 1 220px; min-width: 0; }
`;

/* --- 가상 스크롤 ----------------------------------------------------------- */

const ROW_H = 64;
/** 화면 밖으로 조금 더 그려 둔다 — 빠르게 굴릴 때 빈칸이 보이지 않도록 */
const OVERSCAN = 4;

/**
 * 시간대. 분 단위가 아니라 "15시대"인 것은 3D 지도의 타임라인 프레임과 같은 기준이기
 * 때문이다. 15:30 의 활동은 15시대에 속한다 — 라벨을 "15:00"으로 적으면 그 사실이 어긋난다.
 */
const HOURS = Array.from({ length: 24 }, (_, h) => ({
  value: String(h),
  label: `${String(h).padStart(2, '0')}시대`,
}));

const DECILES = Array.from({ length: 10 }, (_, i) => ({
  value: String(i + 1),
  label: `${i + 1}분위`,
}));

interface Turn {
  key: number;
  question: string;
  when: string;
  /** 기록에서 바로 옮겨 적은 답. 본인 답변일 때는 없다 */
  answer?: Answer;
  /** 본인이 직접 한 말 (대화 모델) */
  said?: string;
  pending?: boolean;
  failed?: string;
}

export function AgentScreen() {

  const [roster, setRoster] = useState<Roster | null>(null);
  const [rosterError, setRosterError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    loadRoster().then(
      (r) => alive && setRoster(r),
      (e: Error) => alive && setRosterError(e.message),
    );
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="stack">
      <style>{SCREEN_CSS}</style>

      <header className="pagehead">
        <div className="pagehead__text">
          <h1 className="pagehead__title">대상자 문답</h1>
          <p className="pagehead__purpose">
            대상자 한 명을 골라, 그 사람이 그날 어디에 갔고 왜 그랬는지 기록에서 확인합니다.
          </p>
        </div>
      </header>

      {/*
        이 화면에서 가장 중요한 한 문단. 사람이 답하는 것처럼 보이는 화면이라
        "무엇이 아닌지"를 먼저 말하지 않으면 읽는 사람이 오해한다.
      */}
      <Callout>
        아래 <strong>질문 버튼</strong>은 저장된 기록을 그대로 옮겨 적습니다 — 숫자가 어디서
        나왔는지 함께 보입니다. 목록에 없는 것을 직접 물으면 <strong>대상자 본인이 자기 기록만
        보고</strong> 답합니다. 기록에 없는 일은 “기억에 없다”고 말합니다.
      </Callout>

      {rosterError ? (
        <ErrorState
          title="대상자 목록을 불러오지 못했습니다"
          body="화면이 읽는 파일이 아직 만들어지지 않았습니다. 아래 명령으로 만든 뒤 새로고침하세요."
          detail={rosterError}
        />
      ) : roster ? (
        <Workspace roster={roster} />
      ) : (
        <Card title="대상자 목록을 불러오는 중">
          <SkeletonText lines={4} />
        </Card>
      )}
    </div>
  );
}

/* ==========================================================================
   작업 영역 — 왼쪽 찾기 / 오른쪽 문답
   ========================================================================== */

function Workspace({ roster }: { roster: Roster }) {
  const [params, setParams] = useSearchParams();

  const [filter, setFilter] = useState<RosterFilter>(EMPTY_FILTER);
  const [dayIdx, setDayIdx] = useState(0);
  const [hour, setHour] = useState(15);

  const days = roster.meta.days;
  const opts = useMemo(() => facets(roster.items), [roster.items]);
  const filtered = useMemo(() => filterRoster(roster.items, filter), [roster.items, filter]);

  /** 고른 대상자는 주소가 들고 있다 — 링크를 그대로 공유할 수 있다 (스펙 §9 deep-linking) */
  const raw = params.get('agent');
  const picked = raw !== null && /^\d+$/.test(raw) ? Number(raw) : null;
  const selected =
    picked !== null && picked >= 0 && picked < roster.items.length ? picked : null;

  const select = useCallback(
    (idx: number) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set('agent', String(idx));
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  const set = <K extends keyof RosterFilter>(key: K, value: RosterFilter[K]) =>
    setFilter((f) => ({ ...f, [key]: value }));

  return (
    <div className="grid">
      <Card className="c4 ag-side" title="대상자 찾기">
        <p className="card__note">
          {int(filtered.length)}명 / 전체 {int(roster.items.length)}명
        </p>

        <TextField
          label="검색"
          value={filter.q}
          onChange={(e) => set('q', e.currentTarget.value)}
          placeholder="AGT_11110515 · 무악동 · 사무직"
          help="식별자 · 자치구 · 행정동 · 직업으로 찾습니다."
          autoComplete="off"
        />

        {/* 입력 컴포넌트는 className 을 받지 않는다. 열 배치는 감싸는 칸이 맡는다 */}
        <div className="grid">
          <div className="c6">
            <SelectField
              label="자치구"
              value={filter.gu}
              onChange={(e) => set('gu', e.currentTarget.value)}
              options={[
                { value: '', label: '전체' },
                ...opts.gu.map((v) => ({ value: v, label: v })),
              ]}
            />
          </div>
          <div className="c6">
            <SelectField
              label="연령"
              value={filter.age}
              onChange={(e) => set('age', e.currentTarget.value)}
              options={[
                { value: '', label: '전체' },
                ...opts.age.map((v) => ({ value: v, label: v })),
              ]}
            />
          </div>
          <div className="c6">
            <SelectField
              label="성별"
              value={filter.sex}
              onChange={(e) => set('sex', e.currentTarget.value)}
              options={[
                { value: '', label: '전체' },
                { value: 'F', label: '여성' },
                { value: 'M', label: '남성' },
              ]}
            />
          </div>
          <div className="c6">
            <SelectField
              label="소비 분위"
              value={filter.dec}
              onChange={(e) => set('dec', e.currentTarget.value)}
              options={[{ value: '', label: '전체' }, ...DECILES]}
              help="평일 하루 소비 예산 순위"
            />
          </div>
        </div>

        {filtered.length === 0 ? (
          <EmptyState
            title="조건에 맞는 대상자가 없습니다"
            body="자치구·연령·성별·소비 분위 중 하나를 넓히거나 검색어를 지우면 다시 나타납니다."
            action={<Button onClick={() => setFilter(EMPTY_FILTER)}>조건 모두 지우기</Button>}
          />
        ) : (
          <RosterList items={filtered} selected={selected} onSelect={select} />
        )}

        <p className="card__note">
          소비 분위는 원본 기록에 없는 값입니다. 이 화면이 평일 하루 소비 예산(daily_wd) 순위로
          10등분해 붙였습니다.
        </p>
      </Card>

      <div className="c8">
        {selected === null ? (
          <Card>
            <EmptyState
              title="대상자를 고르세요"
              body={`왼쪽 목록에서 한 명을 고르면 그 사람의 활동·기억·상태 기록을 여기에서 물어볼 수 있습니다. 지금 고를 수 있는 사람은 ${int(
                filtered.length,
              )}명입니다.`}
            />
          </Card>
        ) : (
          <Conversation
            key={selected}
            idx={selected}
            days={days}
            dayIdx={dayIdx}
            setDayIdx={setDayIdx}
            hour={hour}
            setHour={setHour}
          />
        )}
      </div>
    </div>
  );
}

/* ==========================================================================
   목록 — 가상 스크롤 + 키보드 이동
   ========================================================================== */

function RosterList({
  items,
  selected,
  onSelect,
}: {
  items: RosterItem[];
  selected: number | null;
  onSelect: (idx: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [top, setTop] = useState(0);
  const [viewH, setViewH] = useState(320);

  // 조건이 바뀌면 목록의 맨 위로 돌아간다 — 안 그러면 사라진 줄 자리에 멈춰 있다
  useEffect(() => {
    setTop(0);
    if (ref.current) ref.current.scrollTop = 0;
  }, [items]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewH(el.clientHeight));
    ro.observe(el);
    setViewH(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  const start = Math.max(0, Math.floor(top / ROW_H) - OVERSCAN);
  const end = Math.min(items.length, Math.ceil((top + viewH) / ROW_H) + OVERSCAN);
  const visible = items.slice(start, end);

  const activePos = selected === null ? -1 : items.findIndex((it) => it.i === selected);

  /** 목록 안에서 화살표로 옮긴다. 옮긴 줄이 곧 고른 줄이다 (단일 선택 listbox) */
  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const keys = ['ArrowDown', 'ArrowUp', 'Home', 'End', 'PageDown', 'PageUp'];
    if (!keys.includes(e.key)) return;
    e.preventDefault();
    const page = Math.max(1, Math.floor(viewH / ROW_H) - 1);
    const from = activePos < 0 ? -1 : activePos;
    let next = from;
    if (e.key === 'ArrowDown') next = from + 1;
    else if (e.key === 'ArrowUp') next = from - 1;
    else if (e.key === 'PageDown') next = from + page;
    else if (e.key === 'PageUp') next = from - page;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = items.length - 1;
    next = Math.min(items.length - 1, Math.max(0, next));
    onSelect(items[next].i);

    const el = ref.current;
    if (!el) return;
    const rowTop = next * ROW_H;
    if (rowTop < el.scrollTop) el.scrollTop = rowTop;
    else if (rowTop + ROW_H > el.scrollTop + el.clientHeight) {
      el.scrollTop = rowTop + ROW_H - el.clientHeight;
    }
  };

  return (
    <div
      ref={ref}
      className="ag-list"
      role="listbox"
      tabIndex={0}
      aria-label="대상자 목록"
      aria-activedescendant={selected === null ? undefined : `ag-opt-${selected}`}
      onScroll={(e) => setTop(e.currentTarget.scrollTop)}
      onKeyDown={onKeyDown}
    >
      {/*
        전체 높이만 붙잡아 두는 껍데기. `role="presentation"` 을 주지 않으면
        option 과 listbox 사이에 의미 있는 요소가 끼어 접근성 트리가 끊긴다
      */}
      <div className="ag-vp" role="presentation" style={{ height: items.length * ROW_H }}>
        {visible.map((it, i) => (
          <div
            key={it.i}
            id={`ag-opt-${it.i}`}
            className="ag-row"
            role="option"
            aria-selected={it.i === selected}
            aria-setsize={items.length}
            aria-posinset={start + i + 1}
            style={{ top: (start + i) * ROW_H }}
            onClick={() => onSelect(it.i)}
          >
            <span className="ag-row__id">{it.id}</span>
            <span className="ag-row__meta">{rosterLine(it)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ==========================================================================
   문답
   ========================================================================== */

function Conversation({
  idx,
  days,
  dayIdx,
  setDayIdx,
  hour,
  setHour,
}: {
  idx: number;
  days: string[];
  dayIdx: number;
  setDayIdx: (v: number) => void;
  hour: number;
  setHour: (v: number) => void;
}) {
  const [detail, setDetail] = useState<AgentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState('');
  const [miss, setMiss] = useState(false);
  const seq = useRef(0);

  useEffect(() => {
    let alive = true;
    setDetail(null);
    setError(null);
    loadAgent(idx).then(
      (d) => {
        if (!alive) return;
        setDetail(d);
        // 고르자마자 빈 화면을 두지 않는다. 첫 답도 기록에서 그대로 가져온 것이다
        const q = QUESTIONS.find((x) => x.id === 'who')!;
        seq.current += 1;
        setTurns([
          {
            key: seq.current,
            question: q.text,
            when: '기간 전체',
            answer: answer(d, q, { dayIdx: 0, hour: 15 }),
          },
        ]);
      },
      (e: Error) => alive && setError(e.message),
    );
    return () => {
      alive = false;
    };
  }, [idx]);

  const ask = useCallback(
    (q: Question) => {
      if (!detail) return;
      seq.current += 1;
      const when =
        q.group === 'life'
          ? '기간 전체'
          : q.id === 'where'
            ? `${days[dayIdx]} ${hourBand(hour)}`
            : days[dayIdx];
      const turn: Turn = {
        key: seq.current,
        question: q.text,
        when,
        answer: answer(detail, q, { dayIdx, hour }),
      };
      // 최근 질문이 위에 온다. 20묶음까지만 남긴다
      setTurns((prev) => [turn, ...prev].slice(0, 20));
      setMiss(false);
    },
    [detail, dayIdx, hour, days],
  );

  /**
   * 자유롭게 물었을 때.
   *
   * 기록에서 곧바로 답이 나오는 물음은 위 목록이 즉시·정확하게 답한다. 그 밖의 물음은
   * **본인에게 넘긴다** — 대화 모델이 이 사람의 기록만 근거로 대신 말한다.
   * 예전에는 여기서 "답할 기록이 없습니다"로 끝났는데, 그건 물어본 사람 잘못이 아니다.
   */
  const askFreely = useCallback(
    async (text: string) => {
      if (!detail) return;
      seq.current += 1;
      const key = seq.current;
      setTurns((prev) => [{ key, question: text, when: '본인 답변', pending: true }, ...prev].slice(0, 20));
      try {
        const history = turns
          .filter((t) => t.said)
          .slice(0, 3)
          .flatMap((t) => [
            { role: 'user' as const, content: t.question },
            { role: 'assistant' as const, content: t.said as string },
          ])
          .reverse();
        const reply = await askAgent(detail, text, history);
        setTurns((prev) =>
          prev.map((t) => (t.key === key ? { ...t, pending: false, said: reply.answer } : t)),
        );
      } catch (err) {
        setTurns((prev) =>
          prev.map((t) =>
            t.key === key
              ? { ...t, pending: false, failed: (err as Error).message }
              : t,
          ),
        );
      }
    },
    [detail, turns],
  );

  const submit = useCallback(() => {
    const text = draft.trim();
    if (!text || !detail) return;
    const q = matchQuestion(text);
    if (q) {
      ask(q);
    } else {
      void askFreely(text);
    }
    setDraft('');
  }, [draft, detail, ask, askFreely]);

  if (error) {
    return (
      <ErrorState
        title="이 대상자의 기록을 불러오지 못했습니다"
        body="기록 파일을 찾을 수 없습니다. 목록은 있는데 상세가 없다면 잘라 둔 파일이 목록과 어긋난 것입니다 — 생성 스크립트를 다시 실행하세요."
        detail={error}
      />
    );
  }

  if (!detail) {
    return (
      <Card title="기록을 불러오는 중">
        <SkeletonText lines={5} />
      </Card>
    );
  }

  const p = detail.profile;
  const dayQs = QUESTIONS.filter((q) => q.group === 'day');
  const lifeQs = QUESTIONS.filter((q) => q.group === 'life');

  return (
    <div className="stack">
      {/* 누구에게 묻는 중인지 */}
      <Card>
        <div className="row-between">
          <div className="stack-sm">
            <h2 className="section__title num">{detail.id}</h2>
            <p className="card__note">
              {p.district} {p.home_dong} · {p.age} {SEX_LABEL[p.gender] ?? p.gender} · {p.job} ·{' '}
              {p.life_stage}
            </p>
          </div>
          <p className="card__note">
            활동 {int(detail.events.length)}건 · 기억 {int(detail.visited.length)}건 · 아는 곳{' '}
            {int(detail.knows.length)}곳
          </p>
        </div>
      </Card>

      {/* 언제를 기준으로 묻는가 */}
      <Card title="기준 시점">
        <div className="segment" role="group" aria-label="날짜 선택">
          {days.map((d, i) => (
            <button
              key={d}
              type="button"
              className="segment__btn num"
              aria-pressed={i === dayIdx}
              onClick={() => setDayIdx(i)}
            >
              {d}
            </button>
          ))}
        </div>
        <div className="grid">
          <div className="c4">
            <SelectField
              label="기준 시간대"
              value={String(hour)}
              onChange={(e) => setHour(Number(e.currentTarget.value))}
              options={HOURS}
              help="“지금 어디에 있나요?”가 이 시간대를 기준으로 답합니다. 3D 지도의 타임라인 프레임과 같은 기준입니다."
            />
          </div>
        </div>
      </Card>

      {/* 물어보기 */}
      <Card title="물어보기">
        <div className="ag-ask">
          <p className="ag-ask__cap">{days[dayIdx]} 에 대해</p>
          <div className="segment" role="group" aria-label={`${days[dayIdx]} 에 대한 질문`}>
            {dayQs.map((q) => (
              <button key={q.id} type="button" className="segment__btn" onClick={() => ask(q)}>
                {q.text}
              </button>
            ))}
          </div>

          <p className="ag-ask__cap">기간 전체에 대해</p>
          <div className="segment" role="group" aria-label="기간 전체에 대한 질문">
            {lifeQs.map((q) => (
              <button key={q.id} type="button" className="segment__btn" onClick={() => ask(q)}>
                {q.text}
              </button>
            ))}
          </div>

          <div className="ag-free">
            <TextField
              label="직접 묻기"
              value={draft}
              onChange={(e) => {
                setDraft(e.currentTarget.value);
                setMiss(false);
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder="이 날 왜 거기에 갔나요?"
              help="목록에 없는 것을 물으면 본인이 자기 기록을 보고 직접 답합니다."
              error={
                miss
                  ? '이 질문에 답할 기록이 없습니다 — 위 목록에 있는 질문으로 다시 물어보세요.'
                  : undefined
              }
              autoComplete="off"
            />
            <Button variant="primary" onClick={submit} disabled={draft.trim().length === 0}>
              묻기
            </Button>
          </div>
        </div>
      </Card>

      {/* 문답 기록 */}
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">문답</h2>
          <p className="section__note">최근에 물어본 것이 위에 있습니다.</p>
        </div>
        <Card>
          {turns.length === 0 ? (
            <EmptyState
              title="아직 물어본 것이 없습니다"
              body="위 질문 중 하나를 누르면 그 답이 나오는 기록을 여기에 옮겨 적습니다."
            />
          ) : (
            <div className="stack">
              {turns.map((t) => (
                <TurnView key={t.key} turn={t} />
              ))}
            </div>
          )}
        </Card>
      </section>
    </div>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  const a = turn.answer;
  if (!a) {
    /* 본인 답변 — 기록을 옮겨 적는 대신 본인이 말한다 */
    return (
      <article className="ag-turn">
        <div>
          <p className="ag-turn__q wrap">{turn.question}</p>
          <p className="ag-turn__when num">{turn.when}</p>
        </div>
        <div className="ag-turn__a">
          {turn.pending ? (
            <p className="wrap" style={{ color: 'var(--fg-muted)' }} role="status">
              답을 기다리는 중입니다…
            </p>
          ) : turn.failed ? (
            <p className="wrap" style={{ color: 'var(--danger)' }}>{turn.failed}</p>
          ) : (
            <p className="wrap">{turn.said}</p>
          )}
          {!turn.pending && !turn.failed ? (
            <p className="ag-turn__src wrap">본인이 자기 기록을 보고 답했습니다</p>
          ) : null}
        </div>
      </article>
    );
  }
  return (
    <article className="ag-turn">
      <div>
        <p className="ag-turn__q wrap">{turn.question}</p>
        <p className="ag-turn__when num">{turn.when}</p>
      </div>
      <div className="ag-turn__a">
        {a.empty ? (
          <p className="wrap" style={{ color: 'var(--fg-muted)' }}>
            {a.empty}
          </p>
        ) : (
          a.blocks.map((b, i) => <BlockView key={i} block={b} />)
        )}
        <p className="ag-turn__src wrap">근거 · {a.source}</p>
      </div>
    </article>
  );
}

function BlockView({ block }: { block: AnswerBlock }) {
  if (block.kind === 'p') {
    return <p className="wrap">{block.text}</p>;
  }

  if (block.kind === 'facts') {
    return (
      <dl className="dl">
        {block.items.map((it) => (
          <div className="dl__item" key={it.k}>
            <dt className="dl__k">{it.k}</dt>
            <dd className="dl__v">{it.v}</dd>
          </div>
        ))}
      </dl>
    );
  }

  if (block.kind === 'bars') {
    return (
      <BarList
        items={block.items.map((it) => ({
          key: it.name,
          name: it.name,
          value: it.value,
          display: it.display,
        }))}
      />
    );
  }

  return (
    <div className="ag-log">
      {block.items.map((it, i) => (
        <div className="ag-log__it" key={`${it.lead}-${i}`}>
          <span className="ag-log__lead">{it.lead}</span>
          <span className="ag-log__title wrap">{it.title}</span>
          {it.meta ? <span className="ag-log__meta wrap">{it.meta}</span> : null}
          {it.body ? <span className="ag-log__body wrap">{it.body}</span> : null}
        </div>
      ))}
    </div>
  );
}
