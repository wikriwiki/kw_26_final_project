/**
 * 대상자 문답 화면(`/runs/:runId/agents`)이 쓰는 데이터 — 불러오기 · 추리기 · 답 만들기.
 *
 * ## 왜 이 파일이 따로 있나
 *
 * 원본은 `web/viz_store/demo/` 에 있고 합계 110MB 가 넘는다.
 *
 *     agents.json    1.5MB   1,825명 프로필
 *     memories.json  30.2MB  기억 48,474건
 *     events.json    44.2MB  활동 81,157건
 *     timeline.json  36.5MB  120프레임 시간대별 위치
 *
 * `import` 로 끌어오면 번들에 그대로 들어가 브라우저가 멈춘다 (스펙 §9 / GAUNTLET B5).
 * 그래서 `web/ui/tools/build_agent_console_data.py` 가 미리 두 갈래로 잘라
 * `public/agent-console/` 에 넣어 두고, 여기서는 **필요할 때만** 가져온다.
 *
 *     roster.json        341KB  ·  목록·필터·검색에 쓰는 11개 필드만 (한 번만 받는다)
 *     agents/<idx>.json   25KB  ·  고른 한 명의 활동·기억·상태 전부 (고를 때마다 한 번)
 *
 * `public/` 은 Vite 가 손대지 않고 그대로 내보내는 자리라 개발 서버와 빌드 산출물이
 * 같은 주소를 쓴다. `vite.config.ts` 를 고칠 일이 없다.
 *
 * timeline.json 은 쓰지 않는다. events 를 시간대별로 다시 담은 것이라서,
 * "그 시각에 어디 있었나"는 events 에서 그대로 나온다 (표본 200명 대조, 불일치 0건).
 *
 * ## 답을 지어내지 않는다
 *
 * LLM 이 연결돼 있지 않다. 이 파일은 **저장된 기록을 찾아 옮겨 적을 뿐**이고,
 * 문장을 새로 만들지 않는다. 답마다 `source` 로 어느 파일의 어느 기록인지 밝히고,
 * 기록이 없으면 `empty` 에 "없다"고 적는다. 0 이나 빈 값을 그럴듯하게 채우지 않는다.
 */

import { dec, int, krw } from './format';

/* ==========================================================================
   1. 계약 — 잘라 둔 파일의 모양
   ========================================================================== */

/** 목록 한 줄. 1,825명을 다 들고 있어야 하므로 필드를 최소로 줄였다 */
export interface RosterItem {
  /** `agents/<i>.json` 의 번호. 주소(`?agent=`)에도 이 값을 쓴다 */
  i: number;
  id: string;
  /** 자치구 */
  gu: string;
  /** 거주 행정동 */
  dong: string;
  age: string;
  /** 'F' | 'M' */
  sex: string;
  /** 소득 구간 (하·중하·중·중상·상) */
  inc: string;
  job: string;
  /** 소비 분위 1~10. **원본에 없는 값이다** — `decileBasis` 로 여기서 계산했다 */
  dec: number;
  /** 평일 하루 소비 예산(원) */
  wd: number;
  /** 활동 기록 건수 */
  ev: number;
  /** 5일 결제 합계(원) */
  spent: number;
}

export interface RosterMeta {
  source: string;
  generatedFrom: { agents: number; eventAgents: number; events: number; memoryAgents: number };
  days: string[];
  decileBasis: string;
}

export interface Roster {
  meta: RosterMeta;
  items: RosterItem[];
}

/** `agents.json` 원본 레코드 그대로 */
export interface AgentProfile {
  id: string;
  age: string;
  gender: string;
  income: string;
  life_stage: string;
  job: string;
  tendency: string;
  daily_wd: number;
  daily_we: number;
  top_wd: string;
  lifestyle: string;
  district: string;
  dist_code: string;
  home_dong: string;
  home_poi_name: string;
  work_poi_name: string;
  commute: number;
  has_appointment: boolean;
}

/** 활동 한 건 (`events.json` 의 한 레코드) */
export interface AgentEvent {
  /** 날짜 번호 — `AgentDetail.days` 의 색인 */
  d: number;
  /** "07:30" */
  t: string;
  cat: string;
  sub: string | null;
  l1: string | null;
  intent: string;
  sat: number;
  spent: number;
  /** 계기 (none·lifestyle·top_category·mood·rumor·policy·appointment) */
  trg: string;
  /** 그 활동을 한 이유 — 시뮬레이션이 남긴 문장 그대로 */
  why: string;
  /** 여러 후보 중 그 장소를 고른 이유 */
  pick: string | null;
  /** 고른 기준 (distance·known·satisfaction·novelty·random·rumor·appointment) */
  pf: string | null;
  /** 장소 이름. 직장은 원본에 이름이 없어 빈 문자열이다 */
  poi: string;
  dong: string;
  /** commerce | residence | workplace */
  pt: string;
}

/** 방문 기억 */
export interface VisitedMemory {
  day: string;
  /** 중요도 */
  imp: number;
  sat: number | null;
  poi: string;
  cat: string | null;
  sub: string | null;
  /** commerce | residence | workplace — 집·직장은 원본에 이름이 없어 이 값으로 부른다 */
  pt: string | null;
  /**
   * 저장된 요약문. **원본 46,034건 중 18,876건이 인코딩 손상**으로
   * "(직장) ??, ??? 0.69" 처럼 깨져 있어, 깨진 것은 `null` 로 두고 싣지 않았다.
   */
  s: string | null;
}

/** 다른 대상자에게 전해 들은 이야기 */
export interface RumorMemory {
  day: string;
  imp: number;
  s: string;
  /** 이야기를 전한 대상자 id */
  src: string | null;
  tt: string | null;
  tv: string | null;
  /** 대화 의도 (추천 등) */
  ci: string | null;
}

/** 알고 있는 장소 (방문이 쌓여 만들어진 장소별 누적) */
export interface KnownPoi {
  poi: string;
  cat: string | null;
  sub: string | null;
  pt: string | null;
  n: number;
  sat: number;
  /** 선호도 */
  aff: number;
  last: string;
  /** initial(처음부터 알고 있음) | visited(다녀서 알게 됨) */
  src: string;
}

export interface DayState {
  balance: number;
  mood: number;
  fatigue: number;
  yest_sat: number;
}

export interface Appointment {
  day: string;
  target_day: string;
  target_time: string;
  within_window: boolean;
  hint: string;
  meeting_poi_name: string;
  with_agent: string;
  role: string;
}

export interface AgentDetail {
  id: string;
  idx: number;
  profile: AgentProfile;
  days: string[];
  events: AgentEvent[];
  visited: VisitedMemory[];
  /** 인코딩이 깨져 싣지 못한 요약문 건수 */
  visitedDamaged: number;
  rumors: RumorMemory[];
  knows: KnownPoi[];
  state: Record<string, DayState>;
  appointments: Appointment[];
}

/* ==========================================================================
   2. 불러오기 — 한 번 받은 것은 다시 받지 않는다
   ========================================================================== */

const ROOT = `${import.meta.env.BASE_URL}agent-console`;

const HOWTO =
  'web/ui/tools/build_agent_console_data.py 를 실행해 web/ui/public/agent-console/ 을 먼저 만드세요.';

/**
 * 파일이 없을 때 **404 가 오지 않는다.** 개발 서버도 빌드 산출물도 한 페이지 앱이라,
 * 모르는 주소에는 `index.html` 을 200 으로 돌려준다. 그대로 `res.json()` 을 부르면
 * "Unexpected token <" 같은 파서 오류가 나서, 진짜 원인(파일이 없다)이 가려진다.
 * 그래서 상태 코드만 보지 않고 **콘텐츠 종류까지 확인**한다.
 */
async function getJson<T>(url: string, what: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  const type = res.headers.get('content-type') ?? '';
  if (!res.ok) {
    throw new Error(`${what}을(를) 불러오지 못했습니다 (HTTP ${res.status}). ${HOWTO}`);
  }
  if (!type.includes('json')) {
    throw new Error(
      `${url} 자리에 파일이 없습니다. 서버가 대신 앱 문서를 돌려줬습니다(${type || '종류 미상'}). ${HOWTO}`,
    );
  }
  return (await res.json()) as T;
}

let rosterOnce: Promise<Roster> | null = null;

/** 목록은 화면당 한 번만 받는다 (341KB) */
export function loadRoster(): Promise<Roster> {
  rosterOnce ??= getJson<Roster>(`${ROOT}/roster.json`, '대상자 목록');
  return rosterOnce;
}

/**
 * 고른 한 명만 받는다 (약 25KB).
 * 최근 12명까지 들고 있는다 — 목록을 오가며 몇 명을 견줘 보는 동안 다시 받지 않도록.
 * 그 이상은 버린다. 1,825명을 전부 쥐고 있으면 45MB 가 되어 처음 문제로 돌아간다.
 */
const CACHE_MAX = 12;
const cache = new Map<number, Promise<AgentDetail>>();

export function loadAgent(idx: number): Promise<AgentDetail> {
  const hit = cache.get(idx);
  if (hit) {
    // 최근 쓴 것을 뒤로 보내 오래된 것부터 밀려나게 한다
    cache.delete(idx);
    cache.set(idx, hit);
    return hit;
  }
  const p = getJson<AgentDetail>(
    `${ROOT}/agents/${String(idx).padStart(4, '0')}.json`,
    '대상자 기록',
  );
  p.catch(() => cache.delete(idx)); // 실패한 응답을 캐시에 남기지 않는다
  cache.set(idx, p);
  if (cache.size > CACHE_MAX) {
    const oldest = cache.keys().next();
    if (!oldest.done) cache.delete(oldest.value);
  }
  return p;
}

/* ==========================================================================
   3. 목록 좁히기
   ========================================================================== */

export interface RosterFilter {
  gu: string;
  age: string;
  sex: string;
  dec: string;
  q: string;
}

export const EMPTY_FILTER: RosterFilter = { gu: '', age: '', sex: '', dec: '', q: '' };

/** 실제 데이터에 있는 값만 고를 수 있게 한다 — 없는 선택지를 만들지 않는다 */
export function facets(items: RosterItem[]) {
  const gu = new Set<string>();
  const age = new Set<string>();
  for (const it of items) {
    gu.add(it.gu);
    age.add(it.age);
  }
  return {
    gu: [...gu].sort((a, b) => a.localeCompare(b, 'ko')),
    age: [...age].sort((a, b) => a.localeCompare(b, 'ko')),
  };
}

export function filterRoster(items: RosterItem[], f: RosterFilter): RosterItem[] {
  const q = f.q.trim().toLowerCase();
  return items.filter((it) => {
    if (f.gu && it.gu !== f.gu) return false;
    if (f.age && it.age !== f.age) return false;
    if (f.sex && it.sex !== f.sex) return false;
    if (f.dec && String(it.dec) !== f.dec) return false;
    if (!q) return true;
    return (
      it.id.toLowerCase().includes(q) ||
      it.gu.includes(q) ||
      it.dong.includes(q) ||
      it.job.toLowerCase().includes(q)
    );
  });
}

export const SEX_LABEL: Record<string, string> = { F: '여성', M: '남성' };

/** 목록 한 줄에 붙는 설명. 이미 보이는 것(자치구)은 부르는 쪽에서 뺀다 */
export function rosterLine(it: RosterItem): string {
  return `${it.gu} ${it.dong} · ${it.age} ${SEX_LABEL[it.sex] ?? it.sex} · 소비 ${it.dec}분위`;
}

/* ==========================================================================
   4. 말 바꾸기 — 내부 코드를 정책 담당자의 말로 (스펙 §1)
   ========================================================================== */

const TRIGGER_LABEL: Record<string, string> = {
  none: '특별한 계기 없음',
  lifestyle: '평소 생활 패턴',
  top_category: '평소 자주 쓰는 업종',
  mood: '그날의 기분',
  rumor: '다른 사람에게 들은 이야기',
  policy: '정책',
  appointment: '약속',
};

const FACTOR_LABEL: Record<string, string> = {
  distance: '가까워서',
  known: '가 본 곳이라서',
  satisfaction: '만족스러웠던 곳이라서',
  novelty: '가 보지 않은 곳이라서',
  random: '특별한 기준 없이',
  rumor: '들은 이야기 때문에',
  appointment: '약속 장소라서',
};

const KNOWS_SRC_LABEL: Record<string, string> = {
  initial: '처음부터 알고 있던 곳',
  visited: '다니면서 알게 된 곳',
};

export function triggerLabel(v: string | null): string {
  return (v && TRIGGER_LABEL[v]) || v || '기록 없음';
}

export function factorLabel(v: string | null): string | null {
  return v ? (FACTOR_LABEL[v] ?? v) : null;
}

/** poi_id 앞머리로 갈린 장소 종류 */
const PLACE_KIND_LABEL: Record<string, string> = {
  residence: '집',
  workplace: '직장',
  commerce: '상점',
};

/**
 * 집·직장은 원본에 이름이 없다 (`poi_name: ""`).
 * 빈 문자열을 그대로 찍지 않고, 장소 종류·분류에서 부를 이름을 만든다.
 * 그래도 모르면 "이름 기록 없음"이라고 적는다 — 그럴듯한 이름을 붙이지 않는다.
 */
export function placeName(
  poi: string | null,
  kind?: string | null,
  category?: string | null,
): string {
  const name = (poi ?? '').trim();
  if (name) return name;
  const label = (kind && PLACE_KIND_LABEL[kind]) || category;
  return label ? `${label} (이름 기록 없음)` : '이름 기록 없음';
}

/* ==========================================================================
   5. 답 — 기록을 옮겨 적는다
   ========================================================================== */

export type AnswerBlock =
  | { kind: 'p'; text: string }
  | { kind: 'facts'; items: Array<{ k: string; v: string }> }
  | { kind: 'log'; items: Array<{ lead: string; title: string; body?: string; meta?: string }> }
  | { kind: 'bars'; items: Array<{ name: string; value: number; display: string }> };

export interface Answer {
  /** 기록이 있을 때 그리는 것 */
  blocks: AnswerBlock[];
  /** 기록이 없을 때의 한 문장. 있으면 blocks 대신 이것만 그린다 */
  empty?: string;
  /** 어느 파일의 어느 기록인가 — 항상 화면에 함께 보인다 */
  source: string;
}

export type QuestionGroup = 'day' | 'life';

export interface Question {
  id: string;
  group: QuestionGroup;
  /** 사용자가 누르는 문장 */
  text: string;
  /** 자유 입력을 이 질문으로 잇는 낱말들 */
  keywords: string[];
}

/**
 * 물어볼 수 있는 것 전부. **여기 없는 질문에는 답하지 않는다.**
 * 기록에서 답이 나오는 것만 올려 둔다 — 답을 만들어 내는 질문은 넣지 않는다.
 */
export const QUESTIONS: Question[] = [
  {
    id: 'visits',
    group: 'day',
    text: '이 날 어디를 다녀왔나요?',
    keywords: ['어디', '다녀', '방문', '갔', '동선', '하루', '뭐 했', '무엇을 했'],
  },
  {
    id: 'why',
    group: 'day',
    text: '왜 그곳에 갔나요?',
    keywords: ['왜', '이유', '까닭', '어째', '동기', '계기'],
  },
  {
    id: 'spend',
    group: 'day',
    text: '이 날 얼마를 썼나요?',
    keywords: ['얼마', '돈', '소비', '지출', '결제', '썼', '비용', '금액'],
  },
  {
    id: 'best',
    group: 'day',
    text: '가장 만족스러웠던 곳은 어디인가요?',
    keywords: ['만족', '좋았', '최고', '별로', '싫'],
  },
  {
    id: 'mood',
    group: 'day',
    text: '이 날 기분과 피로는 어땠나요?',
    keywords: ['기분', '컨디션', '피로', '피곤', '지쳤', '잔고', '잔액', '기운'],
  },
  {
    id: 'where',
    group: 'day',
    text: '지금 어디에 있나요?',
    keywords: ['지금', '현재', '위치', '시각', '시간', '있나', '있어'],
  },
  {
    id: 'who',
    group: 'life',
    text: '당신은 어떤 사람인가요?',
    keywords: ['누구', '누구세', '소개', '직업', '나이', '사는', '어떤 사람', '본인'],
  },
  {
    id: 'memory',
    group: 'life',
    text: '무엇을 기억하고 있나요?',
    keywords: ['기억', '떠오', '남는', '인상'],
  },
  {
    id: 'regular',
    group: 'life',
    text: '자주 가는 곳은 어디인가요?',
    keywords: ['자주', '단골', '아는 곳', '늘 가', '즐겨'],
  },
  {
    id: 'heard',
    group: 'life',
    text: '누구에게 무슨 이야기를 들었나요?',
    keywords: ['들었', '소문', '이야기', '얘기', '추천', '대화', '전해'],
  },
  {
    id: 'meet',
    group: 'life',
    text: '누구를 만나기로 했나요?',
    keywords: ['만나', '약속', '만남', '보기로', '누굴'],
  },
  {
    id: 'policy',
    group: 'life',
    text: '정책 때문에 움직인 적이 있나요?',
    keywords: ['정책', '지원금', '쿠폰', '바우처', '보조', '혜택'],
  },
  {
    id: 'total',
    group: 'life',
    text: '기간 전체로는 얼마를 썼나요?',
    keywords: ['전체', '총', '합계', '기간', '닷새', '5일'],
  },
];

/** 자유 입력을 질문 하나에 잇는다. 못 잇겠으면 null — 지어내지 않는다 */
export function matchQuestion(input: string): Question | null {
  const text = input.trim().toLowerCase();
  if (!text) return null;
  let best: { q: Question; score: number } | null = null;
  for (const q of QUESTIONS) {
    if (text.includes(q.text.toLowerCase())) return q;
    let score = 0;
    for (const k of q.keywords) if (text.includes(k)) score += k.length;
    if (score > 0 && (!best || score > best.score)) best = { q, score };
  }
  return best?.q ?? null;
}

/* --- 답 만들기 ------------------------------------------------------------ */

const SRC_EVENTS = 'web/viz_store/demo/events.json';
const SRC_MEMORIES = 'web/viz_store/demo/memories.json';
const SRC_AGENTS = 'web/viz_store/demo/agents.json';

const sat = (v: number | null) => (v === null ? '기록 없음' : `만족도 ${dec(v, 2)}`);

function dayEvents(a: AgentDetail, dayIdx: number): AgentEvent[] {
  return a.events.filter((e) => e.d === dayIdx);
}

/**
 * 그 **시간대**에 마지막으로 남은 활동.
 *
 * 시각을 분 단위로 자르지 않고 시간대(15시 = 15:00~15:59)로 묶는 것은
 * 3D 지도의 타임라인 프레임이 쓰는 기준과 같게 하기 위해서다. 원본 timeline.json
 * 프레임 8(08시)의 1,823명을 이 규칙으로 events 에서 되짚어 200명을 대조했을 때
 * 활동·좌표가 전부 일치했다.
 */
function eventAt(a: AgentDetail, dayIdx: number, hour: number): AgentEvent | null {
  const upto = dayEvents(a, dayIdx).filter((e) => Number(e.t.slice(0, 2)) <= hour);
  return upto.length ? upto[upto.length - 1] : null;
}

/** 0~23 → "15시대(15:00~15:59)" */
export function hourBand(hour: number): string {
  const p = String(hour).padStart(2, '0');
  return `${hour}시대(${p}:00~${p}:59)`;
}

export interface AskContext {
  /** 고른 날짜의 번호 */
  dayIdx: number;
  /** "지금 어디에 있나요?" 의 기준 시간대 (0~23시) */
  hour: number;
}

/**
 * 질문 하나에 대한 답. **문장을 만들지 않고 기록을 옮겨 적는다.**
 * 답을 못 찾으면 `empty` 에 왜 없는지 적는다.
 */
export function answer(a: AgentDetail, q: Question, ctx: AskContext): Answer {
  const day = a.days[ctx.dayIdx];
  const evs = dayEvents(a, ctx.dayIdx);

  switch (q.id) {
    /* --- 그 날 ------------------------------------------------------------ */

    case 'visits': {
      if (!evs.length) {
        return { empty: `${day} 에 남은 활동 기록이 없습니다.`, blocks: [], source: SRC_EVENTS };
      }
      return {
        blocks: [
          {
            kind: 'log',
            items: evs.map((e) => ({
              lead: e.t,
              title: `${placeName(e.poi, e.pt, e.cat)} · ${e.dong}`,
              meta: [e.sub ?? e.cat, e.intent, sat(e.sat), e.spent > 0 ? krw(e.spent) : null]
                .filter(Boolean)
                .join(' · '),
            })),
          },
        ],
        source: `${SRC_EVENTS} → ${day} 활동 기록 ${int(evs.length)}건`,
      };
    }

    case 'why': {
      const moved = evs.filter((e) => e.trg !== 'none' || e.pick);
      if (!moved.length) {
        return {
          empty: `${day} 에는 특별한 계기가 기록된 활동이 없습니다. 모두 "계기 없음"으로 남아 있습니다.`,
          blocks: [],
          source: `${SRC_EVENTS} → ${day} trigger·reasoning`,
        };
      }
      return {
        blocks: [
          {
            kind: 'log',
            items: moved.map((e) => ({
              lead: e.t,
              title: `${placeName(e.poi, e.pt, e.cat)} — ${triggerLabel(e.trg)}`,
              body: [e.why, e.pick].filter(Boolean).join('\n'),
              meta: factorLabel(e.pf) ? `고른 기준: ${factorLabel(e.pf)}` : undefined,
            })),
          },
        ],
        source: `${SRC_EVENTS} → ${day} 의 reasoning·trigger·pick_reason ${int(moved.length)}건`,
      };
    }

    case 'spend': {
      const paid = evs.filter((e) => e.spent > 0);
      if (!paid.length) {
        return { empty: `${day} 에는 결제 기록이 없습니다.`, blocks: [], source: SRC_EVENTS };
      }
      const total = paid.reduce((s, e) => s + e.spent, 0);
      const byCat = new Map<string, number>();
      for (const e of paid) byCat.set(e.l1 ?? e.cat, (byCat.get(e.l1 ?? e.cat) ?? 0) + e.spent);
      return {
        blocks: [
          { kind: 'p', text: `${day} 에 ${int(paid.length)}번 결제해 모두 ${krw(total)}을 썼습니다.` },
          {
            kind: 'bars',
            items: [...byCat.entries()]
              .sort((x, y) => y[1] - x[1])
              .map(([name, value]) => ({ name, value, display: krw(value) })),
          },
          {
            kind: 'log',
            items: paid.map((e) => ({
              lead: e.t,
              title: placeName(e.poi, e.pt, e.cat),
              meta: `${e.sub ?? e.cat} · ${krw(e.spent)}`,
            })),
          },
        ],
        source: `${SRC_EVENTS} → ${day} 결제 기록 ${int(paid.length)}건의 spent 합계`,
      };
    }

    case 'best': {
      if (!evs.length) {
        return { empty: `${day} 에 남은 활동 기록이 없습니다.`, blocks: [], source: SRC_EVENTS };
      }
      const sorted = [...evs].sort((x, y) => y.sat - x.sat);
      const top = sorted[0];
      const low = sorted[sorted.length - 1];
      return {
        blocks: [
          {
            kind: 'log',
            items: [
              {
                lead: top.t,
                title: `가장 높음 — ${placeName(top.poi, top.pt, top.cat)}`,
                body: top.why,
                meta: `${top.intent} · ${sat(top.sat)}`,
              },
              {
                lead: low.t,
                title: `가장 낮음 — ${placeName(low.poi, low.pt, low.cat)}`,
                body: low.why,
                meta: `${low.intent} · ${sat(low.sat)}`,
              },
            ],
          },
        ],
        source: `${SRC_EVENTS} → ${day} 활동 ${int(evs.length)}건 중 sat 최고·최저`,
      };
    }

    case 'mood': {
      const st = a.state[day];
      if (!st) {
        return {
          empty: `${day} 의 상태 기록이 없습니다.`,
          blocks: [],
          source: `${SRC_MEMORIES} → state`,
        };
      }
      return {
        blocks: [
          {
            kind: 'facts',
            items: [
              { k: '기분', v: dec(st.mood, 2) },
              { k: '피로', v: dec(st.fatigue, 2) },
              { k: '어제 만족도', v: dec(st.yest_sat, 2) },
              { k: '지갑 잔액', v: krw(st.balance) },
            ],
          },
          { kind: 'p', text: '기분·피로·만족도는 0에 가까울수록 낮고 1에 가까울수록 높습니다.' },
        ],
        source: `${SRC_MEMORIES} → state["${day}"]`,
      };
    }

    case 'where': {
      const band = hourBand(ctx.hour);
      const e = eventAt(a, ctx.dayIdx, ctx.hour);
      if (!e) {
        const first = evs[0];
        return {
          empty: first
            ? `${day} ${band} 까지 남은 기록이 없습니다. 그 날의 첫 기록은 ${first.t} 입니다.`
            : `${day} 에 남은 활동 기록이 없습니다.`,
          blocks: [],
          source: SRC_EVENTS,
        };
      }
      return {
        blocks: [
          {
            kind: 'facts',
            items: [
              { k: '기준 시간대', v: `${day} ${band}` },
              { k: '있는 곳', v: placeName(e.poi, e.pt, e.cat) },
              { k: '행정동', v: e.dong },
              { k: '하고 있는 일', v: `${e.intent} (${e.sub ?? e.cat})` },
              { k: '이 답의 근거가 된 기록', v: `${e.t} — ${e.cat}` },
            ],
          },
          { kind: 'p', text: e.why },
        ],
        source: `${SRC_EVENTS} → ${day} ${band} 까지의 마지막 기록(${e.t}). 3D 지도의 타임라인 프레임과 같은 기준이며, timeline.json 은 같은 기록을 시간대별로 다시 담은 것이라 따로 싣지 않았습니다`,
      };
    }

    /* --- 기간 전체 --------------------------------------------------------- */

    case 'who': {
      const p = a.profile;
      return {
        blocks: [
          { kind: 'p', text: p.lifestyle },
          {
            kind: 'facts',
            items: [
              { k: '나이·성별', v: `${p.age} ${SEX_LABEL[p.gender] ?? p.gender}` },
              { k: '직업', v: p.job },
              { k: '생애 단계', v: p.life_stage },
              { k: '소득 구간', v: p.income },
              { k: '성향', v: p.tendency },
              { k: '사는 곳', v: `${p.district} ${p.home_dong} ${p.home_poi_name}`.trim() },
              { k: '통근 시간', v: `${int(p.commute)}분` },
              { k: '하루 소비 예산', v: `평일 ${krw(p.daily_wd)} · 주말 ${krw(p.daily_we)}` },
            ],
          },
        ],
        source: `${SRC_AGENTS} → id "${a.id}"`,
      };
    }

    case 'memory': {
      if (!a.visited.length) {
        return { empty: '남은 방문 기억이 없습니다.', blocks: [], source: SRC_MEMORIES };
      }
      const top = [...a.visited].sort((x, y) => y.imp - x.imp).slice(0, 10);
      const blocks: AnswerBlock[] = [
        {
          kind: 'log',
          items: top.map((m) => ({
            lead: m.day.slice(5),
            title: placeName(m.poi, m.pt, m.cat),
            body: m.s ?? undefined,
            meta: [m.sub ?? m.cat, sat(m.sat), `중요도 ${dec(m.imp, 2)}`].filter(Boolean).join(' · '),
          })),
        },
      ];
      if (a.visitedDamaged > 0) {
        blocks.push({
          kind: 'p',
          text: `이 대상자의 방문 기억 ${int(a.visited.length)}건 가운데 ${int(
            a.visitedDamaged,
          )}건은 원본 파일에서 요약문 글자가 깨져 있어(예: "(직장) ??, ??? 0.69") 문장을 싣지 않고 날짜·장소·만족도만 옮겼습니다.`,
        });
      }
      return {
        blocks,
        source: `${SRC_MEMORIES} → memories(visited) ${int(a.visited.length)}건 중 중요도 상위 ${int(
          top.length,
        )}건`,
      };
    }

    case 'regular': {
      if (!a.knows.length) {
        return { empty: '알고 있는 장소로 남은 기록이 없습니다.', blocks: [], source: SRC_MEMORIES };
      }
      const top = [...a.knows].sort((x, y) => y.n - x.n).slice(0, 10);
      return {
        blocks: [
          {
            kind: 'bars',
            items: top.map((k) => ({
              name: placeName(k.poi, k.pt, k.cat),
              value: k.n,
              display: `${int(k.n)}회`,
            })),
          },
          {
            kind: 'log',
            items: top.map((k) => ({
              lead: `${int(k.n)}회`,
              title: placeName(k.poi, k.pt, k.cat),
              meta: [
                k.sub ?? k.cat,
                `평균 만족도 ${dec(k.sat, 2)}`,
                `선호도 ${dec(k.aff, 2)}`,
                `마지막 방문 ${k.last}`,
                KNOWS_SRC_LABEL[k.src] ?? k.src,
              ]
                .filter(Boolean)
                .join(' · '),
            })),
          },
        ],
        source: `${SRC_MEMORIES} → knows_poi ${int(a.knows.length)}곳 중 방문 횟수 상위 ${int(
          top.length,
        )}곳`,
      };
    }

    case 'heard': {
      if (!a.rumors.length) {
        return {
          empty: '다른 대상자에게 이야기를 전해 들은 기록이 없습니다.',
          blocks: [],
          source: `${SRC_MEMORIES} → memories(rumor)`,
        };
      }
      // 전해 들은 이야기의 중요도는 원본에서 거의 전부 0.0 이라 적지 않는다.
      // 늘 같은 값을 보이면 지표가 아니라 잡음이 된다 (§7b)
      const told = [...a.rumors].sort((x, y) => y.day.localeCompare(x.day));
      return {
        blocks: [
          {
            kind: 'log',
            items: told.map((r) => ({
              lead: r.day.slice(5),
              title: r.src ? `${r.src} 에게 들음` : '전한 사람 기록 없음',
              body: r.s,
              meta: [r.ci ? `의도: ${r.ci}` : null, r.tv ? `화제: ${r.tv}` : null]
                .filter(Boolean)
                .join(' · '),
            })),
          },
        ],
        source: `${SRC_MEMORIES} → memories(rumor) ${int(a.rumors.length)}건`,
      };
    }

    case 'meet': {
      if (!a.appointments.length) {
        return {
          empty: '약속 기록이 없습니다.',
          blocks: [],
          source: `${SRC_MEMORIES} → appointments`,
        };
      }
      return {
        blocks: [
          {
            kind: 'log',
            items: a.appointments.map((ap) => ({
              lead: ap.day.slice(5),
              title: `${ap.with_agent} 와(과) 약속`,
              meta: [
                `만나기로 한 때 ${ap.target_day} ${ap.target_time}`,
                ap.meeting_poi_name || ap.hint ? `장소 ${ap.meeting_poi_name || ap.hint}` : null,
                ap.role === 'initiator' ? '먼저 제안함' : '제안을 받음',
                ap.within_window ? '기간 안' : '기록된 기간 밖',
              ]
                .filter(Boolean)
                .join(' · '),
            })),
          },
        ],
        source: `${SRC_MEMORIES} → appointments ${int(a.appointments.length)}건`,
      };
    }

    case 'policy': {
      const moved = a.events.filter((e) => e.trg === 'policy');
      if (!moved.length) {
        return {
          empty:
            '정책이 계기가 된 활동 기록이 없습니다. 기록에 남은 계기는 생활 패턴·업종 선호·기분 같은 것뿐입니다.',
          blocks: [],
          source: `${SRC_EVENTS} → trigger === "policy"`,
        };
      }
      return {
        blocks: [
          {
            kind: 'log',
            items: moved.map((e) => ({
              lead: `${a.days[e.d].slice(5)} ${e.t}`,
              title: placeName(e.poi, e.pt, e.cat),
              body: e.why,
              meta: [e.sub ?? e.cat, e.spent > 0 ? krw(e.spent) : null].filter(Boolean).join(' · '),
            })),
          },
        ],
        source: `${SRC_EVENTS} → trigger === "policy" ${int(moved.length)}건`,
      };
    }

    case 'total': {
      const paid = a.events.filter((e) => e.spent > 0);
      if (!paid.length) {
        return { empty: '기간 전체에 결제 기록이 없습니다.', blocks: [], source: SRC_EVENTS };
      }
      const total = paid.reduce((s, e) => s + e.spent, 0);
      const byDay = a.days.map((d, i) => ({
        name: d,
        value: paid.filter((e) => e.d === i).reduce((s, e) => s + e.spent, 0),
      }));
      return {
        blocks: [
          {
            kind: 'p',
            text: `${a.days[0]} 부터 ${a.days[a.days.length - 1]} 까지 ${int(
              paid.length,
            )}번 결제해 모두 ${krw(total)}을 썼습니다.`,
          },
          {
            kind: 'bars',
            items: byDay.map((b) => ({ ...b, display: krw(b.value) })),
          },
        ],
        source: `${SRC_EVENTS} → 활동 ${int(a.events.length)}건 중 결제 ${int(paid.length)}건의 spent 합계`,
      };
    }

    default:
      return {
        empty: '이 질문에 답할 기록이 없습니다.',
        blocks: [],
        source: '해당 없음',
      };
  }
}
