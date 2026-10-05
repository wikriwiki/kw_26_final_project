/**
 * 1대1 인터뷰 — 미리 정해 둔 질문 밖의 물음을 대상자 본인에게 넘긴다.
 *
 * 기록에서 바로 답이 나오는 질문은 화면이 직접 옮겨 적는다(즉시, 정확).
 * 그 밖의 물음만 여기로 온다. 대화 모델은 **이 사람의 기록만** 근거로 답한다.
 *
 * 키는 서버에만 있다. 화면은 질문과 근거만 보낸다.
 */
import type { AgentDetail } from './agentData';

export interface InterviewStatus {
  ready: boolean;
  model_label: string;
  reason: string | null;
}

export interface InterviewReply {
  answer: string;
  model_label: string;
}

export interface InterviewTurn {
  role: 'user' | 'assistant';
  content: string;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error ?? '요청을 처리하지 못했습니다.');
  return body as T;
}

export function interviewStatus(): Promise<InterviewStatus> {
  return call<InterviewStatus>('/api/interview/status');
}

/**
 * 모델에 넘길 근거를 **추린다.**
 *
 * 상세 기록은 한 명당 85KB 다. 통째로 넘기면 모델이 중요한 사실을 놓치고
 * 응답도 느려진다. 사람이 자기 얘기를 할 때 실제로 꺼내는 것만 남긴다.
 */
function evidence(detail: AgentDetail) {
  const p = detail.profile;
  const days = detail.days ?? [];
  const spend: Record<string, number> = {};
  for (const e of detail.events ?? []) {
    if (!e.spent) continue;
    const key = e.l1 ?? e.cat ?? '기타';
    spend[key] = (spend[key] ?? 0) + e.spent;
  }
  const recent = (detail.events ?? [])
    .filter((e) => e.spent)
    .slice(-12)
    .map((e) => ({
      day: days[e.d] ?? '',
      place: e.sub ?? e.cat,
      category: e.l1 ?? e.cat,
      amount: e.spent,
      why: e.why,
    }));
  const memories = (detail.visited ?? [])
    .slice(-8)
    .map((m) => `${m.day} ${m.poi}${m.cat ? ` (${m.cat})` : ''}`);
  // 정책 지갑은 날짜별 상태에 남는다. 마지막 날의 값이 "지금까지" 를 말한다
  const states = Object.values(detail.state ?? {});
  const lastState = (states[states.length - 1] ?? null) as unknown as
    | Record<string, number>
    | null;

  return {
    age_band: p.age,
    gender: p.gender,
    residence: `${p.district ?? ''} ${p.home_dong ?? ''}`.trim(),
    job: p.job,
    lifestyle: p.lifestyle,
    income: p.income,
    grant_total: lastState?.grant_total ?? lastState?.grant_remaining_total ?? undefined,
    grant_used: lastState?.policy_spend_total ?? lastState?.policy_spend_today ?? undefined,
    spend_by_category: spend,
    recent_visits: recent,
    memories,
  };
}

export function askAgent(
  detail: AgentDetail,
  question: string,
  history: InterviewTurn[],
): Promise<InterviewReply> {
  return call<InterviewReply>('/api/interview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agent: evidence(detail), question, history }),
  });
}
