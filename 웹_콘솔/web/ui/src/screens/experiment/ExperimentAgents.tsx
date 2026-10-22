/**
 * 대상자 — 명단에서 한 사람을 고르고, 그 사람의 하루 기록을 보고, 근거를 달아 묻는다.
 *
 * 인터뷰 규칙(서버 `experiment_interview.py`):
 *   · 고른 날짜까지의 봉인된 기록만 근거로 쓴다. 답의 [E#] 은 실제로 보여 준 근거여야 한다.
 *   · 모델이 밝힌 이유는 "확인되지 않은 진술"이다. 실행된 사실과 다르게 표시한다.
 *   · 이 대화는 콘솔 탐색용이다. 실험 기록(사후 인터뷰)에 들어가지 않는다.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Badge } from '../../components/Badge';
import { Button } from '../../components/Button';
import { Card } from '../../components/Card';
import { Disclosure } from '../../components/Disclosure';
import { Callout, EmptyState, ErrorState, SkeletonCard } from '../../components/Feedback';
import { SelectField, TextAreaField } from '../../components/Field';
import { dec, int, krw } from '../../lib/format';
import { experimentsApi, SMOKING_TEXT, useResource } from '../../lib/experiments';
import type { AgentDay, ExperimentSummary, InterviewReply, InterviewStatus, RosterAgent } from '../../lib/experiments';
import { READ_ONLY } from '../../lib/runtime';
import { useExperiment } from './ExperimentWorkspace';

const PAGE = 120;
const KIND_LABEL: Record<string, string> = {
  profile: '프로필',
  day_state: '하루 상태',
  executed_receipt: '영수증',
  target_facility: '정책 대상 시설',
  skipped_day: '기록 없음',
  executed_event: '실행된 일',
  stated_rationale: '본인이 밝힌 이유',
};

function sexText(sex?: string | null): string {
  return sex === 'F' ? '여' : sex === 'M' ? '남' : '성별 미상';
}

function uniqueValues(agents: RosterAgent[], key: keyof RosterAgent): string[] {
  return Array.from(new Set(agents.map((a) => a[key]).filter((v): v is string => typeof v === 'string' && v !== ''))).sort();
}

function Roster({ experiment, selected, onSelect }: { experiment: ExperimentSummary; selected: string | null; onSelect: (id: string) => void }) {
  const roster = useResource(() => experimentsApi.roster(experiment.experiment_id), experiment.experiment_id);
  const [gu, setGu] = useState('');
  const [smoking, setSmoking] = useState('');
  const [sex, setSex] = useState('');
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(PAGE);
  const agents = useMemo(() => roster.data?.items ?? [], [roster.data]);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return agents.filter(
      (a) =>
        (!gu || a.gu === gu) &&
        (!smoking || (a.smoking_status ?? 'unknown') === smoking) &&
        (!sex || a.sex === sex) &&
        (!q || [a.id, a.dong, a.job].some((v) => (v ?? '').toLowerCase().includes(q))),
    );
  }, [agents, gu, smoking, sex, query]);

  if (roster.loading && !roster.data) return <SkeletonCard rows={6} />;
  if (roster.error) return <ErrorState title="명단을 불러오지 못했습니다" body={roster.error.message} />;

  return (
    <Card title={`시민 ${int(filtered.length)} / ${int(agents.length)}명`} note={roster.data?.source === 'metrics' ? '프로필이 없어 id 만 보입니다.' : undefined} className="agentroster">
      <div className="agentroster__filters">
        <SelectField label="구" value={gu} onChange={(e) => setGu(e.target.value)} options={[{ value: '', label: '전체' }, ...uniqueValues(agents, 'gu').map((v) => ({ value: v, label: v }))]} />
        {experiment.kind === 'no_smoking_zone' ? (
          <SelectField
            label="흡연"
            value={smoking}
            onChange={(e) => setSmoking(e.target.value)}
            options={[{ value: '', label: '전체' }, { value: 'smoker', label: '흡연' }, { value: 'non_smoker', label: '비흡연' }, { value: 'unknown', label: '미상' }]}
          />
        ) : null}
        <SelectField label="성별" value={sex} onChange={(e) => setSex(e.target.value)} options={[{ value: '', label: '전체' }, { value: 'F', label: '여' }, { value: 'M', label: '남' }]} />
        <label className="field agentroster__search">
          <span className="field__label">찾기</span>
          <input className="field__control" type="search" value={query} placeholder="id · 동 · 직업" onChange={(e) => setQuery(e.target.value)} />
        </label>
      </div>
      <ul className="agentlist" aria-label="시민 명단">
        {filtered.slice(0, limit).map((agent) => (
          <li key={agent.id}>
            <button
              type="button"
              className={agent.id === selected ? 'agentlist__row is-selected' : 'agentlist__row'}
              aria-pressed={agent.id === selected}
              onClick={() => onSelect(agent.id)}
            >
              <span className="agentlist__id num">{agent.id}</span>
              <span className="agentlist__meta">
                {[agent.gu, agent.dong].filter(Boolean).join(' ')} · {agent.age ?? agent.age_group ?? '나이 미상'}
                {agent.age ? '세' : ''} {sexText(agent.sex)} · {agent.job ?? '직업 미상'}
              </span>
              {agent.smoking_status ? <span className="agentlist__tag">{SMOKING_TEXT[agent.smoking_status] ?? agent.smoking_status}</span> : null}
            </button>
          </li>
        ))}
      </ul>
      {filtered.length > limit ? (
        <Button variant="ghost" onClick={() => setLimit((n) => n + PAGE)}>
          {int(Math.min(PAGE, filtered.length - limit))}명 더 보기
        </Button>
      ) : null}
    </Card>
  );
}

function DayRecord({ day }: { day: AgentDay }) {
  const target = day.no_smoking?.by_poi ?? [];
  const meta =
    day.status === 'skipped'
      ? '기록 없음'
      : `결제 ${int(day.receipts.filter((r) => r.status === 'purchased').length)}건 · ${krw(day.spend)}${target.length ? ` · 정책 대상 시설 ${target.length}곳` : ''}`;
  return (
    <Disclosure title={day.day} meta={meta}>
      {day.status === 'skipped' ? (
        <p className="card__note">이날은 기록이 없습니다.</p>
      ) : (
        <div className="dayrecord">
          <p className="dayrecord__state">
            기분 {dec(day.mood, 2)} · 피로 {dec(day.fatigue, 2)} · 만족도 {dec(day.avg_sat, 2)} · 잔액 {krw(day.balance)}
            {day.no_smoking?.policy_active ? ' · 금연구역 정책 시행 중' : ''}
          </p>
          <table className="table table--compact">
            <thead>
              <tr>
                <th scope="col">시각</th>
                <th scope="col">분류</th>
                <th scope="col" className="n">금액</th>
                <th scope="col">상태</th>
              </tr>
            </thead>
            <tbody>
              {day.receipts.map((receipt) => (
                <tr key={receipt.event_id ?? `${receipt.time}-${receipt.category}`}>
                  <td className="num">{receipt.time ?? '—'}</td>
                  <td>{receipt.category ?? '—'}</td>
                  <td className="n num">{krw(receipt.amount)}</td>
                  <td>{receipt.status === 'purchased' ? '결제' : receipt.status ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {target.length ? (
            <ul className="plainlist">
              {target.map((poi) => (
                <li key={poi.poi_id}>
                  정책 대상 시설: {poi.facility} · {int(poi.payment_count)}회 · {krw(poi.revenue_krw)}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      )}
    </Disclosure>
  );
}

interface Turn {
  question: string;
  reply?: InterviewReply;
  error?: string;
}

function Interview({ experiment, agentId, arms }: { experiment: ExperimentSummary; agentId: string; arms: Array<{ arm_id: string; label: string; days: AgentDay[] }> }) {
  const status = useResource<InterviewStatus>(() => experimentsApi.interviewStatus(), 'interview-status');
  const usable = arms.filter((arm) => arm.days.length);
  const [armId, setArmId] = useState(() => usable[usable.length - 1]?.arm_id ?? experiment.arms[0].arm_id);
  const lineage = useMemo(() => {
    const byId = new Map(experiment.arms.map((arm) => [arm.arm_id, arm]));
    const chain: string[] = [];
    let current = byId.get(armId);
    while (current) {
      chain.unshift(current.arm_id);
      current = current.parent ? byId.get(current.parent) : undefined;
    }
    return chain;
  }, [experiment.arms, armId]);
  const dayOptions = useMemo(
    () => arms.filter((arm) => lineage.includes(arm.arm_id)).flatMap((arm) => arm.days.map((day) => day.day)).sort(),
    [arms, lineage],
  );
  const [throughDay, setThroughDay] = useState('');
  const [question, setQuestion] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setTurns([]);
  }, [agentId, armId]);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'nearest' });
  }, [turns.length]);

  async function ask(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    // 지난 대화는 오래된 것부터 질문·답 순서로 넘긴다
    const history = turns
      .filter((t) => t.reply)
      .slice(-3)
      .flatMap((t) => [
        { role: 'user', content: t.question },
        { role: 'assistant', content: t.reply!.answer },
      ]);
    setBusy(true);
    setQuestion('');
    setTurns((prev) => [...prev, { question: text }]);
    try {
      const reply = await experimentsApi.interview(experiment.experiment_id, agentId, {
        arm_id: armId,
        question: text,
        through_day: throughDay || null,
        history,
      });
      setTurns((prev) => prev.map((t, i) => (i === prev.length - 1 ? { ...t, reply } : t)));
    } catch (error) {
      const message = (error as { message?: string }).message ?? '알 수 없는 오류';
      setTurns((prev) => prev.map((t, i) => (i === prev.length - 1 ? { ...t, error: message } : t)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      title="인터뷰"
      aside={status.data ? <Badge tone={status.data.ready ? 'ok' : 'warn'}>{status.data.ready ? status.data.model_label : '모델 미연결'}</Badge> : null}
      note="이 시민의 기록만 근거로 답합니다."
    >
      <div className="interview__controls">
        <SelectField label="기록" value={armId} onChange={(e) => setArmId(e.target.value)} options={experiment.arms.map((arm) => ({ value: arm.arm_id, label: arm.label }))} />
        <SelectField
          label="기억하는 날"
          value={throughDay}
          onChange={(e) => setThroughDay(e.target.value)}
          options={[{ value: '', label: '마지막 날' }, ...dayOptions.map((day) => ({ value: day, label: day }))]}
        />
      </div>
      {status.data && !status.data.ready ? <Callout tone="warn">{status.data.reason}</Callout> : null}
      <div className="interview__log" aria-live="polite">
        {turns.length === 0 ? <p className="card__note">예: "지원금은 어디에 썼어요?", "어제 뭐 했어요?", "요즘 돈 쓰는 게 어때요?"</p> : null}
        {turns.map((turn, index) => (
          <div key={index} className="interview__turn">
            <p className="interview__q">
              <span className="sr-only">질문: </span>
              {turn.question}
            </p>
            {turn.error ? <Callout tone="warn">{turn.error}</Callout> : null}
            {turn.reply ? (
              <div className="interview__a">
                <p>{turn.reply.answer}</p>
                {turn.reply.citations.length ? (
                  <details className="interview__src">
                    <summary>답의 바탕이 된 기록 {int(turn.reply.citations.length)}건</summary>
                  <ol className="citations">
                    {turn.reply.citations.map((c) => (
                      <li key={c.n} data-kind={c.kind}>
                        <span className="citations__n num">E{c.n}</span>
                        <span className="citations__kind">{KIND_LABEL[c.kind] ?? c.kind}</span>
                        <span className="citations__text">
                          {c.day ? `${c.day} · ` : ''}
                          {c.text}
                        </span>
                      </li>
                    ))}
                  </ol>
                  </details>
                ) : null}
                <p className="interview__meta">
                  {turn.reply.through_day}까지{turn.reply.model_label ? ` · ${turn.reply.model_label}` : ''}
                </p>
              </div>
            ) : !turn.error ? (
              <p className="card__note">묻는 중…</p>
            ) : null}
          </div>
        ))}
        <div ref={endRef} />
      </div>
      {!READ_ONLY ? (
        <form className="interview__form" onSubmit={ask}>
          <TextAreaField label="질문" value={question} maxLength={500} rows={2} onChange={(e) => setQuestion(e.target.value)} disabled={!status.data?.ready} />
          <Button type="submit" variant="primary" busy={busy} busyLabel="묻는 중" disabled={!question.trim() || !status.data?.ready}>
            묻기
          </Button>
        </form>
      ) : (
        <p className="card__note">읽기 전용 배포에서는 인터뷰를 할 수 없습니다.</p>
      )}
    </Card>
  );
}

function AgentPanel({ experiment, agentId }: { experiment: ExperimentSummary; agentId: string }) {
  const detail = useResource(() => experimentsApi.agent(experiment.experiment_id, agentId), `${experiment.experiment_id}/${agentId}`);
  if (detail.loading && !detail.data) return <SkeletonCard rows={6} />;
  if (detail.error || !detail.data) return <ErrorState title="기록을 불러오지 못했습니다" body={detail.error?.message ?? ''} />;
  const { agent, arms } = detail.data;
  return (
    <div className="stack">
      <Card title={agent.id} note={[agent.gu, agent.dong, agent.age ? `${agent.age}세` : agent.age_group, sexText(agent.sex), agent.job, agent.life_stage, agent.income_level ? `소득 ${agent.income_level}` : null].filter(Boolean).join(' · ')}
        aside={agent.smoking_status ? <Badge tone={agent.smoking_status === 'smoker' ? 'info' : 'neutral'}>{SMOKING_TEXT[agent.smoking_status]}</Badge> : undefined}>
        {arms.map((arm) =>
          arm.days.length ? (
            <section key={arm.arm_id} className="agentarm">
              <h4 className="agentarm__title">
                {arm.label} <span className="card__note">{arm.days.length}일</span>
              </h4>
              {arm.days.map((day) => (
                <DayRecord key={`${arm.arm_id}-${day.day}`} day={day} />
              ))}
            </section>
          ) : null,
        )}
      </Card>
      <Interview experiment={experiment} agentId={agent.id} arms={arms} />
    </div>
  );
}

export function ExperimentAgents() {
  const { experiment } = useExperiment();
  const [params, setParams] = useSearchParams();
  const selected = params.get('agent');
  const select = (id: string) => {
    const next = new URLSearchParams(params);
    next.set('agent', id);
    setParams(next);
  };
  return (
    <div className="agentsplit">
      <Roster experiment={experiment} selected={selected} onSelect={select} />
      <div className="agentsplit__detail">
        {selected ? (
          <AgentPanel key={selected} experiment={experiment} agentId={selected} />
        ) : (
          <EmptyState title="왼쪽 명단에서 한 사람을 고르세요" body="시민을 고르면 하루 기록과 인터뷰가 열립니다." />
        )}
      </div>
    </div>
  );
}
