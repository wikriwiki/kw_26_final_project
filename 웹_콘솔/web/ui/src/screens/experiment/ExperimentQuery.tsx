/**
 * 집단 질의 — 조건으로 집단을 고르고 (1) 기록 통계를 바로 보거나 (2) 표본에게 같은 질문을 한다.
 *
 * 통계는 LLM 없이 봉인된 하루 집계로 계산한다. 집단 인터뷰는 1대1 인터뷰와 같은 근거 규칙으로 한 명씩 묻고,
 * 끝나면 답들을 모델이 요약한다 — 요약은 확인되지 않은 정리라서 개별 답과 나눠 보여 준다.
 */
import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { Badge } from '../../components/Badge';
import { Button } from '../../components/Button';
import { Card, Stat } from '../../components/Card';
import { Callout, ErrorState } from '../../components/Feedback';
import { SelectField, TextAreaField, TextField } from '../../components/Field';
import { dec, int, percent } from '../../lib/format';
import { experimentsApi, SMOKING_TEXT, useResource } from '../../lib/experiments';
import type { ExperimentSummary, GroupDescribe, GroupFilters, GroupJob, GroupStats, RosterAgent } from '../../lib/experiments';
import { READ_ONLY } from '../../lib/runtime';
import { useExperiment } from './ExperimentWorkspace';

type FilterKey = 'gu' | 'smoking_status' | 'sex' | 'age_group' | 'income_level';
const FILTERS: Array<{ key: FilterKey; label: string; text?: (value: string) => string }> = [
  { key: 'gu', label: '사는 구' },
  { key: 'smoking_status', label: '흡연', text: (v) => SMOKING_TEXT[v] ?? v },
  { key: 'sex', label: '성별', text: (v) => (v === 'F' ? '여' : v === 'M' ? '남' : v) },
  { key: 'age_group', label: '연령대' },
  { key: 'income_level', label: '소득' },
];
const BREAKDOWN_LABEL: Record<string, string> = { smoking_status: '흡연 여부', gu: '사는 구', sex: '성별', age_group: '연령대' };

/** 집단끼리 견줄 때는 만원 단위로 뭉개지 않는다 */
function won(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : `${int(value)}원`;
}

const SEX_TEXT: Record<string, string> = { F: '여', M: '남' };

function groupName(key: string, name: string): string {
  if (key === 'smoking_status') return SMOKING_TEXT[name] ?? name;
  if (key === 'sex') return SEX_TEXT[name] ?? name;
  return name;
}

const INCOME_ORDER = ['하', '중하', '중', '중상', '상'];

/** 고를 수 있는 값. 값이 비어 있는 사람('unknown')은 고를 거리가 아니라서 칩으로 두지 않는다 */
function values(agents: RosterAgent[], key: FilterKey): string[] {
  const found = Array.from(new Set(agents.map((a) => (a[key] ?? 'unknown') as string)));
  const known = found.filter((v) => v !== 'unknown');
  const list = key === 'smoking_status' ? found : known;
  if (key === 'income_level') {
    return list.sort((x, y) => (INCOME_ORDER.indexOf(x) + 99 * Number(!INCOME_ORDER.includes(x))) - (INCOME_ORDER.indexOf(y) + 99 * Number(!INCOME_ORDER.includes(y))));
  }
  return list.sort((x, y) => x.localeCompare(y, 'ko'));
}

function DescribeRow({ name, d, target }: { name: string; d: GroupDescribe; target: boolean }) {
  return (
    <tr>
      <th scope="row">{name}</th>
      <td className="n num">{int(d.agents)}</td>
      <td className="n num">{won(d.spend_per_observed_day)}</td>
      <td className="n num">{dec(d.receipts_per_observed_day, 2)}</td>
      {target ? <td className="n num">{int(d.target_payments)}</td> : null}
      {target ? <td className="n num">{percent(d.agents ? d.agents_with_target_visit / d.agents : null)}</td> : null}
      <td className="n num">{dec(d.mood_mean, 3)}</td>
    </tr>
  );
}

function StatsView({ stats, target }: { stats: GroupStats; target: boolean }) {
  const o = stats.overall;
  return (
    <div className="stack">
      <div className="grid">
        <Stat className="c4" label="고른 사람" value={int(o.agents)} unit="명" hint={`${stats.through_day ?? ''}까지 ${int(stats.days)}일`} />
        <Stat className="c4" label="1인 하루 소비 (오프라인)" value={won(o.spend_per_observed_day)} />
        {target ? (
          <Stat className="c4" label="정책 대상 시설 결제" value={int(o.target_payments)} unit="건" hint={`${int(o.agents_with_target_visit)}명이 한 번 이상`} />
        ) : (
          <Stat className="c4" label="1인 하루 결제 건수" value={dec(o.receipts_per_observed_day, 2)} unit="건" />
        )}
      </div>
      {/* 묶음이 하나뿐인 나눔(예: 흡연 여부가 없는 실험)은 견줄 것이 없다 */}
      {Object.entries(stats.breakdowns).filter(([, groups]) => Object.keys(groups).length > 1).map(([key, groups]) => (
        <div key={key} className="table-wrap">
          <table className="table">
            <caption className="table__caption">{BREAKDOWN_LABEL[key] ?? key}별</caption>
            <thead>
              <tr>
                <th scope="col">묶음</th>
                <th scope="col" className="n">인원</th>
                <th scope="col" className="n">하루 소비</th>
                <th scope="col" className="n">하루 영수증</th>
                {target ? <th scope="col" className="n">대상 시설 결제</th> : null}
                {target ? <th scope="col" className="n">이용한 사람 비율</th> : null}
                <th scope="col" className="n">평균 기분</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(groups).map(([name, d]) => (
                <DescribeRow key={name} name={groupName(key, name)} d={d} target={target} />
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <ul className="plainlist card__note">
        {stats.notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </div>
  );
}

function JobView({ job }: { job: GroupJob }) {
  return (
    <div className="stack">
      <p className="card__note">
        조건에 맞는 {int(job.matched)}명 중 {int(job.sampled.length)}명에게 물었습니다 · {int(job.done)} / {int(job.sampled.length)} 완료
      </p>
      {job.summary?.text ? (
        <Callout tone="info">
          <strong>요약 · {job.summary.model_label}</strong>
          <span className="pre-line">{job.summary.text}</span>
        </Callout>
      ) : null}
      {job.summary?.error ? <Callout tone="warn">요약 실패: {job.summary.error}</Callout> : null}
      <ul className="answers">
        {job.results.map((r) => (
          <li key={r.agent_id} className="answers__item">
            <p className="answers__who">
              <span className="num">{r.agent_id}</span>
              {[r.gu, r.age_group, r.smoking_status ? SMOKING_TEXT[r.smoking_status] : null].filter(Boolean).map((v) => ` · ${v}`).join('')}
            </p>
            {r.error ? <Callout tone="warn">{r.error}</Callout> : <p>{r.answer}</p>}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ExperimentQuery() {
  const { experiment } = useExperiment();
  return <GroupQuery experiment={experiment} />;
}

/**
 * 실험 하나에 대한 집단 질의. 정책 실행의 시민 문답 화면도 이것을 쓴다(`armId` 로 '정책 있음'을 고정해 넘긴다).
 */
export function GroupQuery({ experiment, armId: fixedArm }: { experiment: ExperimentSummary; armId?: string }) {
  const id = experiment.experiment_id;
  const target = experiment.kind === 'no_smoking_zone';
  const roster = useResource(() => experimentsApi.roster(id), id);
  const agents = useMemo(() => roster.data?.items ?? [], [roster.data]);
  const defaultArm = [...experiment.arms].reverse().find((arm) => arm.days_present > 0)?.arm_id ?? experiment.arms[0].arm_id;
  const [armId, setArmId] = useState(fixedArm ?? defaultArm);
  const [filters, setFilters] = useState<Record<FilterKey, string[]>>({ gu: [], smoking_status: [], sex: [], age_group: [], income_level: [] });
  const [stats, setStats] = useState<GroupStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [question, setQuestion] = useState('');
  const [sample, setSample] = useState('6');
  const [job, setJob] = useState<GroupJob | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);

  const payloadFilters: GroupFilters = Object.fromEntries(
    Object.entries(filters).filter(([, list]) => list.length).map(([key, list]) => [key, list]),
  );
  const matched = agents.filter((a) =>
    (Object.entries(filters) as Array<[FilterKey, string[]]>).every(([key, list]) => !list.length || list.includes((a[key] ?? 'unknown') as string)),
  ).length;

  function toggle(key: FilterKey, value: string) {
    setFilters((prev) => {
      const list = prev[key].includes(value) ? prev[key].filter((v) => v !== value) : [...prev[key], value];
      return { ...prev, [key]: list };
    });
  }

  function runStats() {
    setStatsError(null);
    experimentsApi.groupStats(id, { arm_id: armId, filters: payloadFilters }).then(setStats, (e) => setStatsError(e.message));
  }

  function startInterview(event: FormEvent) {
    event.preventDefault();
    setJobError(null);
    experimentsApi
      .groupInterview(id, { arm_id: armId, filters: payloadFilters, question: question.trim(), sample: Number(sample) || 8, seed: 0 })
      .then(setJob, (e) => setJobError(e.message));
  }

  useEffect(() => {
    if (!job || job.state !== 'running') return;
    const timer = window.setTimeout(() => {
      experimentsApi.groupInterviewStatus(id, job.job_id).then(setJob, (e) => setJobError(e.message));
    }, 2000);
    return () => window.clearTimeout(timer);
  }, [id, job]);

  if (roster.error) return <ErrorState title="명단을 불러오지 못했습니다" body={roster.error.message} />;

  return (
    <div className="stack">
      <Card title="집단 고르기" note={`지금 조건에 맞는 사람 ${int(matched)}명`}>
        <div className="querybuilder">
          {fixedArm ? null : (
            <SelectField label="기록" value={armId} onChange={(e) => setArmId(e.target.value)} options={experiment.arms.map((arm) => ({ value: arm.arm_id, label: arm.label }))} />
          )}
          {/* 값이 하나뿐인 조건(이 실험에 없는 항목)은 고를 것이 없다 */}
          {FILTERS.filter((filter) => values(agents, filter.key).length > 1).map((filter) => (
            <fieldset key={filter.key} className="querybuilder__set">
              <legend className="field__label">{filter.label}</legend>
              {values(agents, filter.key).map((value) => (
                <label key={value} className="chip">
                  <input type="checkbox" checked={filters[filter.key].includes(value)} onChange={() => toggle(filter.key, value)} />
                  <span>{filter.text ? filter.text(value) : value}</span>
                </label>
              ))}
            </fieldset>
          ))}
        </div>
        <div className="querybuilder__actions">
          <Button variant="secondary" onClick={runStats} disabled={!matched}>
            기록 통계 보기
          </Button>
        </div>
        {statsError ? <Callout tone="warn">{statsError}</Callout> : null}
      </Card>
      {stats ? (
        <Card title="기록 통계" aside={<Badge>기록 집계</Badge>}>
          <StatsView stats={stats} target={target} />
        </Card>
      ) : null}
      <Card title="집단 인터뷰" note="고른 사람 중 몇 명에게 같은 질문을 합니다 (최대 30명).">
        {!READ_ONLY ? (
          <>
          {experiment.questions?.length ? (
            <div className="segment querychips" role="group" aria-label="질문 예시">
              {experiment.questions.map((text) => (
                <button key={text} type="button" className="segment__btn" aria-pressed={question === text} onClick={() => setQuestion(text)}>
                  {text}
                </button>
              ))}
            </div>
          ) : null}
          <form className="queryform" onSubmit={startInterview}>
            <TextAreaField label="질문" value={question} rows={2} maxLength={500} onChange={(e) => setQuestion(e.target.value)} />
            <TextField label="표본 수" type="number" min={1} max={30} value={sample} onChange={(e) => setSample(e.target.value)} />
            <Button type="submit" variant="primary" disabled={!question.trim() || !matched || job?.state === 'running'} busy={job?.state === 'running'} busyLabel="묻는 중">
              묻기
            </Button>
          </form>
          </>
        ) : (
          <p className="card__note">읽기 전용 배포에서는 집단 인터뷰를 할 수 없습니다.</p>
        )}
        {jobError ? <Callout tone="warn">{jobError}</Callout> : null}
        {job ? <JobView job={job} /> : null}
      </Card>
    </div>
  );
}
