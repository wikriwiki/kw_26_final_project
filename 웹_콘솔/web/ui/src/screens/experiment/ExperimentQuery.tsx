/**
 * 인터뷰 — 조건으로 집단을 고르고, 그 집단의 정책 효과를 바로 보고, 표본에게 같은 질문을 한다.
 *
 * 정책 실험(정책 있음·없음을 함께 돌린 것)은 정책 담당자가 쓰는 화면이다.
 *   - 인터뷰는 늘 '정책 있음' 쪽 시민에게, 정책 시행 뒤 마지막 날까지의 기록으로 묻는다(날짜를 고르지 않는다).
 *   - 조건을 바꾸면 그 집단의 효과(같은 시민·같은 날 있음 ÷ 없음 − 1)가 바로 바뀐다.
 *   - 질문은 '누구에게·무엇을·왜' 묶음으로 고른다. 누르면 집단과 질문이 함께 채워진다.
 * 정책 있음·없음이 없는 실험은 기록 통계를 본다.
 *
 * 집단 인터뷰는 1대1 인터뷰와 같은 근거 규칙으로 한 명씩 묻고, 끝나면 답들을 모델이 요약한다 —
 * 요약은 확인되지 않은 정리라서 개별 답과 나눠 보여 준다.
 */
import { useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Badge } from '../../components/Badge';
import { Button } from '../../components/Button';
import { Card, Stat } from '../../components/Card';
import { Callout, ErrorState } from '../../components/Feedback';
import { SelectField, TextAreaField, TextField } from '../../components/Field';
import { dec, int, krw, percent } from '../../lib/format';
import { experimentsApi, questionPresets, SMOKING_TEXT, useResource } from '../../lib/experiments';
import type {
  ExperimentSummary,
  GroupDescribe,
  GroupEffect,
  GroupFilters,
  GroupJob,
  GroupStats,
  QuestionPreset,
  RosterAgent,
} from '../../lib/experiments';
import { READ_ONLY } from '../../lib/runtime';
import { useExperiment } from './ExperimentWorkspace';
import { groupValueLabel, pct } from './PolicyPanels';

type FilterKey = 'gu' | 'smoking_status' | 'sex' | 'age_group' | 'income_level';
// 정책 담당자가 먼저 나누는 순서: 나이 → 소득 → 성별 → 사는 곳
const FILTERS: Array<{ key: FilterKey; label: string; text?: (value: string) => string }> = [
  { key: 'age_group', label: '연령대', text: (v) => groupValueLabel('age_group', v) },
  { key: 'income_level', label: '소득' },
  { key: 'sex', label: '성별', text: (v) => (v === 'F' ? '여' : v === 'M' ? '남' : v) },
  { key: 'smoking_status', label: '흡연', text: (v) => SMOKING_TEXT[v] ?? v },
  { key: 'gu', label: '사는 구' },
];
const FILTER_KEYS = FILTERS.map((f) => f.key);
const EMPTY: Record<FilterKey, string[]> = { gu: [], smoking_status: [], sex: [], age_group: [], income_level: [] };
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

/** 고른 조건을 사람이 읽는 말로: "70대 이상 · 소득 하" */
function describeFilters(filters: GroupFilters): string {
  const parts = (Object.entries(filters) as Array<[string, string[] | undefined]>)
    .filter(([, list]) => list && list.length)
    .map(([key, list]) => (list as string[]).map((v) => (key === 'sex' ? (v === 'F' ? '여성' : '남성') : key === 'smoking_status' ? (SMOKING_TEXT[v] ?? v) : groupValueLabel(key, v))).join('·'));
  return parts.length ? parts.join(' · ') : '전체';
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

/** 고른 집단의 정책 효과 — 조건이 바뀔 때마다 다시 센다 */
function GroupEffectNow({ effect, overall, loading }: { effect: GroupEffect | null; overall: number | null; loading: boolean }) {
  if (!effect) return <p className="card__note">{loading ? '이 집단의 정책 효과를 세는 중…' : ''}</p>;
  if (!effect.available || !effect.rows) return <p className="card__note">{effect.reason ?? '효과를 낼 수 없습니다.'}</p>;
  const total = effect.rows.find((r) => r.key === 'total');
  const shop = effect.rows.find((r) => r.key === 'shop');
  const moved = effect.rows
    .filter((r) => r.group === 'category' && r.significant && r.effect !== null)
    .sort((a, b) => Math.abs(b.effect as number) - Math.abs(a.effect as number))
    .slice(0, 4);
  const grant = (effect.wallet?.on_total ?? 0) > 0;
  const sig = !!total?.significant && total.effect != null;
  const diff = sig && overall != null ? (total.effect as number) - overall : null;
  const range = (r: typeof total) => (r?.ci95 ? `95% 구간 ${pct(r.ci95[0])} ~ ${pct(r.ci95[1])}` : undefined);
  const value = (r: typeof total) => (r?.significant ? pct(r.effect) : '유의하지 않음');
  const elig = effect.rows.find((r) => r.key === 'elig');
  const excl = effect.rows.find((r) => r.key === 'excl');
  // 정책 종류마다 보는 지표가 다르다. 값을 낼 수 없는 칸은 그리지 않는다
  const stats: Array<{ label: string; value: string | null; hint?: string }> = [
    { label: '총소비 효과', value: total ? value(total) : null, hint: range(total) },
  ];
  if (diff !== null) {
    stats.push({ label: '전체 평균과 차이', value: `${diff >= 0 ? '+' : ''}${(diff * 100).toFixed(1)}%p`, hint: overall === null ? undefined : `전체 ${pct(overall)}` });
  }
  if (elig || excl) {
    if (elig) stats.push({ label: '적립 업종 소비', value: value(elig), hint: range(elig) });
    if (excl) stats.push({ label: '제외 업종 소비', value: value(excl), hint: range(excl) });
  } else if (shop && !grant) {
    stats.push({ label: '가게 소비 효과', value: value(shop), hint: shop.significant ? '온라인 제외' : range(shop) });
  }
  if (grant) {
    const use = effect.grant?.use_rate;
    if (use != null) stats.push({ label: '지원금 사용률', value: `${Math.round(use * 100)}%`, hint: '쓴 지원금 ÷ 받은 지원금' });
    if (sig && effect.spend_per_grant != null) {
      stats.push({ label: '지원금 1원당 늘어난 소비', value: `${effect.spend_per_grant.toFixed(2)}원`, hint: `쓴 지원금 1인 하루 ${krw(effect.wallet?.per_agent_day ?? 0)}` });
    }
  } else if (sig && effect.uplift_per_agent_day != null) {
    stats.push({ label: '1인 하루 소비 변화', value: `${effect.uplift_per_agent_day >= 0 ? '+' : '−'}${int(Math.abs(effect.uplift_per_agent_day))}원` });
  }
  return (
    <div className="groupnow" aria-live="polite" style={loading ? { opacity: 0.55 } : undefined}>
      {stats.map((s) => (
        <Stat key={s.label} label={s.label} value={s.value} hint={s.hint} unknownReason="기록 없음" />
      ))}
      <p className="groupnow__moves">
        {moved.length
          ? `유의하게 바뀐 업종: ${moved.map((r) => `${r.label} ${pct(r.effect)}`).join(' · ')}`
          : '이 집단에서 유의하게 바뀐 업종은 없습니다.'}
      </p>
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
              {[
                r.gu,
                r.age_group ? groupValueLabel('age_group', r.age_group) : null,
                r.sex ? (r.sex === 'F' ? '여성' : r.sex === 'M' ? '남성' : r.sex) : null,
                r.income_level ? `소득 ${r.income_level}` : null,
                r.smoking_status ? SMOKING_TEXT[r.smoking_status] : null,
              ]
                .filter(Boolean)
                .join(' · ')}
              <span className="num"> · {r.agent_id}</span>
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

/** 주소에 실린 조건(?age_group=70대이상)을 처음 값으로 */
function filtersFrom(params: URLSearchParams): Record<FilterKey, string[]> {
  const out = { ...EMPTY };
  FILTER_KEYS.forEach((key) => {
    out[key] = params.getAll(key);
  });
  return out;
}

/** 프리셋이 가리키는 집단 — 다른 조건은 비운다 */
function filtersOf(preset: QuestionPreset): Record<FilterKey, string[]> {
  const out = { ...EMPTY };
  Object.entries(preset.who ?? {}).forEach(([key, list]) => {
    if ((FILTER_KEYS as string[]).includes(key)) out[key as FilterKey] = [...(list ?? [])];
  });
  return out;
}

const sameFilters = (a: Record<FilterKey, string[]>, b: Record<FilterKey, string[]>) =>
  FILTER_KEYS.every((key) => [...a[key]].sort().join('|') === [...b[key]].sort().join('|'));

/**
 * 실험 하나에 대한 집단 질의. 정책 실행의 시민 인터뷰 화면도 이것을 쓴다(`armId` 로 '정책 있음'을 고정해 넘긴다).
 */
export function GroupQuery({ experiment, armId: fixedArm }: { experiment: ExperimentSummary; armId?: string }) {
  const id = experiment.experiment_id;
  const target = experiment.kind === 'no_smoking_zone';
  const paired = experiment.kind === 'policy_paired';
  const treated = experiment.arms.find((arm) => arm.role === 'treatment')?.arm_id;
  const [params] = useSearchParams();
  const roster = useResource(() => experimentsApi.roster(id), id);
  const overall = useResource(
    () => (paired ? experimentsApi.comparison(id) : Promise.resolve(null)),
    `comparison:${paired ? id : '-'}`,
  );
  const agents = useMemo(() => roster.data?.items ?? [], [roster.data]);
  const defaultArm = [...experiment.arms].reverse().find((arm) => arm.days_present > 0)?.arm_id ?? experiment.arms[0].arm_id;
  // 정책 실험은 늘 정책 있음 쪽, 시행 뒤 기록으로 묻는다
  // 정책 있는 쪽이 있는 실험은 늘 그쪽, 시행 뒤 기록으로 묻는다(기록을 고르게 하지 않는다)
  const lockedArm = fixedArm ?? treated;
  const [armId, setArmId] = useState(lockedArm ?? defaultArm);
  const [filters, setFilters] = useState<Record<FilterKey, string[]>>(() => filtersFrom(params));
  const [stats, setStats] = useState<GroupStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [effect, setEffect] = useState<GroupEffect | null>(null);
  const [effectLoading, setEffectLoading] = useState(false);
  const [question, setQuestion] = useState('');
  const [sample, setSample] = useState('6');
  const [job, setJob] = useState<GroupJob | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const presets = questionPresets(experiment);

  const payloadFilters: GroupFilters = Object.fromEntries(
    Object.entries(filters).filter(([, list]) => list.length).map(([key, list]) => [key, list]),
  );
  const filterKey = JSON.stringify(payloadFilters);
  const matched = agents.filter((a) =>
    (Object.entries(filters) as Array<[FilterKey, string[]]>).every(([key, list]) => !list.length || list.includes((a[key] ?? 'unknown') as string)),
  ).length;

  function toggle(key: FilterKey, value: string) {
    setFilters((prev) => {
      const list = prev[key].includes(value) ? prev[key].filter((v) => v !== value) : [...prev[key], value];
      return { ...prev, [key]: list };
    });
  }

  function choose(preset: QuestionPreset) {
    setFilters(filtersOf(preset));
    setQuestion(preset.q);
    setJob(null);
  }

  // 조건이 바뀌면 그 집단의 효과를 다시 센다(정책 실험만). 빠르게 여러 번 눌러도 마지막 것만 남긴다
  useEffect(() => {
    if (!paired) return;
    let alive = true;
    setEffectLoading(true);
    const timer = window.setTimeout(() => {
      experimentsApi.groupEffect(id, JSON.parse(filterKey) as GroupFilters).then(
        (value) => {
          if (!alive) return;
          setEffect(value);
          setEffectLoading(false);
        },
        (e: Error) => {
          if (!alive) return;
          setEffect({ filters: {}, matched: 0, available: false, reason: e.message });
          setEffectLoading(false);
        },
      );
    }, 250);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [id, paired, filterKey]);

  function runStats() {
    setStatsError(null);
    experimentsApi.groupStats(id, { arm_id: armId, filters: payloadFilters }).then(setStats, (e) => setStatsError(e.message));
  }

  function startInterview(event: FormEvent) {
    event.preventDefault();
    setJobError(null);
    experimentsApi
      .groupInterview(id, { arm_id: armId, filters: payloadFilters, question: question.trim(), sample: Number(sample) || 6, seed: 0 })
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

  const overallEffect = overall.data?.rows?.find((r) => r.key === 'total')?.effect ?? null;
  const who = describeFilters(payloadFilters);

  return (
    <div className="stack">
      {presets.some((p) => p.who || p.why) && !READ_ONLY ? (
        <Card title="질문 예시">
          <div className="presets" role="group" aria-label="질문 예시">
            {presets.map((preset) => (
              <button
                key={preset.q}
                type="button"
                className="preset"
                aria-pressed={question === preset.q && sameFilters(filters, filtersOf(preset))}
                onClick={() => choose(preset)}
              >
                <span className="preset__who">{preset.who_label ?? describeFilters(preset.who ?? {})}</span>
                <span className="preset__q">{preset.q}</span>
                {preset.why ? <span className="preset__why">{preset.why}</span> : null}
              </button>
            ))}
          </div>
        </Card>
      ) : null}

      <Card
        title={`대상 · ${who}`}
        note={`${int(matched)}명`}
      >
        <div className="querybuilder">
          {lockedArm ? null : (
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
        {paired ? (
          <>
            <h4 className="groupnow__title">{who} 정책 효과</h4>
            <GroupEffectNow effect={effect} overall={overallEffect} loading={effectLoading} />
          </>
        ) : (
          <div className="querybuilder__actions">
            <Button variant="secondary" onClick={runStats} disabled={!matched}>
              기록 통계 보기
            </Button>
          </div>
        )}
        {statsError ? <Callout tone="warn">{statsError}</Callout> : null}
      </Card>
      {stats && !paired ? (
        <Card title="기록 통계" aside={<Badge>기록 집계</Badge>}>
          <StatsView stats={stats} target={target} />
        </Card>
      ) : null}
      <Card title="질문" note="대상에서 표본을 뽑아 묻습니다. 표본은 최대 30명.">
        {!READ_ONLY ? (
          <>
            {!presets.some((p) => p.who || p.why) ? (
              presets.length ? (
                <div className="segment querychips" role="group" aria-label="질문 예시">
                  {presets.map((preset) => (
                    <button key={preset.q} type="button" className="segment__btn" aria-pressed={question === preset.q} onClick={() => setQuestion(preset.q)}>
                      {preset.q}
                    </button>
                  ))}
                </div>
              ) : null
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
