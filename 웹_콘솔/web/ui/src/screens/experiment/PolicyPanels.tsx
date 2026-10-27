/**
 * 정책 실행 결과의 세 칸 — 실행 결과 화면과 실험 결과 화면이 함께 쓴다.
 *
 * 1. 정책 효과: 같은 시민·같은 날의 정책 있음 ÷ 없음 − 1, 업종별, 날짜별
 * 2. 집단별 효과: 연령대·소득·성별로 누가 반응했는지. 줄마다 그 집단에게 묻는 길을 둔다
 * 3. 실측 연구와 비교
 */
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Card, Stat } from '../../components/Card';
import { EffectBars } from '../../components/EffectBars';
import { LineChart } from '../../components/LineChart';
import { int, krw } from '../../lib/format';
import type { Comparison, GroupDimension, GroupEffectRow, GroupFilters, Validation } from '../../lib/experiments';

export const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? null : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
const ciText = (ci: [number, number] | null) => (ci ? `95% 신뢰구간 ${pct(ci[0])} ~ ${pct(ci[1])}` : undefined);
const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;
const wonSigned = (v: number) => `${v >= 0 ? '+' : '−'}${int(Math.abs(v))}원`;

export const DIMENSION_LABEL: Record<GroupDimension, string> = { age_group: '연령대', income_level: '소득', sex: '성별' };

export function groupValueLabel(dim: string, value: string): string {
  if (dim === 'sex') return value === 'F' ? '여성' : value === 'M' ? '남성' : value;
  if (dim === 'income_level') return `소득 ${value}`;
  if (value === '70대이상') return '70대 이상';
  return value;
}

/** 정책 효과 — 숫자 셋(넷)과 업종별·날짜별 */
export function PolicyEffect({ comparison, marker }: { comparison: Comparison; marker?: { x: string; label: string } }) {
  if (!comparison.available || !comparison.rows) {
    return (
      <Card title="정책 효과" className="c12">
        <p className="card__note">{comparison.reason ?? '비교할 날이 없습니다.'}</p>
      </Card>
    );
  }
  const total = comparison.rows.find((r) => r.key === 'total');
  const shop = comparison.rows.find((r) => r.key === 'shop');
  const categories = comparison.rows.filter((r) => r.group === 'category');
  const hidden = categories.filter((r) => !r.significant).length;
  const daily = comparison.daily ?? [];
  // 캐시백(다음 달 적립)·거리두기는 결제 때 지갑에서 나가는 돈이 없다. 0원으로 보이지 않게 비운다
  const wallet = comparison.wallet && comparison.wallet.on_total > 0 ? comparison.wallet : null;
  const uplift = comparison.uplift_per_agent_day ?? null;
  return (
    <section className="section">
      <div className="section__head">
        <h2 className="section__title">정책 효과</h2>
        <p className="section__note">
          {int(comparison.agents ?? 0)}명 · {int(comparison.days?.length ?? 0)}일 · 같은 시민·같은 날 비교
        </p>
      </div>
      <div className="grid statrow">
        <Stat className="c3" label="총소비 (온라인 포함)" value={pct(total?.effect)} hint={ciText(total?.ci95 ?? null)} unknownReason="기록 없음" />
        <Stat className="c3" label="가게 소비 (온라인 제외)" value={pct(shop?.effect)} hint={ciText(shop?.ci95 ?? null)} unknownReason="기록 없음" />
        <Stat
          className="c3"
          label="1인 하루 소비 변화"
          value={uplift === null ? null : wonSigned(uplift)}
          hint={total ? `${krw(total.off_per_agent_day)} → ${krw(total.on_per_agent_day)}` : undefined}
          unknownReason="기록 없음"
        />
        {wallet ? (
          <Stat
            className="c3"
            label="지원금 1원당 늘어난 소비"
            value={comparison.spend_per_grant == null ? null : `${comparison.spend_per_grant.toFixed(2)}원`}
            hint={`쓴 지원금 1인 하루 ${krw(wallet.per_agent_day)}`}
            unknownReason="계산할 수 없음"
          />
        ) : (
          <Stat className="c3" label="지원금" value="해당 없음" hint="결제 때 쓰는 지원금이 없는 정책" />
        )}
      </div>
      <div className="grid">
        <Card className="c7" title="업종별 효과">
          <EffectBars
            items={categories.map((r) => ({
              key: r.key,
              label: r.label,
              effect: r.effect,
              ci: r.ci95,
              significant: r.significant,
              sub: `1인 하루 ${krw(r.off_per_agent_day)} → ${krw(r.on_per_agent_day)}`,
            }))}
          />
          <p className="card__note">
            옅은 막대는 95% 신뢰구간이 0을 포함합니다{hidden ? ` (${hidden}개 업종)` : ''}.
          </p>
        </Card>
        <Card className="c5">
          <LineChart
            title="날짜별 총소비 효과"
            series={[
              {
                key: 'effect',
                label: '정책 있음 ÷ 없음 − 1',
                tone: 'primary',
                points: daily.map((d) => ({ x: d.day, y: d.effect === null ? null : d.effect * 100 })),
              },
            ]}
            format={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}
            marker={marker}
          />
          <p className="card__note">신뢰구간: 시민 단위 부트스트랩 500회</p>
        </Card>
      </div>
    </section>
  );
}

/** 집단별 효과에서 한 줄로 읽을 것: 가장 크게·작게 반응한 집단 */
function headline(rows: GroupEffectRow[], dim: GroupDimension): string | null {
  const known = rows.filter((r) => r.total.effect !== null);
  if (known.length < 2) return null;
  const sorted = [...known].sort((a, b) => Math.abs(b.total.effect as number) - Math.abs(a.total.effect as number));
  const top = sorted[0];
  const low = sorted[sorted.length - 1];
  const name = (r: GroupEffectRow) => groupValueLabel(dim, r.value);
  const verb = (top.total.effect as number) < 0 ? '가장 크게 줄어든' : '가장 크게 반응한';
  const overlap =
    top.total.ci95 && low.total.ci95
      ? Math.max(top.total.ci95[0], low.total.ci95[0]) <= Math.min(top.total.ci95[1], low.total.ci95[1])
      : true;
  return `${verb} 집단은 ${name(top)} ${pct(top.total.effect)}, 가장 작은 집단은 ${name(low)} ${pct(low.total.effect)}${
    overlap ? '' : ' — 두 구간이 겹치지 않는다'
  }.`;
}

/** 집단별 효과 — 누가 반응했나. 줄마다 그 집단에게 묻는 길(askHref)을 둔다 */
export function GroupEffectCard({
  comparison,
  askHref,
}: {
  comparison: Comparison;
  askHref?: (filters: GroupFilters) => string;
}) {
  const groups = comparison.by_group ?? {};
  const dims = (Object.keys(DIMENSION_LABEL) as GroupDimension[]).filter((d) => (groups[d]?.length ?? 0) > 1);
  const [dim, setDim] = useState<GroupDimension>(dims[0] ?? 'age_group');
  if (!dims.length) return null;
  const rows = groups[dim] ?? [];
  const grant = (comparison.wallet?.on_total ?? 0) > 0;
  const line = headline(rows, dim);
  return (
    <section className="section">
      <div className="section__head">
        <h2 className="section__title">누가 반응했나</h2>
        <div className="segment" role="group" aria-label="집단 나누는 기준">
          {dims.map((d) => (
            <button key={d} type="button" className="segment__btn" aria-pressed={d === dim} onClick={() => setDim(d)}>
              {DIMENSION_LABEL[d]}
            </button>
          ))}
        </div>
      </div>
      <Card>
        {line ? <p className="groupeffect__headline">{line}</p> : null}
        <EffectBars
          items={rows.map((r) => ({
            key: r.value,
            label: groupValueLabel(dim, r.value),
            effect: r.total.effect,
            ci: r.total.ci95,
            significant: r.total.significant,
            sub: grant && r.spend_per_grant !== null
              ? `${int(r.agents)}명 · 지원금 1원당 ${r.spend_per_grant.toFixed(2)}원`
              : `${int(r.agents)}명 · 1인 하루 ${wonSigned(r.uplift_per_agent_day)}`,
            action: askHref ? (
              <Link className="btn btn--secondary btn--sm" to={askHref({ [dim]: [r.value] })}>
                이 집단에게 묻기
              </Link>
            ) : undefined,
          }))}
        />
        <p className="card__note">
          총소비 효과(같은 시민·같은 날). 옅은 막대는 95% 신뢰구간이 0을 포함합니다.
          {grant ? ' 지원금 1원당: 늘어난 1인 하루 소비 ÷ 쓴 지원금.' : ''}
        </p>
      </Card>
    </section>
  );
}

/** 실측 연구와 비교 — 같은 업종 묶음·정의로 다시 센 시뮬 값과 공개 연구의 추정치 */
export function ValidationCard({ validation }: { validation: Validation }) {
  return (
    <section className="section">
      <div className="section__head">
        <h2 className="section__title">실측 연구와 비교</h2>
        <p className="section__note">
          방향이 같은 항목 {validation.rows.filter((r) => Math.sign(r.sim_pct) === Math.sign(r.truth_pct)).length} / {validation.rows.length}
        </p>
      </div>
      <Card>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">항목</th>
                <th scope="col" className="n">시뮬</th>
                <th scope="col" className="n">실측</th>
                <th scope="col">방향</th>
                <th scope="col">출처</th>
              </tr>
            </thead>
            <tbody>
              {validation.rows.map((row) => (
                <tr key={row.label}>
                  <th scope="row">{row.label}</th>
                  <td className="n num">
                    {signed(row.sim_pct)}
                    {row.sim_ci95_pct ? (
                      <span className="cell-sub">
                        {signed(row.sim_ci95_pct[0])} ~ {signed(row.sim_ci95_pct[1])}
                      </span>
                    ) : null}
                  </td>
                  <td className="n num">{signed(row.truth_pct)}</td>
                  <td>{Math.sign(row.sim_pct) === Math.sign(row.truth_pct) ? '같음' : '반대'}</td>
                  <td className="wrap">
                    <span className="cell-sub">{row.source}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="card__note">{validation.basis}</p>
        <p className="card__note">{validation.note}</p>
      </Card>
    </section>
  );
}
