/**
 * 정책 실행 결과의 세 칸 — 실행 결과 화면과 실험 결과 화면이 함께 쓴다.
 *
 * 1. 정책 효과: 같은 시민·같은 날의 정책 있음 ÷ 없음 − 1, 업종별, 날짜별
 * 2. 집단별 효과: 연령대·소득·성별로 누가 반응했는지. 줄마다 그 집단에게 묻는 길을 둔다
 * 3. 실측 연구와 비교
 */
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Badge } from '../../components/Badge';
import { Card, Stat } from '../../components/Card';
import { Callout } from '../../components/Feedback';
import { EffectBars } from '../../components/EffectBars';
import { LineChart } from '../../components/LineChart';
import { int, krw } from '../../lib/format';
import type { Comparison, GroupDimension, GroupEffectRow, GroupFilters, PairedRow, Validation } from '../../lib/experiments';

export const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? null : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
const ciText = (ci: [number, number] | null) => (ci ? `95% 신뢰구간 ${pct(ci[0])} ~ ${pct(ci[1])}` : undefined);
/** 유의한 효과만 숫자로. 구간이 0을 지나면 '유의하지 않음' */
const sigPct = (r: PairedRow | undefined) => (!r || r.effect === null ? null : r.significant ? pct(r.effect) : '유의하지 않음');
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
  const flat = categories.filter((r) => !r.significant);
  const totalSig = !!total?.significant;
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
        <Stat className="c3" label="총소비 (온라인 포함)" value={sigPct(total)} hint={ciText(total?.ci95 ?? null)} unknownReason="기록 없음" />
        <Stat className="c3" label="가게 소비 (온라인 제외)" value={sigPct(shop)} hint={ciText(shop?.ci95 ?? null)} unknownReason="기록 없음" />
        <Stat
          className="c3"
          label="1인 하루 소비 변화"
          value={uplift === null ? null : totalSig ? wonSigned(uplift) : '유의하지 않음'}
          hint={total ? `${krw(total.off_per_agent_day)} → ${krw(total.on_per_agent_day)}` : undefined}
          unknownReason="기록 없음"
        />
        {wallet ? (
          <Stat
            className="c3"
            label="지원금 1원당 늘어난 소비"
            value={comparison.spend_per_grant == null ? null : totalSig ? `${comparison.spend_per_grant.toFixed(2)}원` : '유의하지 않음'}
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
            items={categories.filter((r) => r.significant).map((r) => ({
              key: r.key,
              label: r.label,
              effect: r.effect,
              ci: r.ci95,
              significant: r.significant,
              sub: `1인 하루 ${krw(r.off_per_agent_day)} → ${krw(r.on_per_agent_day)}`,
            }))}
          />
          {flat.length ? (
            <p className="card__note">
              유의한 변화가 없는 업종: {flat.map((r) => r.label).join(', ')} (95% 신뢰구간이 0을 포함)
            </p>
          ) : null}
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

const share = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`);

/** 유의한 효과만 숫자로 쓴다. 구간이 0을 지나면 '—' — 차이가 있다고 말할 수 없는 값을 결과처럼 늘어놓지 않는다 */
function EffectCell({ row }: { row: PairedRow | undefined | null }) {
  if (!row || row.effect === null) return <td className="n num muted">—</td>;
  if (!row.significant) {
    return (
      <td className="n num muted" title={`${pct(row.effect)} · 95% 구간이 0을 포함`}>
        —
      </td>
    );
  }
  return <td className={`n num${row.effect < 0 ? ' neg' : ''}`}>{pct(row.effect)}</td>;
}

/** 받침에 맞춰 '은/는'을 붙인다 — "50대는", "70대 이상은" */
function topic(word: string): string {
  const code = word.charCodeAt(word.length - 1) - 0xac00;
  return `${word}${code >= 0 && code < 11172 && code % 28 ? '은' : '는'}`;
}

const TITLE: Record<string, string> = {
  grant: '집단별 지원금 사용과 소비 증가',
  cashback: '집단별 적립 업종 소비',
  regulation: '집단별 소비 감소',
  other: '집단별 소비 변화',
};

/** 표 위 한 줄 — 유의한 값만으로 쓴다 */
function groupSummary(rows: GroupEffectRow[], dim: GroupDimension, type: string): string | null {
  const name = (r: GroupEffectRow) => groupValueLabel(dim, r.value);
  const sig = (r: GroupEffectRow, key: string) => r.metrics[key]?.significant && r.metrics[key]?.effect !== null;
  if (type === 'grant') {
    const used = rows.filter((r) => r.grant.use_rate !== null).sort((a, b) => (b.grant.use_rate as number) - (a.grant.use_rate as number));
    const perWon = rows.filter((r) => sig(r, 'total') && r.spend_per_grant !== null).sort((a, b) => (b.spend_per_grant as number) - (a.spend_per_grant as number));
    const parts = [];
    if (used.length > 1) parts.push(`사용률은 ${name(used[0])} ${share(used[0].grant.use_rate)}, ${name(used[used.length - 1])} ${share(used[used.length - 1].grant.use_rate)}`);
    if (perWon.length > 1) parts.push(`지원금 1원당 늘어난 소비는 ${name(perWon[0])} ${perWon[0].spend_per_grant?.toFixed(2)}원, ${name(perWon[perWon.length - 1])} ${perWon[perWon.length - 1].spend_per_grant?.toFixed(2)}원`);
    return parts.length ? `${parts.join('. ')}.` : null;
  }
  if (type === 'cashback') {
    const moved = rows.filter((r) => sig(r, 'elig') && !sig(r, 'total'));
    const grew = rows.filter((r) => sig(r, 'elig') && sig(r, 'total'));
    const parts = [];
    if (grew.length) parts.push(`${topic(grew.map(name).join('·'))} 적립 업종이 늘고 총소비도 늘었다`);
    if (moved.length) parts.push(`${topic(moved.map(name).join('·'))} 적립 업종이 늘어난 만큼 제외 업종이 줄어 총소비 변화가 없다. 소비처만 옮겼다`);
    return parts.length ? `${parts.join('. ')}.` : null;
  }
  const first = rows[0]?.metrics ? Object.keys(rows[0].metrics).find((k) => k.startsWith('cat:')) : undefined;
  if (first) {
    const hit = rows.filter((r) => sig(r, first)).sort((a, b) => (a.metrics[first].effect as number) - (b.metrics[first].effect as number));
    if (hit.length > 1) {
      return `${first.slice(4)} 소비는 ${name(hit[0])} ${pct(hit[0].metrics[first].effect)}로 가장 크게, ${name(hit[hit.length - 1])} ${pct(hit[hit.length - 1].metrics[first].effect)}로 가장 작게 줄었다.`;
    }
  }
  return null;
}

/** 집단별 표 — 정책 종류마다 다른 줄. 줄 끝에서 그 집단에게 묻는다 */
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
  const type = comparison.policy_type ?? 'other';
  const metrics = (comparison.group_metrics ?? [{ key: 'total', label: '총소비' }]).filter((m) => type !== 'grant' || m.key === 'total');
  // 지원금을 모두가 받은 정책(서울시)은 '받은 비율' 칸이 뜻이 없다
  const showRecipients = type === 'grant' && rows.some((r) => (r.grant.recipient_share ?? 1) < 0.99);
  const line = groupSummary(rows, dim, type);
  const label = (key: string, text: string) =>
    key === 'total' ? '총소비' : key === 'elig' ? '적립 업종' : key === 'excl' ? '제외 업종' : key.startsWith('cat:') ? key.slice(4) : text;
  return (
    <section className="section">
      <div className="section__head">
        <h2 className="section__title">{TITLE[type] ?? TITLE.other}</h2>
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
        <div className="table-wrap">
          <table className="table groupeffect__table">
            <thead>
              <tr>
                <th scope="col">{DIMENSION_LABEL[dim]}</th>
                <th scope="col" className="n">시민</th>
                {showRecipients ? <th scope="col" className="n">받은 비율</th> : null}
                {type === 'grant' ? <th scope="col" className="n">사용률</th> : null}
                {metrics.map((m) => (
                  <th key={m.key} scope="col" className="n">
                    {label(m.key, m.label)}
                  </th>
                ))}
                {type === 'grant' ? <th scope="col" className="n">1원당 늘어난 소비</th> : null}
                {type === 'cashback' ? <th scope="col" className="n">3% 넘게 늘린 비율</th> : null}
                {askHref ? <th scope="col"><span className="visually-hidden">인터뷰</span></th> : null}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.value}>
                  <th scope="row">{groupValueLabel(dim, r.value)}</th>
                  <td className="n num">{int(r.agents)}명</td>
                  {showRecipients ? <td className="n num">{share(r.grant.recipient_share)}</td> : null}
                  {type === 'grant' ? <td className="n num">{share(r.grant.use_rate)}</td> : null}
                  {metrics.map((m) => (
                    <EffectCell key={m.key} row={r.metrics[m.key]} />
                  ))}
                  {type === 'grant' ? (
                    <td className={`n num${r.metrics.total?.significant ? '' : ' muted'}`}>
                      {r.metrics.total?.significant && r.spend_per_grant !== null ? `${r.spend_per_grant.toFixed(2)}원` : '—'}
                    </td>
                  ) : null}
                  {type === 'cashback' ? <td className="n num">{share(r.elig_raised_share)}</td> : null}
                  {askHref ? (
                    <td className="n">
                      <Link className="btn btn--secondary btn--sm" to={askHref({ [dim]: [r.value] })}>
                        인터뷰
                      </Link>
                    </td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="card__note">
          {type === 'grant' ? '사용률 = 쓴 지원금 ÷ 받은 지원금(정책 있음 기간 안). 1원당 늘어난 소비 = 늘어난 소비 ÷ 쓴 지원금. ' : ''}
          {type === 'cashback'
            ? '적립 업종 = 캐시백 실적으로 치는 가게. 제외 업종 = 대형마트·백화점·온라인 등. 3% 넘게 늘린 시민 = 같은 시민의 정책 없는 쪽보다 적립 업종 소비가 3% 넘게 많은 사람(실제 2분기 실적은 기록에 없어 대신 씀). '
            : ''}
          변화율은 같은 시민·같은 날 정책 있음 ÷ 없음 − 1. '—'는 95% 신뢰구간이 0을 포함해 차이를 말할 수 없는 값.
        </p>
      </Card>
    </section>
  );
}

/**
 * 사후 검증: 실제 연구와 비교 — 같은 업종 묶음·정의로 다시 센 시뮬 값과 공개 연구의 추정치.
 * 서비스를 쓰는 시점(정책 시행 전)에는 없는 정보라, 시행 전에 보는 결과와 섞이지 않게 따로 표시한다.
 */
export function ValidationCard({ validation }: { validation: Validation }) {
  return (
    <section className="section">
      <div className="section__head">
        <h2 className="section__title">사후 검증: 실제 연구와 비교</h2>
        <Badge tone="info">정책 시행 후에만 알 수 있는 정보</Badge>
      </div>
      <Callout>
        이 서비스를 쓰는 시점은 정책 시행 전이라, 그때는 실제 연구 결과가 없습니다. 이 표는 이미 시행된 정책으로
        시뮬레이션이 맞았는지 확인하려고, 시뮬레이션이 끝난 뒤에 맞춰 본 것입니다. 위의 결과는 이 정보 없이 나온 값입니다.
      </Callout>
      <Card>
        <p className="card__note">
          늘고 줄어든 것이 실제와 같은 항목 {validation.rows.filter((r) => Math.sign(r.sim_pct) === Math.sign(r.truth_pct)).length} / {validation.rows.length}
        </p>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">항목</th>
                <th scope="col" className="n">시뮬레이션</th>
                <th scope="col" className="n">실제</th>
                <th scope="col" className="n">차이</th>
                <th scope="col">증감</th>
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
                  <td className="n num">{Math.abs(row.sim_pct - row.truth_pct).toFixed(1)}%p</td>
                  <td>{Math.sign(row.sim_pct) === Math.sign(row.truth_pct) ? '일치' : '반대'}</td>
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
