/**
 * 정책 효과 한 칸 — 정책 본런 실행이면 같은 시민·같은 날의 정책 없음과 비교한 총소비 변화를 보여 준다.
 *
 * 엔진 기록에는 "정책이 없었다면 쓰지 않았을 돈"이 따로 없다. 본런은 정책 없음 쪽도 함께 돌렸으므로
 * 그 차이가 곧 정책 효과다. 실험 비교(시민 단위 부트스트랩 95% 구간)를 그대로 가져온다.
 * 실험이 없는 실행은 원래 칸(결제 기록의 추가 소비 추정)을 쓰고, 그 값이 비어 있으면 비어 있다고 적는다.
 */
import { Link } from 'react-router-dom';
import { Stat } from './Card';
import { experimentsApi, useResource } from '../lib/experiments';
import { krw } from '../lib/format';

const pct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;

export function RunEffectStat({
  experimentId,
  extraSpent,
  extraKnown,
  className,
  compareHref,
}: {
  experimentId: string | null;
  /** '자세히' 링크가 갈 곳. 없으면 실험의 결과 화면 */
  compareHref?: string;
  extraSpent: number | null;
  extraKnown: boolean;
  className?: string;
}) {
  if (!experimentId) {
    return (
      <Stat
        className={className}
        label="정책이 만든 추가 소비"
        value={extraKnown && extraSpent !== null ? krw(extraSpent) : null}
        hint="값이 비어 있는 건이 많아 하한값입니다"
        unknownReason="결제 기록에 없음"
      />
    );
  }
  return <ExperimentEffect experimentId={experimentId} className={className} compareHref={compareHref} />;
}

function ExperimentEffect({ experimentId, className, compareHref }: { experimentId: string; className?: string; compareHref?: string }) {
  const comparison = useResource(() => experimentsApi.comparison(experimentId), `comparison:${experimentId}`);
  const total = comparison.data?.rows?.find((r) => r.key === 'total') ?? null;
  return (
    <div className={['stat', className ?? ''].filter(Boolean).join(' ')}>
      <span className="stat__label">정책 효과 (총소비)</span>
      <span className={total?.effect != null ? 'stat__value' : 'stat__value stat__value--unknown'}>
        {comparison.loading ? '…' : total?.effect != null ? pct(total.effect) : '알 수 없음'}
      </span>
      <span className="stat__hint">
        {total?.ci95 ? `95% 구간 ${pct(total.ci95[0])} ~ ${pct(total.ci95[1])} · ` : ''}
        <Link to={compareHref ?? `/experiments/${encodeURIComponent(experimentId)}/results`}>결과 자세히</Link>
      </span>
    </div>
  );
}
