/**
 * 효과 막대 — 0 을 가운데에 두고 정책 효과(%)를 좌우로 그린다. 가는 선은 95% 신뢰구간.
 * 구간이 0 을 걸치면 막대를 옅게 그린다(유의하지 않음). 점추정의 부호는 그대로 적는다.
 */
export interface EffectItem {
  key: string;
  label: string;
  effect: number | null;
  ci: [number, number] | null;
  significant: boolean;
  sub?: string;
}

const pct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;

export function EffectBars({ items, span }: { items: EffectItem[]; span?: number }) {
  const shown = items.filter((it) => it.effect !== null);
  // 축 끝은 구간 끝까지 들어오게, 너무 큰 값 하나가 나머지를 납작하게 만들지 않게 60% 에서 자른다
  const extent =
    span ??
    Math.min(0.6, Math.max(0.05, ...shown.flatMap((it) => [Math.abs(it.effect as number), ...(it.ci ?? []).map(Math.abs)])));
  const pos = (v: number) => 50 + (Math.max(-extent, Math.min(extent, v)) / extent) * 50;

  return (
    <ul className="effects" role="list">
      {shown.map((it) => {
        const e = it.effect as number;
        const left = Math.min(pos(0), pos(e));
        const width = Math.abs(pos(e) - pos(0));
        return (
          <li key={it.key} className={`effects__row${it.significant ? '' : ' is-faint'}`}>
            <span className="effects__label">
              {it.label}
              {it.sub ? <small>{it.sub}</small> : null}
            </span>
            <span className="effects__track" aria-hidden="true">
              <span className="effects__zero" />
              <span className={`effects__bar${e < 0 ? ' is-neg' : ''}`} style={{ left: `${left}%`, width: `${width}%` }} />
              {it.ci ? (
                <span className="effects__ci" style={{ left: `${pos(it.ci[0])}%`, width: `${pos(it.ci[1]) - pos(it.ci[0])}%` }} />
              ) : null}
            </span>
            <span className="effects__value num">
              <b>{pct(e)}</b>
              {it.ci ? <small>{`${pct(it.ci[0])} ~ ${pct(it.ci[1])}`}</small> : null}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
