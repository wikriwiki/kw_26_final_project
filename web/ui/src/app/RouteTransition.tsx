/**
 * 라우트 전환 껍데기 — SKILL §7 Animation, 스타일은 `src/styles/motion.css`.
 *
 * 화면이 딱 하고 바뀌는 대신 새 본문이 8px 아래에서 280ms 동안 떠오른다.
 * **지연은 0이다.** 새 화면은 평소와 똑같은 시점에 그려지고, 그 위에 전환이
 * 얹힐 뿐이다. 나가는 화면을 붙잡아 두지 않는 이유가 둘 있다.
 *   1. 붙잡는 만큼 새 내용이 늦게 보인다 — "부드럽게"를 "느리게"로 바꿔 버린다
 *   2. 주소는 이미 바뀐 뒤라, 붙잡아 둔 옛 화면이 새 주소의 run·일자로 다시
 *      그려진다. 부드럽기는커녕 없는 데이터를 읽다 깨진다
 *
 * 셸(사이드바)은 감싸지 않는다. 움직이는 것은 본문 한 덩어리뿐이다
 * (§7 `excessive-motion` — 한 화면에 1~2개).
 */
import { useState } from 'react';
import type { ReactNode } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * - `rise` 옅게 떠오른다. 대부분의 화면
 * - `fade` 불투명도만. 무거운 임베드가 있어 transform 리페인트가 비싼 지도 화면
 * - `none` 전환 없음
 */
export type RouteMotion = 'rise' | 'fade' | 'none';

type Props = {
  children: ReactNode;
  motion?: RouteMotion;
};

export function RouteTransition({ children, motion = 'rise' }: Props) {
  const { pathname } = useLocation();

  /**
   * 라우트가 같고 파라미터만 바뀌면(`/runs/BASE/results` → `/runs/P010/results`)
   * 리액트는 이 요소를 다시 마운트하지 않아 CSS 애니메이션이 되감기지 않는다.
   * 그래서 이름만 다른 두 keyframes 를 번갈아 건다. 클래스가 바뀌는 순간
   * 진행 중이던 애니메이션은 취소되고 새 것이 처음부터 돈다 (§7 `interruptible`).
   *
   * 렌더 도중의 setState 다 — 리액트가 문서에서 권하는 "props 로부터 파생된
   * 상태 보정" 패턴이고, 커밋 전에 즉시 다시 렌더된다. effect 로 미루면
   * 전환 없는 프레임이 한 장 스쳐 지나간다.
   */
  const [seen, setSeen] = useState<{ path: string; phase: 'a' | 'b' }>({
    path: pathname,
    phase: 'a',
  });

  if (seen.path !== pathname) {
    setSeen({ path: pathname, phase: seen.phase === 'a' ? 'b' : 'a' });
  }

  // 쓸데없는 껍데기를 남기지 않는다
  if (motion === 'none') return <>{children}</>;

  return <div className={`route route--${motion} is-${seen.phase}`}>{children}</div>;
}
