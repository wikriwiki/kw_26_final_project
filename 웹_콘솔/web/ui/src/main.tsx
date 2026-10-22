import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

// 랜딩 페이지와 같은 글꼴. 네트워크 없이 로컬에서 읽는다 — 시연장 망이 막혀도 같은 모습이다.
// Pretendard 는 유니코드 구간별로 잘린 가변 글꼴이라 화면에 나온 글자의 조각만 받는다.
import 'pretendard/dist/web/variable/pretendardvariable-dynamic-subset.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';

import './styles/tokens.css';
import './styles/base.css';
import './styles/layout.css';
import './styles/components.css';
import './styles/screens.css';
import './styles/experiments.css';

import { App } from './App';

// 사이드바 상태를 첫 페인트 전에 반영해 열린 채로 새로고침해도 깜빡이지 않게 한다
try {
  const expanded = window.localStorage.getItem('simconsole.nav.expanded') === 'true';
  document.documentElement.dataset.nav = expanded ? 'expanded' : 'rail';
} catch {
  document.documentElement.dataset.nav = 'rail';
}

const container = document.getElementById('root');
if (!container) throw new Error('#root 를 찾지 못했습니다');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
