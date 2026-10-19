import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

// 한글 본문용 가변 폰트. 네트워크 없이 로컬에서 서브셋을 읽는다.
import '@fontsource-variable/noto-sans-kr';

import './styles/tokens.css';
import './styles/base.css';
import './styles/layout.css';
import './styles/components.css';
import './styles/screens.css';

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
