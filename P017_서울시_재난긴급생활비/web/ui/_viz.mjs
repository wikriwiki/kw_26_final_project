import { chromium } from '@playwright/test';

const OUT = process.argv[2];
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

const errors = [];
page.on('console', m => { if (m.type() === 'error') errors.push(m.text().slice(0, 160)); });
page.on('pageerror', e => errors.push('PAGEERROR: ' + String(e).slice(0, 160)));

const t0 = Date.now();
await page.goto('http://localhost:5173/visualize?run=BASE&day=2025-07-21', { waitUntil: 'load', timeout: 120000 });

// iframe 로드 대기
const frame = await page.waitForSelector('iframe.mapstage__frame', { timeout: 60000 });
console.log('iframe present @', Date.now() - t0, 'ms');

// 내부 문서가 실제로 3D 뷰어인지 확인
let inner = null;
for (let i = 0; i < 40; i++) {
  await page.waitForTimeout(3000);
  const f = page.frames().find(f => f.url().includes('/viz/standalone.html'));
  if (!f) continue;
  try {
    inner = await f.evaluate(() => ({
      title: document.title,
      hasMap: !!document.querySelector('#map'),
      canvases: document.querySelectorAll('canvas').length,
      sim3d: typeof window.Sim3D, deck: typeof window.deck, maplibre: typeof window.maplibregl, webgl: !!document.querySelector("canvas")&&!!document.querySelector("canvas").getContext("webgl2"),
      agents: Array.isArray(window.__AGENTS__) ? window.__AGENTS__.length : null,
      frames: Array.isArray(window.__TIMELINE__) ? window.__TIMELINE__.length : null,
      infoFrame: (document.querySelector('#info-frame-label') || {}).textContent,
      activeCnt: (document.querySelector('#info-active-cnt') || {}).textContent,
      totalAgents: (document.querySelector('#info-total-agents') || {}).textContent,
    }));
  } catch (e) { continue; }
  if (inner && inner.canvases > 0 && inner.agents) break;
}
console.log('inner:', JSON.stringify(inner, null, 2));
console.log('elapsed', Date.now() - t0, 'ms');
console.log('errors:', errors.slice(0, 6));

await page.screenshot({ path: OUT, fullPage: false });
await browser.close();
