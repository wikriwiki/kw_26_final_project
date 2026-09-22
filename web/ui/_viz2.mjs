import { chromium } from '@playwright/test';

const OUT = process.argv[2];
const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

const netFail = [];
page.on('requestfailed', r => netFail.push(r.url().slice(0, 110)));

await page.goto('http://localhost:5173/visualize?run=BASE&day=2025-07-21', { waitUntil: 'load', timeout: 120000 });
await page.waitForSelector('iframe.mapstage__frame', { timeout: 60000 });

let f = null;
for (let i = 0; i < 40 && !f; i++) {
  await page.waitForTimeout(2000);
  f = page.frames().find(x => x.url().includes('/viz/standalone.html')) || null;
  if (f) { try { await f.evaluate(() => window.deck); } catch { f = null; } }
}
if (!f) { console.log('frame not ready'); process.exit(1); }

// 재생 시작
await f.evaluate(() => {
  const b = document.querySelector('#play-btn') || [...document.querySelectorAll('button')].find(x => /play|재생/i.test(x.textContent));
  if (b) b.click();
});
await page.waitForTimeout(14000);

const st = await f.evaluate(() => ({
  frame: (document.querySelector('#info-frame-label') || {}).textContent,
  active: (document.querySelector('#info-active-cnt') || {}).textContent,
  frameLabel: (document.querySelector('#frame-label') || {}).textContent,
  mapCanvas: !!document.querySelector('#map canvas'),
  canvasCount: document.querySelectorAll('canvas').length,
  deckLayers: (window.__deckInstance && window.__deckInstance.props && window.__deckInstance.props.layers || []).length,
}));
console.log('state:', JSON.stringify(st, null, 2));
console.log('netFail sample:', [...new Set(netFail)].slice(0, 6));

await page.screenshot({ path: OUT, fullPage: false });
await browser.close();
