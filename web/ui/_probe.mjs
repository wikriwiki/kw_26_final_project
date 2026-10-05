import { chromium } from '@playwright/test';
const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
const p = await (await b.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
await p.goto('http://localhost:5173/visualize', { waitUntil: 'load', timeout: 180000 });
let f = null;
for (let i = 0; i < 40 && !f; i++) {
  await p.waitForTimeout(2000);
  f = p.frames().find(x => x.url().includes('/viz/standalone.html')) || null;
  if (f) { try { await f.evaluate(() => window.Sim3D); } catch { f = null; } }
}
const out = await f.evaluate(() => {
  const res = {};
  // "Seoul Sim 3D" 텍스트를 가진 최말단 요소
  const el = [...document.querySelectorAll('*')].find(e => e.children.length === 0 && /Seoul Sim 3D/.test(e.textContent));
  if (el) {
    const cs = getComputedStyle(el);
    res.ghost = { tag: el.tagName, id: el.id, cls: el.className, parentId: el.parentElement?.id, parentTag: el.parentElement?.tagName, color: cs.color, display: cs.display };
  }
  const lg = document.querySelector('#legend-panel');
  if (lg) { const r = lg.getBoundingClientRect(); const cs = getComputedStyle(lg); res.legend = { h: Math.round(r.height), top: Math.round(r.top), maxH: cs.maxHeight, overflow: cs.overflowY, childCount: lg.children.length }; }
  const ip = document.querySelector('#info-panel');
  if (ip) { const r = ip.getBoundingClientRect(); res.info = { w: Math.round(r.width), h: Math.round(r.height) }; }
  const lab = document.querySelector('.legacy-label');
  if (lab) { const r = lab.getBoundingClientRect(); res.label = { w: Math.round(r.width), text: lab.textContent.trim() }; }
  return res;
});
console.log(JSON.stringify(out, null, 2));
await b.close();
