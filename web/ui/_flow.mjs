import { chromium } from '@playwright/test';
import fs from 'node:fs';

const OUT = process.argv[2];
fs.mkdirSync(OUT, { recursive: true });
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1440, height: 900 } });
const p = await ctx.newPage();
const errs = [];
p.on('pageerror', e => errs.push(String(e).slice(0, 120)));
p.on('console', m => { if (m.type() === 'error' && !/about:blank/.test(m.text())) errs.push(m.text().slice(0, 120)); });

const routes = [
  ['home', '/'],
  ['new', '/new'],
  ['overview', '/runs/BASE'],
  ['monitor', '/runs/BASE/monitor'],
  ['results', '/runs/BASE/results'],
  ['policy', '/runs/BASE/policy'],
  ['overview-stopped', '/runs/BASE7500'],
];

const report = [];
for (const [name, path] of routes) {
  await p.goto('http://localhost:5173' + path, { waitUntil: 'networkidle', timeout: 60000 });
  await p.waitForTimeout(500);
  const m = await p.evaluate(() => {
    const main = document.querySelector('main');
    const root = main ? (main.children.length === 1 ? main.children[0] : main) : document.body;
    return {
      url: location.pathname,
      h1: (document.querySelector('h1') || {}).textContent?.trim().slice(0, 40),
      blocks: root ? root.children.length : 0,
      sw: document.documentElement.scrollWidth,
      iw: window.innerWidth,
      // 화면 안 run 선택기가 남아있나
      runPicker: !!document.querySelector('.segment, [class*="runselect"]'),
      navItems: [...document.querySelectorAll('nav a')].map(a => a.textContent.trim()).filter(Boolean).slice(0, 8),
      buttons: [...document.querySelectorAll('button')].map(x => x.textContent.trim()).filter(Boolean).slice(0, 10),
    };
  });
  report.push({ name, ...m });
  await p.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });
}

// 구 경로 리다이렉트
for (const old of ['/results?run=FINAL', '/visualize?run=BASE&day=2025-07-21', '/monitor']) {
  await p.goto('http://localhost:5173' + old, { waitUntil: 'networkidle', timeout: 60000 });
  await p.waitForTimeout(400);
  report.push({ name: 'redirect ' + old, url: p.url().replace('http://localhost:5173', '') });
}

console.log(JSON.stringify(report, null, 1));
console.log('ERRORS:', [...new Set(errs)].slice(0, 8));
await b.close();
