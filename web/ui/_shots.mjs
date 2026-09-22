import { chromium } from '@playwright/test';
import fs from 'node:fs';

const OUT = process.argv[2];
fs.mkdirSync(OUT, { recursive: true });

const routes = [
  ['policy', '/'],
  ['monitor', '/monitor'],
  ['results', '/results'],
  ['visualize', '/visualize'],
];

const browser = await chromium.launch();
const results = [];

for (const [w, h] of [[1440, 900], [768, 900]]) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  for (const [name, path] of routes) {
    await page.goto(`http://localhost:5173${path}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(600);
    const file = `${OUT}/${name}-${w}.png`;
    await page.screenshot({ path: file, fullPage: true });
    const metrics = await page.evaluate(() => ({
      sw: document.documentElement.scrollWidth,
      iw: window.innerWidth,
      docH: document.documentElement.scrollHeight,
    }));
    results.push({ name, w, file, ...metrics });
  }
  // 사이드바 펼침 상태 (1440에서만)
  if (w === 1440) {
    await page.goto('http://localhost:5173/results', { waitUntil: 'networkidle' });
    const toggle = page.locator('button[aria-label*="메뉴"], button[aria-label*="내비"], header button').first();
    if (await toggle.count()) {
      await toggle.click();
      await page.waitForTimeout(500);
      await page.screenshot({ path: `${OUT}/results-1440-nav-open.png`, fullPage: false });
      results.push({ name: 'results-nav-open', w, file: `${OUT}/results-1440-nav-open.png` });
    }
  }
  await ctx.close();
}

await browser.close();
console.log(JSON.stringify(results, null, 2));
