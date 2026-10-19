import { expect, test } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const evidenceRoot = process.env.GAUNTLET_EVIDENCE_DIR
  ? path.resolve(process.env.GAUNTLET_EVIDENCE_DIR)
  : path.resolve(process.cwd(), '../../docs/gauntlet/evidence/s7-redesign');

const views = [
  { name: 'library', hash: '#/simulations' },
  { name: 'setup-about', hash: '#/simulations/new/about' },
  { name: 'setup-environment', hash: '#/simulations/new/environment' },
  { name: 'setup-policy', hash: '#/simulations/new/policy' },
  { name: 'run-overview', hash: '#/simulations/BASE/overview' },
  { name: 'run-timeline', hash: '#/simulations/BASE/timeline' },
  { name: 'run-agents', hash: '#/simulations/BASE/agents' },
  { name: 'run-visualization', hash: '#/simulations/BASE/visualization' },
  { name: 'run-reports', hash: '#/simulations/BASE/reports' },
];

const viewports = [
  { name: 'wide', width: 1440, height: 900 },
  { name: 'desktop', width: 1280, height: 800 },
];

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.removeItem('simconsole.simulationDraft.v1');
    if (!localStorage.getItem('simconsole.theme')) localStorage.setItem('simconsole.theme', 'light');
  });
});

test('전면 재설계 화면을 1440·1280 PC에서 검증한다', async ({ page, baseURL }) => {
  await mkdir(evidenceRoot, { recursive: true });
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('pageerror', (error) => pageErrors.push(error.message));

  const observations: Array<Record<string, unknown>> = [];
  for (const viewport of viewports) {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    for (const view of views) {
      await page.goto(`${baseURL}/${view.hash}`, { waitUntil: 'networkidle' });
      await page.waitForTimeout(250);
      const layout = await page.evaluate(() => ({
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
        horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
        nestedVerticalScrollers: [...document.querySelectorAll<HTMLElement>('#studio-main *')]
          .filter((element) => {
            const style = getComputedStyle(element);
            return /(auto|scroll)/.test(style.overflowY) && element.scrollHeight > element.clientHeight + 2;
          })
          .map((element) => element.className || element.tagName),
        theme: document.documentElement.dataset.theme,
      }));
      expect(layout.horizontalOverflow, `${viewport.name}/${view.name} 가로 오버플로`).toBe(false);
      expect(layout.nestedVerticalScrollers, `${viewport.name}/${view.name} 중첩 세로 스크롤`).toEqual([]);
      expect(layout.theme).toBe('light');
      await page.screenshot({ path: path.join(evidenceRoot, `${viewport.name}-${view.name}.png`), fullPage: true });
      observations.push({ viewport, view: view.name, layout });
    }
  }

  await writeFile(
    path.join(evidenceRoot, 'visual-manifest.json'),
    JSON.stringify({ generatedAt: new Date().toISOString(), observations, consoleErrors, pageErrors }, null, 2),
    'utf8',
  );
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test('설정 마법사가 단계별 URL·자동 저장·정책 주입·검토 흐름을 유지한다', async ({ page, baseURL }) => {
  await page.setViewportSize({ width: 1280, height: 820 });
  await page.goto(`${baseURL}/#/simulations/new/about`, { waitUntil: 'networkidle' });

  await expect(page.getByRole('heading', { name: '이 시뮬레이션은 무엇을 검증하나요?' })).toBeVisible();
  await page.getByLabel('시뮬레이션 이름', { exact: true }).fill('민생회복 소비쿠폰 2주 효과 검증');
  await page.getByLabel('시뮬레이션 설명', { exact: true }).fill('정책 주입 전후의 소비와 방문 변화를 비교합니다.');
  await page.getByRole('button', { name: '다음 단계' }).click();

  await expect(page).toHaveURL(/\/environment$/);
  await expect(page.getByRole('heading', { name: '실행 환경을 설정하세요' })).toBeVisible();
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0);
  await page.getByLabel('에이전트 수', { exact: true }).fill('15000');
  await page.getByRole('button', { name: '다음 단계' }).click();

  await expect(page).toHaveURL(/\/policy$/);
  await expect(page.getByRole('heading', { name: '어떤 정책을 언제 주입할까요?' })).toBeVisible();
  await page.getByRole('radio', { name: /민생회복 소비쿠폰 1차/ }).check();
  const injectionDate = page.getByLabel('정책 주입 날짜', { exact: true });
  await expect(injectionDate).toBeVisible();
  await expect(injectionDate).toHaveValue('');
  const simulationStart = await injectionDate.getAttribute('min');
  expect(simulationStart).toBeTruthy();
  const explicitInjectionDate = new Date(`${simulationStart}T00:00:00Z`);
  explicitInjectionDate.setUTCDate(explicitInjectionDate.getUTCDate() + 7);
  await injectionDate.fill(explicitInjectionDate.toISOString().slice(0, 10));
  await expect(page.getByText(/정책 전 7일 · 주입일부터 정책 후 7일/)).toBeVisible();
  await page.screenshot({ path: path.join(evidenceRoot, 'interaction-policy-selected.png'), fullPage: true });
  await page.getByRole('button', { name: '다음 단계' }).click();

  await expect(page).toHaveURL(/\/review$/);
  await expect(page.getByRole('heading', { name: '설계를 마지막으로 검토하세요' })).toBeVisible();
  await expect(page.getByText('민생회복 소비쿠폰 2주 효과 검증')).toBeVisible();
  await expect(page.getByText(/P010 · 민생회복 소비쿠폰 1차/)).toBeVisible();
  const storedDraft = await page.evaluate(() => localStorage.getItem('simconsole.simulationDraft.v1'));
  expect(storedDraft).toContain('15000');
  expect(storedDraft).toContain('P010');

  await page.screenshot({ path: path.join(evidenceRoot, 'interaction-wizard-review.png'), fullPage: true });
});

test('PC 테마·reduced-motion·키보드 탐색 접근성을 검증한다', async ({ page, baseURL }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(`${baseURL}/#/simulations`, { waitUntil: 'networkidle' });
  const themeButton = page.getByRole('button', { name: '다크 테마로 전환' });
  await expect(themeButton).toBeVisible();
  await themeButton.click();
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');
  await page.reload({ waitUntil: 'networkidle' });
  await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');

  for (const view of [
    { name: 'dark-run-agents', hash: '#/simulations/BASE/agents' },
    { name: 'dark-run-reports', hash: '#/simulations/BASE/reports' },
  ]) {
    await page.goto(`${baseURL}/${view.hash}`, { waitUntil: 'networkidle' });
    const overflow = await page.evaluate(() => ({
      horizontal: document.documentElement.scrollWidth > document.documentElement.clientWidth,
      nested: [...document.querySelectorAll<HTMLElement>('#studio-main *')]
        .some((element) => /(auto|scroll)/.test(getComputedStyle(element).overflowY)
          && element.scrollHeight > element.clientHeight + 2),
    }));
    expect(overflow).toEqual({ horizontal: false, nested: false });
    await page.screenshot({ path: path.join(evidenceRoot, `${view.name}.png`), fullPage: true });
  }

  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(`${baseURL}/#/simulations/new/about`, { waitUntil: 'networkidle' });
  const motionState = await page.locator('#studio-main').evaluate((element) => ({
    mediaMatches: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    transitionDuration: getComputedStyle(element).transitionDuration,
    animationDuration: getComputedStyle(element).animationDuration,
  }));
  expect(motionState.mediaMatches).toBe(true);
  const durations = `${motionState.transitionDuration},${motionState.animationDuration}`
    .split(',')
    .filter(Boolean)
    .map((value) => value.endsWith('ms') ? Number.parseFloat(value) / 1000 : Number.parseFloat(value));
  expect(durations.every((value) => Number.isFinite(value) && value <= 0.001)).toBe(true);

  const skipLink = page.getByRole('link', { name: '본문으로 바로가기' });
  const routeBeforeSkip = page.url();
  await skipLink.focus();
  await expect(skipLink).toBeFocused();
  await page.keyboard.press('Enter');
  await expect.poll(() => page.evaluate(() => ({
    id: document.activeElement?.id ?? '',
    tag: document.activeElement?.tagName ?? '',
    name: document.activeElement?.getAttribute('aria-label') ?? '',
  }))).toEqual({ id: 'studio-main', tag: 'MAIN', name: '' });
  await expect(page).toHaveURL(routeBeforeSkip);
  await expect(page.getByRole('heading', { name: '이 시뮬레이션은 무엇을 검증하나요?' })).toBeVisible();
  await page.screenshot({ path: path.join(evidenceRoot, 'interaction-pc-dark-reduced-motion.png'), fullPage: true });

  const layout = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(layout.scrollWidth).toBeLessThanOrEqual(layout.clientWidth);
});

test('핵심 흐름을 light·dark WCAG A·AA·AAA 규칙으로 검사한다', async ({ page, baseURL }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const auditRoutes = [
    '#/simulations',
    '#/simulations/new/about',
    '#/simulations/new/policy',
    '#/simulations/BASE/overview',
    '#/simulations/BASE/agents',
    '#/simulations/BASE/visualization',
    '#/simulations/BASE/reports',
  ];
  const auditTags = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'wcag2aaa'];
  const audits: Array<Record<string, unknown>> = [];
  const violations: Array<Record<string, unknown>> = [];

  for (const theme of ['light', 'dark'] as const) {
    await page.goto(`${baseURL}/#/simulations`, { waitUntil: 'networkidle' });
    await page.evaluate((nextTheme) => localStorage.setItem('simconsole.theme', nextTheme), theme);
    await page.reload({ waitUntil: 'networkidle' });

    for (const hash of auditRoutes) {
      await page.goto(`${baseURL}/${hash}`, { waitUntil: 'networkidle' });
      // Framer Motion의 진입 opacity가 끝나기 전에 대비를 측정하면 실제 정지
      // 화면보다 낮은 합성 색상을 검사하게 된다.
      await page.waitForTimeout(400);
      await expect.poll(() => page.evaluate(() => document.documentElement.dataset.theme)).toBe(theme);
      const result = await new AxeBuilder({ page })
        .withTags(auditTags)
        .analyze();
      audits.push({
        theme,
        hash,
        passes: result.passes.length,
        incomplete: result.incomplete.length,
        inapplicable: result.inapplicable.length,
        violations: result.violations.length,
      });
      for (const violation of result.violations) {
        violations.push({
          theme,
          hash,
          id: violation.id,
          impact: violation.impact,
          help: violation.help,
          nodes: violation.nodes.map((node) => ({ target: node.target, summary: node.failureSummary })),
        });
      }
    }
  }

  await writeFile(
    path.join(evidenceRoot, 'axe-wcag-report.json'),
    JSON.stringify({ generatedAt: new Date().toISOString(), tags: auditTags, audits, violations }, null, 2),
    'utf8',
  );
  expect(violations).toEqual([]);
});
