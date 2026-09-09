// Node >=22.13: run actual TypeScript functions with type erasure, no packages.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { stripTypeScriptTypes } from 'node:module';

const dataUrl = text => `data:text/javascript;base64,${Buffer.from(text).toString('base64')}`;
const format = stripTypeScriptTypes(await readFile(new URL('../src/lib/format.ts', import.meta.url), 'utf8'));
const source = await readFile(new URL('../src/lib/agentData.ts', import.meta.url), 'utf8');
const load = async text => import(dataUrl(stripTypeScriptTypes(text)
  .replace("from './format'", `from '${dataUrl(format)}'`)
  .replaceAll('import.meta.env.BASE_URL', "'/'")));
const optimized = await load(source);
const needle = 'value: totalsByDay.get(i) ?? 0,';
assert.ok(source.includes(needle));
const original = await load(source.replace(needle,
  'value: paid.filter((e) => e.d === i).reduce((s, e) => s + e.spent, 0),'));
const question = optimized.QUESTIONS.find(q => q.id === 'total');
assert.ok(question);

for (const nDays of [0, 1, 5, 30, 120]) {
  const agent = {days: Array.from({length: nDays}, (_, i) => `2021-day-${i}`),
    events: Array.from({length: 1000}, (_, i) => ({d: i % (nDays + 3),
      spent: [0, -10, 0.01, 1e16, 35.3][i % 5]}))};
  const snapshot = structuredClone(agent);
  const ctx = {dayIdx: 0, hour: 12};
  assert.deepEqual(optimized.answer(agent, question, ctx), original.answer(agent, question, ctx));
  assert.deepEqual(agent, snapshot);
}
const empty = {days: ['2021-09-06'], events: []};
assert.deepEqual(optimized.answer(empty, question, {dayIdx: 0, hour: 12}),
  original.answer(empty, question, {dayIdx: 0, hour: 12}));

// Count reads rather than asserting machine-dependent timing. The existing
// dayEvents lookup remains one scan; total-by-day no longer rescans per day.
function countDayReads(module, nDays) {
  let reads = 0;
  const events = Array.from({length: 1000}, (_, i) => ({spent: i + 1,
    get d() { reads++; return i % nDays; }}));
  module.answer({days: Array.from({length: nDays}, (_, i) => String(i)), events},
    question, {dayIdx: 0, hour: 12});
  return reads;
}
assert.equal(countDayReads(optimized, 100), countDayReads(optimized, 10));
assert.ok(countDayReads(original, 100) > 10 * countDayReads(optimized, 100));
console.log('PASS: full total-answer output, input immutability, empty/out-of-range days, float order, linear work');
