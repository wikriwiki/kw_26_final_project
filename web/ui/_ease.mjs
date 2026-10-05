import { chromium } from '@playwright/test';
const b = await chromium.launch();
for (const rm of ['no-preference','reduce']) {
  const ctx = await b.newContext({ viewport:{width:1440,height:900}, reducedMotion: rm });
  const p = await ctx.newPage();
  await p.goto('http://localhost:5173/runs/BASE/results',{waitUntil:'networkidle'});
  const v = await p.evaluate(()=>{const cs=getComputedStyle(document.documentElement);
    return {ease:cs.getPropertyValue('--ease').trim(), easeOut:cs.getPropertyValue('--ease-out').trim(),
      dur:cs.getPropertyValue('--dur').trim(), durBase:cs.getPropertyValue('--dur-base').trim(),
      mainTrans:getComputedStyle(document.querySelector('.app__main')).transitionDuration};});
  console.log(rm, JSON.stringify(v));
  await ctx.close();
}
await b.close();
