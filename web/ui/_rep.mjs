import { chromium } from '@playwright/test';
const b=await chromium.launch();const p=await(await b.newContext({viewport:{width:1440,height:900}})).newPage();
const errs=[];p.on('pageerror',e=>errs.push(String(e).slice(0,90)));
await p.goto('http://localhost:5173/runs/BASE/report',{waitUntil:'networkidle',timeout:90000});
await p.waitForTimeout(2500);
console.log(JSON.stringify(await p.evaluate(()=>{
  const d=document.querySelector('.reportdoc');
  const nested=[...document.querySelectorAll('.reportdoc *')].filter(e=>e.scrollHeight>e.clientHeight+2&&getComputedStyle(e).overflowY!=='visible').length;
  const fixed=[...document.querySelectorAll('.reportdoc *')].filter(e=>getComputedStyle(e).position==='fixed').length;
  // 콘솔 스타일이 오염됐나 — 셸 배경·본문색 확인
  const bodyBg=getComputedStyle(document.body).backgroundColor;
  const h1=document.querySelector('.pagehead__title');
  return {docFound:!!d, docNodes:d?d.querySelectorAll('*').length:0, docH:d?Math.round(d.getBoundingClientRect().height):0,
    nestedScroll:nested, fixedInDoc:fixed, iframes:document.querySelectorAll('iframe').length,
    sw:document.documentElement.scrollWidth, iw:window.innerWidth,
    bodyBg, h1Color:h1?getComputedStyle(h1).color:null, h1Text:h1?.textContent};
})));
console.log('ERR:',[...new Set(errs)].slice(0,4));
await p.screenshot({path:process.argv[2],fullPage:false});
await b.close();
