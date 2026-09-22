import { chromium } from '@playwright/test';
import fs from 'node:fs';
const OUT = process.argv[2]; fs.mkdirSync(OUT,{recursive:true});
const b = await chromium.launch();
const ctx = await b.newContext({viewport:{width:1440,height:900}});
const p = await ctx.newPage();
const errs=[]; const big=[];
p.on('pageerror',e=>errs.push(String(e).slice(0,100)));
p.on('console',m=>{if(m.type()==='error'&&!/about:blank/.test(m.text()))errs.push(m.text().slice(0,100));});
p.on('response',async r=>{try{const l=+(r.headers()['content-length']||0); if(l>500000) big.push(`${(l/1e6).toFixed(1)}MB ${r.url().split('/').pop()}`);}catch{}});
const rs=[['home','/'],['overview','/runs/BASE'],['agents','/runs/BASE/agents'],['report','/runs/BASE/report'],['results','/runs/BASE/results']];
for(const [n,path] of rs){
  await p.goto('http://localhost:5173'+path,{waitUntil:'networkidle',timeout:90000});
  await p.waitForTimeout(1200);
  const m=await p.evaluate(()=>({h1:(document.querySelector('h1')||{}).textContent?.trim().slice(0,30),
    nav:[...document.querySelectorAll('nav a')].map(a=>a.textContent.trim()).filter(Boolean),
    sw:document.documentElement.scrollWidth,iw:window.innerWidth,nodes:document.querySelectorAll('*').length}));
  console.log(n.padEnd(9), JSON.stringify(m));
  await p.screenshot({path:`${OUT}/${n}.png`,fullPage:true});
}
console.log('ERR:',[...new Set(errs)].slice(0,5));
console.log('BIG(>500KB):',[...new Set(big)].slice(0,6));
await b.close();
