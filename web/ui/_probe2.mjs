import { chromium } from '@playwright/test';
const b = await chromium.launch({ args: ['--use-gl=swiftshader','--enable-unsafe-swiftshader'] });
const p = await (await b.newContext({viewport:{width:1440,height:900}})).newPage();
await p.goto('http://localhost:5173/visualize',{waitUntil:'load',timeout:180000});
let f=null; for(let i=0;i<40&&!f;i++){await p.waitForTimeout(2000);f=p.frames().find(x=>x.url().includes('/viz/standalone.html'))||null;if(f){try{await f.evaluate(()=>window.Sim3D)}catch{f=null}}}
console.log(JSON.stringify(await f.evaluate(()=>{
  const hud=document.querySelector('#top-hud');
  return {
    children:[...hud.children].map(c=>({tag:c.tagName,id:c.id,cls:String(c.className).slice(0,30),disp:getComputedStyle(c).display,kids:[...c.children].map(k=>({tag:k.tagName,id:k.id,txt:(k.textContent||'').trim().slice(0,12),disp:getComputedStyle(k).display,w:Math.round(k.getBoundingClientRect().width)}))})),
    hudRect:(()=>{const r=hud.getBoundingClientRect();const cs=getComputedStyle(hud);return{l:Math.round(r.left),w:Math.round(r.width),overflow:cs.overflow,flexWrap:cs.flexWrap}})(),
  };
})));
await b.close();
