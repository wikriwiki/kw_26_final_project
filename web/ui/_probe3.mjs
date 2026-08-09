import { chromium } from '@playwright/test';
const b = await chromium.launch({ args: ['--use-gl=swiftshader','--enable-unsafe-swiftshader'] });
const p = await (await b.newContext({viewport:{width:1440,height:900}})).newPage();
await p.goto('http://localhost:5173/visualize',{waitUntil:'load',timeout:180000});
let f=null; for(let i=0;i<40&&!f;i++){await p.waitForTimeout(2000);f=p.frames().find(x=>x.url().includes('/viz/standalone.html'))||null;if(f){try{await f.evaluate(()=>window.Sim3D)}catch{f=null}}}
await p.waitForTimeout(8000);
console.log(JSON.stringify(await f.evaluate(()=>{
  const gu=document.querySelector('#toggle-gu-boundary-btn');
  const r=gu.getBoundingClientRect(); const cs=getComputedStyle(gu);
  const hud=document.querySelector('#top-hud'); const hr=hud.getBoundingClientRect();
  return {themeTag:!!document.getElementById('__console-theme__'),
    gu:{x:Math.round(r.left),y:Math.round(r.top),w:Math.round(r.width),h:Math.round(r.height),color:cs.color,bg:cs.backgroundColor,border:cs.borderColor,vis:cs.visibility,op:cs.opacity},
    hud:{x:Math.round(hr.left),y:Math.round(hr.top),w:Math.round(hr.width),h:Math.round(hr.height)},
    vw:window.innerWidth};
})));
await b.close();
