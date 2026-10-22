/**
 * 서울 점 지도 — 랜딩 페이지 표지와 같은 그림을 **실제 시뮬레이션 시민**으로 그린다.
 *
 * 점 하나가 시민 한 명의 집이다(3D 지도 팩의 `home_lon/lat`). 그중 몇 명씩 번갈아
 * 직장으로 갔다가 돌아온다(`work_lon/lat`). 직장 좌표가 없는 사람은 움직이지 않는다 —
 * 없는 출근길을 지어내지 않는다. 움직임 줄이기를 켠 사용자에게는 정지 화면만 보인다.
 *
 * 캔버스 하나, 의존성 없음. 좌표는 서버가 1e-4 도 정수로 준다(`/api/viz/packs/<팩>/residents`).
 */
import { useEffect, useRef } from 'react';

export interface Residents {
  count: number;
  origin: [number, number];
  scale: number;
  stride: number;
  points: Array<number | null>;
}

// 한강 — 설명용 중심선(랜딩 페이지와 같은 점들)
const RIVER: Array<[number, number]> = [
  [126.795, 37.592], [126.845, 37.572], [126.885, 37.552], [126.925, 37.534], [126.962, 37.519],
  [127.0, 37.516], [127.035, 37.524], [127.07, 37.53], [127.1, 37.532], [127.125, 37.545], [127.17, 37.562],
];

const INK = 'rgba(17,19,22,0.55)';
const SIM = '#1f3b5c';
const RIVER_COLOR = '#e8ebef';

export function CityDots({ data, label }: { data: Residents; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // 경위도 → 화면. 경도 1도와 위도 1도의 실제 거리 비(서울 위도)를 지켜 찌그러지지 않게 한다
    const n = Math.floor(data.points.length / data.stride);
    const lon = new Float32Array(n), lat = new Float32Array(n);
    const wlon = new Float32Array(n), wlat = new Float32Array(n);
    const hasWork = new Uint8Array(n);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i += 1) {
      const p = data.points;
      const o = i * data.stride;
      lon[i] = data.origin[0] + (p[o] as number) * data.scale;
      lat[i] = data.origin[1] + (p[o + 1] as number) * data.scale;
      if (p[o + 2] !== null && p[o + 3] !== null) {
        hasWork[i] = 1;
        wlon[i] = data.origin[0] + (p[o + 2] as number) * data.scale;
        wlat[i] = data.origin[1] + (p[o + 3] as number) * data.scale;
      }
      minX = Math.min(minX, lon[i]); maxX = Math.max(maxX, lon[i]);
      minY = Math.min(minY, lat[i]); maxY = Math.max(maxY, lat[i]);
    }
    const KX = 88.2, KY = 111; // 서울 위도에서 경도·위도 1도의 km
    let W = 0, H = 0, s = 1, ox = 0, oy = 0, dpr = 1;
    const fx = (x: number) => ox + (x - minX) * KX * s;
    const fy = (y: number) => oy + (maxY - y) * KY * s;

    function layout() {
      const rect = canvas!.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      W = rect.width; H = rect.height;
      canvas!.width = Math.round(W * dpr); canvas!.height = Math.round(H * dpr);
      const kmW = (maxX - minX) * KX, kmH = (maxY - minY) * KY;
      s = Math.min((W - 16) / kmW, (H - 16) / kmH);
      ox = (W - kmW * s) / 2; oy = (H - kmH * s) / 2;
    }

    // 지금 길 위에 있는 사람들 — 집에서 직장까지 갔다가 돌아온다
    type Trip = { i: number; t0: number; dur: number };
    let trips: Trip[] = [];
    const workers: number[] = [];
    for (let i = 0; i < n; i += 1) if (hasWork[i]) workers.push(i);
    let lastSpawn = 0;

    function draw(now: number) {
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx!.clearRect(0, 0, W, H);
      ctx!.beginPath();
      RIVER.forEach(([x, y], k) => (k ? ctx!.lineTo(fx(x), fy(y)) : ctx!.moveTo(fx(x), fy(y))));
      ctx!.strokeStyle = RIVER_COLOR;
      ctx!.lineWidth = Math.max(3, s * 0.55);
      ctx!.lineCap = 'round';
      ctx!.lineJoin = 'round';
      ctx!.stroke();

      const moving = new Set(trips.map((t) => t.i));
      ctx!.fillStyle = INK;
      const d = W > 520 ? 1.6 : 1.3;
      for (let i = 0; i < n; i += 1) if (!moving.has(i)) ctx!.fillRect(fx(lon[i]) - d / 2, fy(lat[i]) - d / 2, d, d);

      ctx!.fillStyle = SIM;
      for (const tr of trips) {
        const u = Math.min(1, (now - tr.t0) / tr.dur);
        // 갔다가 돌아온다 — 0→1→0, 양 끝에서 천천히
        const e = 0.5 - 0.5 * Math.cos(u * 2 * Math.PI);
        const x = lon[tr.i] + (wlon[tr.i] - lon[tr.i]) * e;
        const y = lat[tr.i] + (wlat[tr.i] - lat[tr.i]) * e;
        ctx!.beginPath();
        ctx!.arc(fx(x), fy(y), 2.2, 0, Math.PI * 2);
        ctx!.fill();
      }
    }

    let raf = 0;
    function tick(now: number) {
      if (workers.length && now - lastSpawn > 70 && trips.length < 90) {
        lastSpawn = now;
        trips.push({ i: workers[(Math.random() * workers.length) | 0], t0: now, dur: 5200 + Math.random() * 3000 });
      }
      trips = trips.filter((t) => now - t.t0 < t.dur);
      draw(now);
      raf = window.requestAnimationFrame(tick);
    }

    layout();
    if (reduce) draw(0);
    else raf = window.requestAnimationFrame(tick);

    const ro = new ResizeObserver(() => {
      layout();
      if (reduce) draw(0);
    });
    ro.observe(canvas);
    // 화면에 안 보이면 멈춘다 — 다른 탭에서 CPU 를 쓰지 않게
    const onVis = () => {
      if (reduce) return;
      if (document.hidden) window.cancelAnimationFrame(raf);
      else raf = window.requestAnimationFrame(tick);
    };
    document.addEventListener('visibilitychange', onVis);
    return () => {
      window.cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener('visibilitychange', onVis);
    };
  }, [data]);

  return <canvas ref={ref} className="citydots" role="img" aria-label={label} />;
}
