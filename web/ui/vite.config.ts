import { createReadStream, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * 시연용 3D 지도 서빙 — `docs/DESIGN_VIZ3D_INTEGRATION.md` 의 축소판.
 *
 * 설계도의 최종형은 run 별 JSON 을 API 로 받아 껍데기에 주입하는 구조지만,
 * 지금 필요한 것은 "한 케이스 시연"이므로 미리 구운 standalone 산출물을
 * 그대로 iframe 에 물린다. Neo4j 도 API 도 필요 없다.
 *
 * 이 파일은 `web/tools/extract_viz_json.py` 로 기존 Leaflet 산출물에서
 * 데이터 4종을 꺼낸 뒤 `scripts/sim/build_standalone_html.py` 로 구운 것이다.
 * deck.gl + maplibre 기반이라 기존 Leaflet 2D 산출물과 다르다.
 *
 * 127MB 파일을 public/ 으로 복사하면 저장소가 부풀고 빌드가 느려진다.
 * 원본을 스트리밍으로 서빙해서 사본을 만들지 않는다.
 */
const VIZ_FILE = resolve(__dirname, '../viz_store/demo/sim_standalone.html');
const VIZ_ROUTE = '/viz/standalone.html';

/**
 * 뷰어는 프레임 0(자정)에서 시작한다. 그 시각에는 전원이 집에 있어 지도에
 * 아무도 없고, 처음 여는 사람에게는 "고장난 화면"으로 보인다.
 * 실측: 120 프레임 중 0~5시(6개)만 비어 있고 08:00 프레임에 1,823 명이 있다.
 *
 * 산출물을 고치지 않고, 서빙할 때 뒤에 짧은 스크립트만 덧붙여 활동 시간대로
 * 옮긴다. HTML 파서는 문서 끝의 script 를 정상 실행한다.
 */
/**
 * 임베드 테마 — 뷰어 HUD 를 콘솔 디자인 시스템으로 덮어쓴다.
 * 원본 styles.css 는 수정 금지 대상이므로 뒤에 덧붙여 이긴다.
 * 파일을 매 요청 읽는다: 시안 조정 중 새로고침만으로 반영되게.
 */
const VIZ_THEME = resolve(__dirname, 'viz-embed.css');

function vizTail(): Buffer {
  let css = '';
  try {
    css = readFileSync(VIZ_THEME, 'utf8');
  } catch {
    /* 테마가 없으면 원본 모습 그대로 — 지도는 계속 뜬다 */
  }
  return Buffer.from(
    `\n<style id="__console-theme__">\n${css}\n</style>\n` +
      `<script>(function(){function go(){var S=window.Sim3D;if(!S||typeof S.setFrame!=="function"){return false;}` +
      `var tl=window.__TIMELINE__||[];var i=tl.findIndex(function(f){return (f.agents||[]).length>0;});` +
      `if(i>0){S.setFrame(i,0);var s=document.getElementById("frame-slider");if(s){s.value=String(i);}}return true;}` +
      `var n=0;var t=setInterval(function(){if(go()||++n>60){clearInterval(t);}},250);})();</script>\n`,
    'utf8',
  );
}

function vizStandalone(): Plugin {
  const handler = (req: { url?: string }, res: any, next: () => void) => {
    if (!req.url || !req.url.startsWith(VIZ_ROUTE)) return next();
    try {
      const { size } = statSync(VIZ_FILE);
      const tail = vizTail();
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      res.setHeader('Content-Length', String(size + tail.length));
      // 산출물 본체는 안 바뀌지만 테마는 조정 중이라 매번 새로 받는다.
      res.setHeader('Cache-Control', 'no-cache');
      const stream = createReadStream(VIZ_FILE);
      stream.on('end', () => res.end(tail));
      stream.pipe(res, { end: false });
    } catch {
      res.statusCode = 404;
      res.end('시각화 산출물을 찾을 수 없습니다. web/tools/extract_viz_json.py 로 먼저 생성하세요.');
    }
  };
  return {
    name: 'viz-standalone',
    configureServer: (s) => void s.middlewares.use(handler),
    configurePreviewServer: (s) => void s.middlewares.use(handler),
  };
}

/* ==========================================================================
   시연용 아티팩트 서빙 — `/artifacts/*`

   API 서버가 아직 없다. 그런데 실제 산출물은 이미 로컬에 있다:
     · 실행 산출물   C:\\Users\\srdyh\\gpu_exp_data\\20260802\\out_*  (12~48MB)
     · 최종 보고서   output/sim/report/FINAL_REPORT_5D_FULL.html      (601KB)

   시연에서 "내려받기"가 눌리지 않거나 보고서가 없는 것보다,
   실제 파일을 그대로 내보내는 편이 정직하고 완성도도 높다.
   API 가 붙으면 이 미들웨어를 걷어내고 프록시로 돌리면 된다.

   경로 밖으로 나가는 요청(`..`)은 막는다.
   ========================================================================== */
const ARTIFACT_ROOTS: Record<string, string> = {
  // run 산출물 — /artifacts/runs/<runId>/<파일경로>
  BASE: 'C:/Users/srdyh/gpu_exp_data/20260802/out_BASE',
  FINAL: 'C:/Users/srdyh/gpu_exp_data/20260802/out_FINAL',
  BASE7500: 'C:/Users/srdyh/gpu_exp_data/20260802/rescue/out_BASE7500',
};
const REPORT_FILE = resolve(__dirname, '../../output/sim/report/FINAL_REPORT_5D_FULL.html');

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.jsonl': 'application/x-ndjson; charset=utf-8',
};

function artifacts(): Plugin {
  const handler = (req: { url?: string }, res: any, next: () => void) => {
    const url = req.url ?? '';
    if (!url.startsWith('/artifacts/')) return next();

    const [pathOnly] = url.split('?');
    const rel = decodeURIComponent(pathOnly.slice('/artifacts/'.length));

    // 경로 탈출 차단 — `..` 이 섞인 요청은 받지 않는다
    if (rel.includes('..')) {
      res.statusCode = 400;
      return res.end('잘못된 경로입니다.');
    }

    let file: string | null = null;
    if (rel === 'report') {
      file = REPORT_FILE;
    } else if (rel.startsWith('runs/')) {
      const [, runId, ...rest] = rel.split('/');
      const root = ARTIFACT_ROOTS[runId];
      if (root && rest.length) file = resolve(root, rest.join('/'));
    }

    if (!file) return next();

    try {
      const { size } = statSync(file);
      const ext = file.slice(file.lastIndexOf('.'));
      res.setHeader('Content-Type', MIME[ext] ?? 'application/octet-stream');
      res.setHeader('Content-Length', String(size));
      // 완료된 실행의 산출물은 바뀌지 않는다
      res.setHeader('Cache-Control', 'public, max-age=3600');
      createReadStream(file).pipe(res);
    } catch {
      res.statusCode = 404;
      res.end('산출물을 찾을 수 없습니다: ' + rel);
    }
  };
  return {
    name: 'artifacts',
    configureServer: (s) => void s.middlewares.use(handler),
    configurePreviewServer: (s) => void s.middlewares.use(handler),
  };
}

// 콘솔 UI 빌드 설정.
// - `/api` 는 S2(FastAPI)로 프록시한다. 포트는 VITE_API_ORIGIN 으로 덮어쓸 수 있다.
// - 산출물은 web/ui/dist 에만 쓴다. web/api, web/fixtures 는 다른 조각의 소유이므로 건드리지 않는다.
const API_ORIGIN = process.env.VITE_API_ORIGIN ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react(), vizStandalone(), artifacts()],
  server: {
    port: 5173,
    strictPort: false,
    fs: {
      // 화면을 채우는 픽스처가 web/fixtures 에 있다. 읽기만 하며 수정하지 않는다.
      allow: ['..'],
    },
    proxy: {
      '/api': {
        target: API_ORIGIN,
        changeOrigin: true,
      },
      // 기존 3D 뷰 / 리포트 같은 정적 산출물 서빙 경로 (S5·S2 합의 대상)
      '/artifacts': {
        target: API_ORIGIN,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
  },
});
