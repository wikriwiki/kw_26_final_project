/**
 * 보고서 HTML 을 콘솔 페이지 안에 **그대로 펴 넣기** 위한 준비.
 *
 * 왜 iframe 을 걷어냈나
 * ---------------------
 * iframe 은 페이지 안에 또 하나의 스크롤 영역을 만든다. 바깥은 콘솔이,
 * 안쪽은 보고서가 따로 굴러서 어느 쪽을 굴리는지 알 수 없고, 문서 길이도
 * 밖에서 보이지 않는다 (SKILL §5 `scroll-behavior` — 중첩 스크롤 금지).
 *
 * 그래서 본문을 페이지에 직접 넣는다. 다만 보고서는 자기 CSS 를 들고 있고
 * 그 안에 `body`·`html`·`:root` 같은 전역 셀렉터가 있어서, 그대로 넣으면
 * 콘솔 전체가 보고서 스타일에 먹힌다. **모든 셀렉터를 래퍼 안으로 가둔다.**
 */

/** 보고서 본문을 감싸는 래퍼. 모든 규칙이 이 안에서만 산다 */
export const SCOPE = 'reportdoc';

/**
 * CSS 의 모든 셀렉터를 `.reportdoc` 아래로 밀어 넣는다.
 *
 * 정규식으로 규칙을 쪼개지 않고 중괄호 깊이를 세며 훑는다 —
 * `@media` 안에 규칙이 중첩돼 있어 단순 치환으로는 경계를 못 잡는다.
 */
export function scopeCss(css: string): string {
  const out: string[] = [];
  let i = 0;
  let buf = '';
  let depth = 0;

  const scopeSelector = (sel: string): string =>
    sel
      .split(',')
      .map((one) => {
        const s = one.trim();
        if (!s) return s;
        // 전역 셀렉터는 래퍼 자신으로 바꾼다. 콘솔 문서를 건드리면 안 된다
        if (/^(html|body|:root)$/i.test(s)) return `.${SCOPE}`;
        if (/^(html|body)\b/i.test(s)) return `.${SCOPE}${s.replace(/^(html|body)/i, '')}`;
        if (s === '*') return `.${SCOPE} *`;
        return `.${SCOPE} ${s}`;
      })
      .join(', ');

  while (i < css.length) {
    const ch = css[i];

    if (ch === '{') {
      if (depth === 0) {
        const head = buf.trim();
        // @media·@supports 는 껍데기라 셀렉터가 아니다. 그대로 두고 안쪽만 스코프한다
        out.push(head.startsWith('@') ? head + ' {' : scopeSelector(head) + ' {');
        buf = '';
        depth++;
        i++;
        continue;
      }
      depth++;
      buf += ch;
      i++;
      continue;
    }

    if (ch === '}') {
      depth--;
      if (depth === 0) {
        out.push(buf, '}');
        buf = '';
        i++;
        continue;
      }
      if (depth === 1) {
        // @media 안쪽 규칙 하나가 끝났다 — 그 덩어리를 재귀로 스코프한다
        out.push(scopeCss(buf.trim()), '}');
        buf = '';
        i++;
        continue;
      }
      buf += ch;
      i++;
      continue;
    }

    buf += ch;
    i++;
  }
  if (buf.trim()) out.push(buf);
  return out.join('\n');
}

export interface ReportDoc {
  css: string;
  html: string;
  title: string;
}

/**
 * 보고서를 받아 스코프된 CSS 와 본문 HTML 로 나눈다.
 * `DOMParser` 를 쓰므로 스크립트는 실행되지 않는다 — 문서를 읽는 화면에
 * 실행 가능한 코드를 들일 이유가 없다.
 */
export async function loadReportDoc(src: string): Promise<ReportDoc> {
  const res = await fetch(src);
  if (!res.ok) throw new Error(`보고서를 불러오지 못했습니다 (${res.status})`);
  const text = await res.text();

  const doc = new DOMParser().parseFromString(text, 'text/html');
  const css = [...doc.querySelectorAll('style')].map((s) => s.textContent ?? '').join('\n');
  doc.querySelectorAll('style, script, link').forEach((n) => n.remove());

  return {
    css: scopeCss(css),
    html: doc.body?.innerHTML ?? '',
    title: doc.title || '분석 보고서',
  };
}
