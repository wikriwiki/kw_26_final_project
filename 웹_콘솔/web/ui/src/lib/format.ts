/**
 * 표시 포맷 — 세 화면이 같은 숫자를 다르게 찍지 않도록 여기에만 둔다.
 * 값이 없을 때는 절대 0 이나 임의값을 만들지 않고 EMPTY('—')를 돌려준다.
 */

export const EMPTY = '—';

const nf = new Intl.NumberFormat('ko-KR');
const nf1 = new Intl.NumberFormat('ko-KR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });

function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** 시작일과 끝일 → "2020-04-13 ~ 05-10". 해가 같으면 끝일의 해를 뺀다 */
export function dateRange(start: string | null | undefined, end: string | null | undefined): string {
  if (!start) return EMPTY;
  if (!end || end === start) return start;
  return `${start} ~ ${end.slice(0, 4) === start.slice(0, 4) ? end.slice(5) : end}`;
}

/** 시작일에서 일수만큼 간 마지막 날 (시작일 포함) */
export function lastDay(start: string, days: number): string {
  const end = new Date(`${start}T00:00:00Z`);
  end.setUTCDate(end.getUTCDate() + days - 1);
  return end.toISOString().slice(0, 10);
}

/** 시작일과 일수 → "2020-04-13 ~ 05-10 · 28일" */
export function dayRange(start: string | null | undefined, days: number | null | undefined): string {
  if (!start || !days) return start ?? EMPTY;
  return `${dateRange(start, lastDay(start, days))} · ${days}일`;
}

/** 1234567 → "1,234,567" */
export function int(value: number | null | undefined): string {
  return isNum(value) ? nf.format(Math.round(value)) : EMPTY;
}

/** 소수 자리 지정 */
export function dec(value: number | null | undefined, digits = 1): string {
  if (!isNum(value)) return EMPTY;
  return new Intl.NumberFormat('ko-KR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

/** 0.734 → "73.4%" (ratio 는 0~1) */
export function percent(ratio: number | null | undefined, digits = 1): string {
  if (!isNum(ratio)) return EMPTY;
  return `${dec(ratio * 100, digits)}%`;
}

/** 원 단위 금액 → "1억 2,340만원" / "8,500원" */
export function krw(value: number | null | undefined): string {
  if (!isNum(value)) return EMPTY;
  const abs = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  // 100만원 미만은 원 단위 그대로 — 19,189원을 "1만원"으로 내리면 하루 소비 같은 작은 금액이 틀려진다.
  // 그 이상은 만원 단위로 반올림한다(억 단위는 "억 만원")
  if (abs < 1e6) return `${sign}${nf.format(Math.round(abs))}원`;
  const totalMan = Math.round(abs / 1e4);
  const eok = Math.floor(totalMan / 1e4);
  const man = totalMan % 1e4;
  if (eok === 0) return `${sign}${nf.format(man)}만원`;
  return man > 0 ? `${sign}${nf.format(eok)}억 ${nf.format(man)}만원` : `${sign}${nf.format(eok)}억원`;
}

/** 초 → "2시간 42분" / "3분 12초" / "48초" */
export function duration(seconds: number | null | undefined): string {
  if (!isNum(seconds) || seconds < 0) return EMPTY;
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}시간 ${m}분`;
  if (m > 0) return `${m}분 ${sec}초`;
  return `${sec}초`;
}

/** ISO 문자열 → "08-02 18:59" */
export function shortTime(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return EMPTY;
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

/** ISO 문자열 → "2026-08-02 18:59:07" */
export function dateTime(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return EMPTY;
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(
    d.getMinutes(),
  )}:${p(d.getSeconds())}`;
}

/** 바이트 → "18.7MB" */
export function bytes(value: number | null | undefined): string {
  if (!isNum(value)) return EMPTY;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = value;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${i === 0 ? Math.round(v) : nf1.format(v)}${units[i]}`;
}

/** 그래프 축 눈금용 짧은 원화 표기 — "2.5억", "8,000만", "500만". 본문 숫자에는 krw 를 쓴다 */
export function krwAxis(value: number): string {
  const abs = Math.abs(value);
  const trim = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(1).replace(/\.0$/, ''));
  if (abs >= 1e8) return `${trim(value / 1e8)}억`;
  if (abs >= 1e4) return `${Math.round(value / 1e4).toLocaleString('ko-KR')}만`;
  return Math.round(value).toLocaleString('ko-KR');
}
