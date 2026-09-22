/**
 * 비동기 데이터 상태 모델.
 *
 * 세 화면(S3·S4·S5)이 로딩/빈/에러/부분 데이터를 각자 다르게 그리면
 * 하나의 제품처럼 보이지 않는다. 그래서 상태 자체를 여기서 한 벌로 정의하고,
 * 화면은 `AsyncBoundary` 에 이 값을 넘기기만 한다.
 *
 * 중요: "부분 데이터"는 별도 status 가 아니라 ready + coverage 다.
 * rescue/ 같은 중단된 run 은 Day 0 만 있어도 그 Day 0 은 진짜 데이터이므로
 * 화면을 막지 않고 보여주되, 무엇이 없는지 배너로 명시한다.
 */

export type CoverageCell = 'present' | 'partial' | 'missing';

export interface DataCoverage {
  /** 실제로 존재하는 단위 수 (CONTRACT §3.1 `days_present`) */
  available: number;
  /**
   * 계약상 있어야 하는 단위 수 (CONTRACT §3.1 `days_planned`).
   * **null 은 0 이 아니라 "미확인"이다** (CONTRACT §4.1-2). rescue 처럼 summary.json 이
   * 없으면 계획 일수를 알 수 없고, 그때는 비율을 그리지 않는다.
   */
  expected: number | null;
  /** 단위 이름. 기본 '일' */
  unit?: string;
  /**
   * 단위별 상태. 길이는 expected 와 같아야 한다.
   * 생략하면 앞에서부터 available 개가 present 인 것으로 그린다.
   */
  cells?: CoverageCell[];
  /** 왜 불완전한지. 서버가 준 사유만 넣는다 — UI 가 지어내지 않는다. */
  reason?: string;
  /** 이 데이터가 끊긴 시각 등 부가 메타 (표시용 문자열) */
  note?: string;
  /**
   * expected 를 몰라도 불완전함이 확실할 때 (run status === "incomplete").
   * "중단됨"이지 "실패함"이 아니다 (CONTRACT §4.1-5).
   */
  partial?: boolean;
}

/** CONTRACT §3.1 runs.index 항목 → 커버리지. S4·S5 가 그대로 쓴다 */
export function coverageFromRun(
  run: { days_present: number; days_planned: number | null; status: 'completed' | 'incomplete' },
  extra?: { reason?: string; note?: string },
): DataCoverage {
  return {
    available: run.days_present,
    expected: run.days_planned,
    unit: '일',
    partial: run.status === 'incomplete',
    ...extra,
  };
}

export interface ConsoleError {
  /** 사용자에게 보여줄 한 줄 */
  message: string;
  /** 접었다 펼 수 있는 원문 (스택, 응답 바디 등) */
  detail?: string;
  /** HTTP status 또는 서버 에러 코드 */
  code?: string | number;
  /** 재시도 버튼을 띄울지 */
  retryable?: boolean;
}

export type AsyncState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'empty' }
  | { status: 'error'; error: ConsoleError }
  | { status: 'ready'; data: T; coverage?: DataCoverage };

export const idle = (): AsyncState<never> => ({ status: 'idle' });
export const loading = (): AsyncState<never> => ({ status: 'loading' });
export const empty = (): AsyncState<never> => ({ status: 'empty' });

export const failed = (error: ConsoleError | Error | string): AsyncState<never> => ({
  status: 'error',
  error: toConsoleError(error),
});

export const ready = <T,>(data: T, coverage?: DataCoverage): AsyncState<T> =>
  coverage ? { status: 'ready', data, coverage } : { status: 'ready', data };

export function toConsoleError(input: ConsoleError | Error | string): ConsoleError {
  if (typeof input === 'string') return { message: input, retryable: true };
  if (input instanceof Error) {
    return { message: input.message || '알 수 없는 오류', detail: input.stack, retryable: true };
  }
  return input;
}

/** coverage 가 실제로 불완전한가 */
export function isPartial(coverage?: DataCoverage): boolean {
  if (!coverage) return false;
  if (coverage.partial) return true;
  if (coverage.cells) return coverage.cells.some((c) => c !== 'present');
  if (coverage.expected == null) return false; // 미확인은 불완전이 아니다 — 모른다는 뜻이다
  return coverage.available < coverage.expected;
}

/** expected 를 모르면 비율을 만들지 않는다 (null 반환). 0 을 지어내지 않는다 */
export function coverageRatio(coverage: DataCoverage): number | null {
  if (!coverage.expected) return null;
  return Math.min(1, Math.max(0, coverage.available / coverage.expected));
}

export function coverageCells(coverage: DataCoverage): CoverageCell[] {
  if (coverage.cells) return coverage.cells;
  const total = Math.max(coverage.expected ?? 0, coverage.available);
  return Array.from({ length: total }, (_, i) =>
    i < coverage.available ? 'present' : 'missing',
  );
}

export function mapAsync<A, B>(state: AsyncState<A>, fn: (value: A) => B): AsyncState<B> {
  if (state.status !== 'ready') return state;
  return state.coverage
    ? { status: 'ready', data: fn(state.data), coverage: state.coverage }
    : { status: 'ready', data: fn(state.data) };
}
