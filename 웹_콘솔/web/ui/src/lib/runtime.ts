/** Build-time deployment mode. The API independently enforces the same rule. */
export const READ_ONLY = import.meta.env.VITE_READ_ONLY === 'true';
/** 공개 시연 배포: 인터뷰는 되고, 새 실행·보고서 생성·정책 수정·동기화는 안 된다. API 도 같은 규칙을 지킨다 */
export const DEMO = import.meta.env.VITE_DEMO === 'true';
/** 무언가를 새로 만들거나 바꿀 수 있는가 */
export const CAN_EDIT = !READ_ONLY && !DEMO;
