# `src/infra/llm/` — LLM 엔진 인프라

**vLLM/SGLang 엔진과의 통신을 담당.** 비즈니스 로직 없음, 순수 인프라.

## 예상 파일

| 파일 | 역할 |
|------|------|
| `engine_client.py` | vLLM/SGLang HTTP 클라이언트. 단일 요청/스트리밍 |
| `batch_controller.py` | Agent Batch Controller — 수백 에이전트 요청을 배치로 묶어 throughput 최적화 |
| `load_balancer.py` | 여러 엔진 인스턴스에 라운드로빈/로드기반 라우팅 |

## 규칙

- **이 폴더는 프롬프트 문자열을 모른다.** 호출자가 완성된 프롬프트를 넘김.
- **이 폴더는 도메인 모델을 모른다.** `str → str` 또는 `list[str] → list[str]` 인터페이스만.
- 캐시는 별도(`infra/cache/`) → 여기서 호출 직전에 prefix cache hit 시도

## 외부 의존성

- `httpx` (또는 `aiohttp`)
- `tenacity` (재시도)

## 교체 가능성

`engine_client.py` 인터페이스만 유지하면 vLLM → SGLang → TGI 교체 가능해야 함.
