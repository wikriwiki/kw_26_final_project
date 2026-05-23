# 페르소나 외부 데이터

## `nvidia_seoul_sample.json`

NVIDIA **Nemotron-Personas-Korea** 데이터셋에서 `province == "서울"` 인 레코드
120개를 streaming 으로 추출한 샘플.

- 출처: https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea
- 라이선스: **CC BY 4.0** (상업·비상업 사용 가능, NVIDIA attribution 필요)
- 용도: 빈약한 페르소나 정성 레이어 보강 (직업관·취미·가치관·가족 서사)

### 필드 (26)
정성 서사 7종(`professional/sports/arts/travel/culinary/family_persona` + `persona`),
구조화(`cultural_background`, `skills_and_expertise(_list)`, `hobbies_and_interests(_list)`,
`career_goals_and_ambitions`), 인구통계(`sex`, `age`, `marital_status`, `military_status`,
`family_type`, `housing_type`, `education_level`, `bachelors_field`, `occupation`,
`district`, `province`, `country`), `uuid`.

### 전체 데이터로 교체
실제 풀런에서는 1M 전체에서 서울 subset(약 13%, ~13만)을 받아 쓰면 매칭 풀이 커진다:

```python
from datasets import load_dataset
import json
ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train", streaming=True)
seoul = [r for r in ds if r.get("province") == "서울"]
json.dump(seoul, open("data/personas/nvidia_seoul_full.json","w",encoding="utf-8"),
          ensure_ascii=False)
```

로더는 `nvidia_seoul_full.json` 이 있으면 우선 사용, 없으면 `nvidia_seoul_sample.json` fallback.
