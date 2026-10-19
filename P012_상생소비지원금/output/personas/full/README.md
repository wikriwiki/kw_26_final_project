# 전체 페르소나 (full generation)

3가지 방식으로 생성한 **전체** 페르소나. (10명 미리보기는 `../samples/` 참고)

| 파일 | 방식 | 개수 | 비고 |
|------|------|------|------|
| `A_rank_coupling_full.json.gz` | A · rank-coupling | 15,000 | gzip 압축(원본 88MB→4.8MB) |
| `B_conditional_graft_full.json` | B · conditional-graft | 120 | NVIDIA fixture 1명당 1개 |
| `C_hybrid_full.json` | C · hybrid | 120 | 봉합 29건, 잔여경고 8건 |

## 규모가 다른 이유

- **A(15,000)**: BDC 통계 셀(5,052개) 기반으로 생성하고 NVIDIA 서사를 SES 순위로 부착.
  현재 NVIDIA fixture가 120명뿐이라 **9,830명(66%)이 `sex_age` 폴백 매칭**이고
  서사 120개가 평균 ~125회 재사용됨 → 파일은 크지만 서사 다양성은 낮음.
- **B·C(각 120)**: NVIDIA 사람 1명에서 1개를 파생하므로 fixture 크기(120)가 곧 상한.

## 진짜 전체 규모로 키우려면

NVIDIA 서울 서브셋(약 13만)을 받아 `data/personas/nvidia_seoul_sample.json`를
교체하면 B·C는 그만큼, A는 `gu_sex_age` 매칭률이 크게 올라감.
→ `data/personas/README.md` 참고.

## 압축 해제 / 재생성

```bash
# A 압축 해제
gunzip -k output/personas/full/A_rank_coupling_full.json.gz

# 재생성 (전체)
python scripts/persona/build_rank_coupling.py --out output/personas/full/A_rank_coupling_full.json
python scripts/persona/build_conditional.py            --out output/personas/full/B_conditional_graft_full.json
python scripts/persona/build_conditional.py --reconcile --out output/personas/full/C_hybrid_full.json
```
