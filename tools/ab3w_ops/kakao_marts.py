# 서울 대형마트·SSM 체인 매장 수집(2026-10-10). 기존 scripts/scrape/kakao/client.py(무인증 카카오맵 검색)를 그대로 쓴다.
import json, sys, time
sys.path.insert(0, "/data/repo_ab3w_20261009/scripts/scrape")
from kakao.client import KakaoClient
GUS = "종로구 중구 용산구 성동구 광진구 동대문구 중랑구 성북구 강북구 도봉구 노원구 은평구 서대문구 마포구 양천구 강서구 구로구 금천구 영등포구 동작구 관악구 서초구 강남구 송파구 강동구".split()
BRANDS = ["이마트", "롯데마트", "하나로마트", "하나로클럽", "GS더프레시", "홈플러스", "트레이더스", "롯데슈퍼", "이마트에브리데이"]
c = KakaoClient(mean_pace=0.8)
out = {}
for gu in GUS:
    for b in BRANDS:
        try:
            res = c.search(query=f"서울 {gu} {b}")
        except Exception as e:
            print("ERR", gu, b, repr(e)[:80]); continue
        for p in res or []:
            pid = p.get("confirmid") or p.get("id")
            if pid and pid not in out:
                out[pid] = {k: p.get(k) for k in ("confirmid", "name", "address", "new_address", "cate_name_depth1", "cate_name_depth2", "cate_name_depth3", "last_cate_name", "x", "y", "lat", "lon", "lng", "is_new_open", "brandName")}
                out[pid]["q"] = f"{gu} {b}"
    print(gu, len(out), flush=True)
json.dump(out, open("/data/ab3w/audit_tools/kakao_marts_20261010.json", "w"), ensure_ascii=False, indent=0)
print("done", len(out))
