"""판정 세 자가 잡음을 실제로 걷어내는지 못 박는다.

세 자 모두 '잡음을 성적으로 바꾸지 않는' 장치다. 그래서 각 자가
**잡음에는 반응하지 않고 신호에만 반응**하는지를 확인한다.
"""
from __future__ import annotations

from collections import defaultdict

from scripts.report.score_p012_two_arm import (
    gap_ci, ratio_ci, region_evenness, sign_test_p,
)


def sec(values: dict[str, dict[str, float]]):
    """원장 집계는 defaultdict 로 들고 다닌다 — 없는 업종은 0 이다."""
    out = defaultdict(lambda: defaultdict(float))
    for aid, row in values.items():
        for k, v in row.items():
            out[aid][k] = float(v)
    return out


def test_sign_test_discards_ties_and_uses_discordant_pairs_only():
    # 동점이 몇 개 붙어도 p 는 바뀌지 않는다 — 유효표본은 엇갈린 쌍뿐이다.
    assert abs(sign_test_p(9, 1) - 0.021484375) < 1e-9
    assert sign_test_p(6, 6) == 1.0
    assert sign_test_p(0, 0) is None
    # 한쪽으로 완전히 쏠리면 작아지고, 표본이 커질수록 더 작아진다.
    assert sign_test_p(12, 0) < sign_test_p(6, 0)


def test_sign_confidence_is_one_for_signal_and_uncertain_for_balanced_noise():
    # 모든 사람이 오른 경우 — 부호가 뒤집힐 재표집이 없다.
    pct, lo, hi, conf = ratio_ci([100.0] * 20, [110.0] * 20, n=800)
    assert abs(pct - 10.0) < 1e-9 and conf == 1.0 and lo > 0

    # 정확히 반반 엇갈린 잡음 — 합은 같고, 부호는 재표집마다 뒤집힌다.
    off = [1000.0] * 40
    on = [1000.0 * (1.2 if i % 2 else 0.8) for i in range(40)]
    pct, lo, hi, conf = ratio_ci(off, on, n=1500)
    assert abs(pct) < 1e-9          # 점추정이 0 이다
    assert lo < 0 < hi              # 구간이 0 을 지난다
    assert conf < 0.975             # 방향을 주장할 수 없다


def test_rank_gap_reads_which_side_rose_more_not_the_levels():
    aids = ["a", "b", "c", "d"]
    # A 는 두 배로 오르고 B 는 그대로. 수준은 B 가 크지만 격차는 A 쪽이다.
    off = sec({a: {"kdi:x": 10.0, "kdi:y": 1000.0} for a in aids})
    on = sec({a: {"kdi:x": 20.0, "kdi:y": 1000.0} for a in aids})
    g, lo, hi, conf = gap_ci(off, on, {}, {}, aids, "kdi:x", "kdi:y", n=600)
    assert abs(g - 100.0) < 1e-9
    assert conf == 1.0 and lo > 0


def test_region_evenness_calls_a_uniform_response_even_and_a_planted_gap_uneven():
    # 자치구가 둘인데 반응이 같다 — '고르다'(무작위 분할과 구분되지 않는다).
    aids = (["AGT_11110001_M_30대_%03d" % i for i in range(12)]
            + ["AGT_11140001_M_30대_%03d" % i for i in range(12)])
    off = sec({a: {"total": 1000.0} for a in aids})
    on = sec({a: {"total": 1100.0} for a in aids})
    ngu, spread, p95, p = region_evenness(off, on, aids, n=300)
    assert ngu == 2 and spread is not None and p is not None and p >= 0.05

    # 구 하나만 크게 올린다 — 퍼짐이 영분포를 넘어야 한다.
    on = sec({a: {"total": (3000.0 if a.startswith("AGT_11110") else 1000.0)}
              for a in aids})
    ngu, spread, p95, p = region_evenness(off, on, aids, n=300)
    assert ngu == 2 and p is not None and p < 0.05


def test_excluded_sector_series_reads_the_new_eligible_breakdown(tmp_path):
    """K10('제외업종 중 유통') 은 업종 총액에서 적립분을 뺀 값이다."""
    import io as _io
    import json as _json

    from scripts.report.score_p012_two_arm import load_arm, series

    rows = [{"aid": "a", "day": "2021-10-01", "arm": "off", "total_spent": 100,
             "sangsaeng_eligible_offline_spent": 40, "online_spent": 0,
             "offline_spent": 100,
             "by_l1": {"편의점": 100}, "eligible_by_l1": {"편의점": 40},
             "by_sub": {}, "eligible_by_sub": {}}]
    d = tmp_path
    for arm in ("off", "on"):
        with _io.open(d / ("%s.sector.ledger.jsonl" % arm), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(_json.dumps(dict(r, arm=arm), ensure_ascii=False) + "\n")
        _io.open(d / ("%s.cashback.ledger.jsonl" % arm), "w", encoding="utf-8").write(
            _json.dumps({"aid": "a", "day": "2021-10-01"}) + "\n")

    s, c = load_arm(d, "off")
    assert s["a"]["_has_elig_sub"] == 1.0          # 새 원장임을 알아본다
    assert series(s, c, ["a"], "kdi:유통") == [100.0]
    assert series(s, c, ["a"], "exclkdi:유통") == [60.0]   # 100 - 40
