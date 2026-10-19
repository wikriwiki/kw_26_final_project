"""쌍체부호검정이 3,000명 규모에서도 넘치지 않고, 작은 표본의 정확값과 같다."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import score_p012_two_arm as S  # noqa: E402


def test_small_sample_exact_values_unchanged():
    assert S.sign_test_p(10, 0) == 2 / 1024
    assert S.sign_test_p(5, 5) == 1.0
    assert S.sign_test_p(0, 0) is None
    # 이전 공식(2.0**m)과 m < 1024 에서 같은 값
    up, dn = 300, 250
    m, k = up + dn, min(up, dn)
    old = min(1.0, 2.0 * sum(math.comb(m, i) for i in range(k + 1)) / (2.0 ** m))
    assert math.isclose(S.sign_test_p(up, dn), old, rel_tol=1e-12)


def test_three_thousand_people_does_not_overflow():
    p_even = S.sign_test_p(1500, 1500)
    assert p_even == 1.0
    p = S.sign_test_p(1600, 1400)          # 정규근사로 z≈3.65, p≈0.00026
    z = (1600 - 1500) / math.sqrt(3000 / 4)
    approx = math.erfc(z / math.sqrt(2))
    assert 0 < p < 0.001
    assert math.isclose(p, approx, rel_tol=0.1)
    assert S.sign_test_p(2990, 10) >= 0.0   # 아주 작은 p 는 0 으로 내려가도 넘치지 않는다
