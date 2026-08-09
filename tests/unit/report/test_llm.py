"""해설 LLM 클라이언트 검사 — 네트워크를 쓰지 않는다.

가장 중요한 것은 **숫자 가드**다. 모델이 계산 결과에 없는 숫자를 쓰면
그 문단을 채택하지 않는다는 규칙이 실제로 동작하는지 확인한다.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.report import llm, narrator


ENV_KEYS = (
    "REPORT_LLM_PROVIDER",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "REPORT_LLM_BASE_URL",
    "REPORT_LLM_MODEL",
    "REPORT_LLM_API_KEY",
)


def clean_env(**overrides: str):
    values = {key: "" for key in ENV_KEYS}
    values.update(overrides)
    return mock.patch.dict(os.environ, values, clear=False)


class DotenvTests(unittest.TestCase):
    def test_env_file_is_parsed_without_clobbering_real_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".env"
            path.write_text(
                '# 주석\nGEMINI_API_KEY="abc123"\nexport REPORT_LLM_MODEL=some-model\nBROKEN LINE\n',
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "already-set"}, clear=False):
                loaded = llm.load_dotenv([path])
                self.assertEqual(loaded["GEMINI_API_KEY"], "abc123")
                self.assertEqual(loaded["REPORT_LLM_MODEL"], "some-model")
                # 이미 있는 환경변수는 덮어쓰지 않는다
                self.assertEqual(os.environ["GEMINI_API_KEY"], "already-set")


class ProviderStatusTests(unittest.TestCase):
    def test_no_key_is_reported_as_unconfigured_with_a_reason(self) -> None:
        with clean_env():
            status = llm.provider_status(load_env=False)
        self.assertEqual(status["provider"], "none")
        self.assertFalse(status["configured"])
        self.assertIn(".env", status["reason"])

    def test_gemini_key_switches_the_provider(self) -> None:
        with clean_env(GEMINI_API_KEY="test-key"):
            status = llm.provider_status(load_env=False)
        self.assertEqual(status["provider"], "gemini")
        self.assertTrue(status["configured"])
        self.assertEqual(status["model"], llm.GEMINI_DEFAULT_MODEL)

    def test_status_never_leaks_the_key_value(self) -> None:
        with clean_env(GEMINI_API_KEY="super-secret-value"):
            status = llm.provider_status(load_env=False)
        self.assertNotIn("super-secret-value", str(status))
        self.assertTrue(status["key_present"]["GEMINI_API_KEY"])

    def test_openai_compatible_needs_both_url_and_model(self) -> None:
        with clean_env(REPORT_LLM_BASE_URL="http://localhost:30000/v1"):
            status = llm.provider_status(load_env=False)
        self.assertEqual(status["provider"], "openai")
        self.assertFalse(status["configured"])
        with clean_env(REPORT_LLM_BASE_URL="http://localhost:30000/v1", REPORT_LLM_MODEL="m"):
            status = llm.provider_status(load_env=False)
        self.assertTrue(status["configured"])

    def test_complete_without_a_provider_returns_a_result_not_an_exception(self) -> None:
        with clean_env():
            result = llm.complete("s", "u", load_env=False)
        self.assertFalse(result.ok)
        self.assertEqual(result.provider, "none")
        self.assertTrue(result.error)


class NumericGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed = llm.allowed_number_set({"did": 281913.12, "growth": 0.2189})

    def test_numbers_from_the_computation_pass(self) -> None:
        ok, offenders = llm.numeric_guard("이중차분 추정치는 281,913원입니다.", self.allowed)
        self.assertTrue(ok, offenders)

    def test_rounded_and_abbreviated_forms_pass(self) -> None:
        # 28.19만 == 281913.12 / 1e4 를 반올림한 표현
        ok, _ = llm.numeric_guard("약 28.19만원 증가했습니다.", self.allowed)
        self.assertTrue(ok)
        ok, _ = llm.numeric_guard("반사실 대비 21.89% 입니다.", self.allowed)
        self.assertTrue(ok)

    def test_invented_numbers_are_caught(self) -> None:
        ok, offenders = llm.numeric_guard("소비가 무려 4,821,777원 늘었습니다.", self.allowed)
        self.assertFalse(ok)
        self.assertIn("4,821,777", offenders)

    def test_small_counting_integers_are_allowed(self) -> None:
        ok, _ = llm.numeric_guard("3개 업종에서 7일 동안 관측되었습니다.", self.allowed)
        self.assertTrue(ok)


class NarratorGuardTests(unittest.TestCase):
    """가드가 실패하면 그 문단을 버리고 그 사실을 기록하는지."""

    BUNDLE = {
        "meta": {"run_id": "T", "policy_id": "P777", "policy_name": "n", "policy_type": "grant",
                 "day_count": 8, "policy_from_used": "2025-07-18", "days": ["2025-07-18"]},
        "period": {"pre": ["2025-07-17"], "post": ["2025-07-18"], "policy_from": "2025-07-18",
                   "usable": True, "reason": None},
        "totals": {"amt": 100.0},
        "mix": {"policy_paid": 10.0},
        "did": {"did_absolute": 25.0, "did_pct_of_counterfactual": 25.0},
        "did_by_category": [],
        "categories": [],
        "daily": [],
        "overlay": {"available": False, "reason": "테스트", "overall": {}, "by_category": {}},
        "deciles": {},
        "event_study": {"available": False, "reason": "테스트", "points": []},
        "targets": {"categories": []},
        "control_categories": [],
    }
    CHECKS = {"counts": {"total": 1, "pass": 1, "fail": 0, "skip": 0}, "verdict": "ok", "checks": []}

    def test_hallucinated_numbers_are_rejected_and_recorded(self) -> None:
        fake = llm.LlmResult(True, text="소비가 987654321원 늘었습니다.", provider="gemini", model="m")
        with clean_env(GEMINI_API_KEY="k"), mock.patch.object(llm, "complete", return_value=fake), \
                mock.patch.object(narrator.llm_module, "complete", return_value=fake):
            narration = narrator.narrate_report(self.BUNDLE, self.CHECKS, sections=("did",))
        entry = narration["sections"]["did"]
        self.assertEqual(entry["guard"], "rejected")
        self.assertEqual(entry["source"], "deterministic")
        self.assertIn("did", narration["guard_rejected"])
        self.assertFalse(narration["used_llm"])

    def test_grounded_text_is_accepted(self) -> None:
        fake = llm.LlmResult(True, text="이중차분은 25원입니다.", provider="gemini", model="m")
        with clean_env(GEMINI_API_KEY="k"), mock.patch.object(narrator.llm_module, "complete", return_value=fake):
            narration = narrator.narrate_report(self.BUNDLE, self.CHECKS, sections=("did",))
        entry = narration["sections"]["did"]
        self.assertEqual(entry["guard"], "passed")
        self.assertTrue(narration["used_llm"])

    def test_llm_failure_falls_back_and_records_the_error(self) -> None:
        fail = llm.LlmResult(False, provider="gemini", model="m", error="HTTP 401")
        with clean_env(GEMINI_API_KEY="k"), mock.patch.object(narrator.llm_module, "complete", return_value=fail):
            narration = narrator.narrate_report(self.BUNDLE, self.CHECKS, sections=("did",))
        self.assertEqual(narration["sections"]["did"]["guard"], "llm_error")
        self.assertEqual(narration["errors"][0]["error"], "HTTP 401")


if __name__ == "__main__":
    unittest.main()
