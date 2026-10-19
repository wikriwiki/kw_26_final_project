"""AI API 호출 로그 — 가이드 형식과 키 차단을 네트워크 없이 확인한다."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from scripts import ai_call_log, ai_call_log_export
from scripts.report import llm
from web.api import interview

FAKE_KEY = "flp_ABCDEFGH12345678secret"
RESPONSE = {
    "id": "chatcmpl-1",
    "model": "dep-test",
    "choices": [{"message": {"role": "assistant", "content": "네, 쿠폰으로 장을 봤어요."}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_urlopen(request, timeout=None):
    return _Response(json.dumps(RESPONSE).encode("utf-8"))


class LogDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        patcher = mock.patch.dict(os.environ, {"AI_CALL_LOG_DIR": self.temp.name, "AI_CALL_LOG": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)

    def rows(self, feature: str) -> list[dict]:
        path = self.dir / f"{feature}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RecordTests(LogDirCase):
    def test_one_line_holds_every_required_field(self) -> None:
        ai_call_log.record(
            "기능",
            endpoint="https://api.friendli.ai/dedicated/v1/chat/completions",
            model="dep-test",
            request_body={"messages": [{"role": "user", "content": "안녕"}]},
            requested_at=ai_call_log.now_kst(),
            response_body=RESPONSE,
        )
        (row,) = self.rows("기능")
        self.assertEqual(set(row["request"]), {"requested_at", "endpoint", "model", "body"})
        self.assertEqual(set(row["response"]), {"responded_at", "body"})
        self.assertTrue(row["request"]["requested_at"].endswith("+09:00"))
        self.assertEqual(row["response"]["body"]["usage"]["total_tokens"], 15)

    def test_secrets_are_masked_but_structure_stays(self) -> None:
        masked = ai_call_log.mask_secrets(
            {"api_key": "x", "messages": [{"content": f"키는 {FAKE_KEY} 이고 Bearer abc.def"}]}
        )
        self.assertEqual(masked["api_key"], ai_call_log.MASK)
        self.assertNotIn(FAKE_KEY, json.dumps(masked))
        self.assertNotIn("abc.def", json.dumps(masked))
        self.assertIn("키는", masked["messages"][0]["content"])

    def test_disabled_writes_nothing(self) -> None:
        with mock.patch.dict(os.environ, {"AI_CALL_LOG": "0"}):
            self.assertIsNone(
                ai_call_log.record("기능", endpoint="e", model="m", request_body={}, requested_at="t")
            )
        self.assertFalse((self.dir / "기능.jsonl").exists())


class CallSiteTests(LogDirCase):
    def test_interview_dedicated_call_is_logged_without_the_key(self) -> None:
        env = {
            "INTERVIEW_BASE_URL": "https://api.friendli.ai/dedicated/v1",
            "INTERVIEW_MODEL": "dep-test",
            "INTERVIEW_API_KEY": FAKE_KEY,
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(interview, "load_dotenv"), \
                mock.patch.object(interview.urllib.request, "urlopen", fake_urlopen):
            answer = interview.ask({"agent_id": "A1"}, "쿠폰 썼어요?")
        self.assertEqual(answer["answer"], "네, 쿠폰으로 장을 봤어요.")
        (row,) = self.rows(interview.FEATURE)
        self.assertEqual(row["request"]["endpoint"], "https://api.friendli.ai/dedicated/v1/chat/completions")
        self.assertEqual(row["request"]["body"]["messages"][-1]["content"], "쿠폰 썼어요?")
        self.assertEqual(row["response"]["body"]["id"], "chatcmpl-1")
        self.assertNotIn(FAKE_KEY, (self.dir / f"{interview.FEATURE}.jsonl").read_text(encoding="utf-8"))

    def test_report_commentary_call_is_logged_under_its_feature(self) -> None:
        env = {
            "REPORT_LLM_PROVIDER": "exaone",
            "REPORT_LLM_BASE_URL": "https://api.friendli.ai/dedicated/v1",
            "REPORT_LLM_MODEL": "dep-test",
            "REPORT_LLM_API_KEY": FAKE_KEY,
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(llm.urllib.request, "urlopen", fake_urlopen):
            result = llm.complete("system", "user", load_env=False)
        self.assertTrue(result.ok)
        (row,) = self.rows(llm.DEFAULT_FEATURE)
        self.assertEqual(row["request"]["model"], "dep-test")
        self.assertNotIn(FAKE_KEY, json.dumps(row, ensure_ascii=False))


class ExportTests(LogDirCase):
    def write(self, feature: str, *, at: str, endpoint: str = "https://api.friendli.ai/dedicated/v1/chat/completions",
              error: str | None = None) -> None:
        ai_call_log.record(
            feature, endpoint=endpoint, model="dep-test", request_body={"n": at}, requested_at=at,
            response_body=None if error else RESPONSE, error=error,
        )

    def export(self, *args: str) -> tuple[int, Path]:
        out = self.dir / "out.zip"
        code = ai_call_log_export.main(["--log-dir", str(self.dir), "--out", str(out), *args])
        return code, out

    def test_latest_successful_call_after_the_cutoff_per_feature(self) -> None:
        self.write("1대1_인터뷰", at="2026-10-13T23:59:00.000+09:00")
        self.write("1대1_인터뷰", at="2026-10-15T10:00:00.000+09:00")
        self.write("1대1_인터뷰", at="2026-10-16T10:00:00.000+09:00", error="HTTP 500")
        self.write("연결_점검", at="2026-10-16T10:00:00.000+09:00")
        code, out = self.export()
        self.assertEqual(code, 0)
        with zipfile.ZipFile(out) as bundle:
            self.assertEqual(bundle.namelist(), ["1대1_인터뷰.json"])
            document = json.loads(bundle.read("1대1_인터뷰.json").decode("utf-8"))
        self.assertEqual(document["records"][0]["request"]["body"], {"n": "2026-10-15T10:00:00.000+09:00"})

    def test_foreign_endpoint_stops_the_export(self) -> None:
        self.write("보고서_해설", at="2026-10-15T10:00:00.000+09:00",
                   endpoint="https://generativelanguage.googleapis.com/v1beta/models/x:generateContent")
        code, out = self.export()
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
