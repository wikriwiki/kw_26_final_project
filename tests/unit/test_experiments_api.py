"""실험 레지스트리·미러 집계·인터뷰·집단 질의·동기화 (web/api/experiment*.py)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts.sim.evidence_integrity import seal
from web.api import experiment_interview, experiment_query, experiment_sync, interview
from web.api.app import create_app
from web.api.experiments import ExperimentStore, read_day_rows
from web.api.runner import RunLock, Runner
from web.api.store import ArtifactStore, StoreError

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "web" / "fixtures"
EXP = "nsz_test"
AGENTS = ["AGT_11350560_M_30대_001", "AGT_11650510_F_40대_002", "AGT_11710520_M_20대_003"]

DEFINITION = {
    "experiment_id": EXP,
    "title": "테스트 금연 실험",
    "kind": "no_smoking_zone",
    "window": {"start": "2017-11-19", "days": 4, "effective_date": "2017-11-21"},
    "cohort_size": 3,
    "arms": [
        {"arm_id": "shared-pre", "role": "shared_pre", "label": "사전", "policy": "off", "directory": "pre", "start": "2017-11-19", "days": 2},
        {"arm_id": "post-off", "role": "control", "label": "OFF", "policy": "off", "parent": "shared-pre", "directory": "off", "start": "2017-11-21", "days": 2},
        {"arm_id": "post-on", "role": "treatment", "label": "ON", "policy": "on", "parent": "shared-pre", "directory": "on", "start": "2017-11-21", "days": 2},
    ],
    "analysis": {
        "facilities": {"billiard": "당구장", "indoor_golf": "실내골프"},
        "facility_aliases": {"screen_golf": "indoor_golf"},
        "districts": {"11350": "노원구", "11650": "서초구", "11710": "송파구"},
        "limitations": ["테스트"],
    },
    "bundle": {"directory": "bundle", "files": ["runtime.json", "personas.json"]},
    "remote": {"profile": "TEST_REMOTE", "results_root": "/srv/results", "bundles_root": "/srv"},
}

SMOKING = {AGENTS[0]: "smoker", AGENTS[1]: "non_smoker", AGENTS[2]: "smoker"}


def _row(aid: str, day: str, *, status: str = "ok", arm: str = "off", hits=(), spend: int = 10000, journal=None) -> dict:
    if status == "skipped":
        return seal({"aid": aid, "status": "skipped", "skip_kind": "failed_after_retries", "attempts": 6,
                     "experience_day": day, "observed_behavior": False,
                     "no_smoking": {"arm": arm, "smoking_status": SMOKING[aid], "observed_behavior": False}})
    receipts = [seal({"kind": "purchase_receipt", "event_id": f"EX_{aid[-3:]}_{day}", "scheduled_time": "12:00",
                      "category": "식사", "amount": 7000, "own_paid": 7000, "purchase_status": "purchased", "poi_id": "C_1"})]
    row = {
        "aid": aid, "status": "ok", "experience_day": day, "execution_receipts": receipts,
        "cm_today_total": spend, "cm_today_total_incl_online": spend * 2, "mood": 0.5, "fatigue": 0.4,
        "avg_sat": 0.6, "balance": 100000, "tokens_in": 10, "tokens_out": 2,
        "no_smoking": {"arm": arm, "policy_active": arm == "on", "smoking_status": SMOKING[aid],
                       "payment_count": sum(h[3] for h in hits), "revenue_krw": sum(h[2] for h in hits),
                       "by_poi": [{"poi_id": f"P_{i}", "facility_type": h[0], "district_code": h[1],
                                   "revenue_krw": h[2], "payment_count": h[3]} for i, h in enumerate(hits)]},
    }
    if journal:
        row["interview_evidence"] = journal
    return seal(row)


def _write_day(directory: Path, day: str, rows: list[dict], *, tail: bytes = b"") -> None:
    path = directory / "metrics" / f"day_{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n" for row in rows) + tail)


def build_fixture(root: Path) -> None:
    registry = root / "registry"
    registry.mkdir(parents=True)
    (registry / f"{EXP}.json").write_text(json.dumps(DEFINITION, ensure_ascii=False), encoding="utf-8")
    base = root / "mirror" / EXP
    arms = {name: base / "arms" / name for name in ("shared-pre", "post-off", "post-on")}
    for arm, directory in arms.items():
        directory.mkdir(parents=True)
        (directory / "experiment_run.json").write_text(
            json.dumps({"status": "running", "arm": "on" if arm == "post-on" else "off", "cohort_ids": AGENTS, "model": "M"}),
            encoding="utf-8",
        )
    # 저널 하나: 첫 사람의 첫날
    journal_record = seal({"schema_version": 1, "record_id": "EV_1", "kind": "agent_day", "day": "2017-11-19",
                           "executed_events": [{"time": "19:00", "category": "여가", "facility_type": "billiard",
                                                "policy_target": True, "actual_spent": 12000}],
                           "stated_rationales": [{"public_statement": {"intent": "친구와 당구", "reasoning": "[E0012]에서 친구 약속이 있었다"}}]})
    journal_path = arms["shared-pre"] / "evidence/v1/2017-11-19/abc.jsonl"
    journal_path.parent.mkdir(parents=True)
    journal_path.write_text(json.dumps(journal_record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    reference = {"schema_version": 1, "path": "evidence/v1/2017-11-19/abc.jsonl", "record_id": "EV_1",
                 "integrity_sha256": journal_record["integrity_sha256"]}

    _write_day(arms["shared-pre"], "2017-11-19", [
        _row(AGENTS[0], "2017-11-19", hits=[("billiard", "11350", 12000, 1)], journal=reference),
        _row(AGENTS[1], "2017-11-19"),
        _row(AGENTS[2], "2017-11-19", hits=[("screen_golf", "11710", 20000, 2)]),
    ])
    _write_day(arms["shared-pre"], "2017-11-20", [
        _row(AGENTS[0], "2017-11-20"),
        _row(AGENTS[1], "2017-11-20", status="skipped"),
        _row(AGENTS[2], "2017-11-20"),
    ])
    # 시행 후: OFF 이틀 완료, ON 첫날만. ON 첫날에 둘째 사람은 skipped → 짝 비교에서 빠진다
    _write_day(arms["post-off"], "2017-11-21", [
        _row(AGENTS[0], "2017-11-21", hits=[("billiard", "11350", 10000, 1)]),
        _row(AGENTS[1], "2017-11-21", hits=[("billiard", "11650", 9000, 1)]),
        _row(AGENTS[2], "2017-11-21"),
    ])
    _write_day(arms["post-off"], "2017-11-22", [_row(a, "2017-11-22") for a in AGENTS])
    tampered = _row(AGENTS[2], "2017-11-21", arm="on")
    tampered["cm_today_total"] = 999999  # 봉인과 다른 값 → 집계에서 빠진다
    _write_day(arms["post-on"], "2017-11-21", [
        _row(AGENTS[0], "2017-11-21", arm="on", hits=[("billiard", "11350", 15000, 2)]),
        _row(AGENTS[1], "2017-11-21", arm="on", status="skipped"),
        tampered,
    ], tail=b'{"aid": "AGT_partial"')
    bundle = base / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "runtime.json").write_text(json.dumps({"cohort": [{"id": a, "smoking_status": SMOKING[a]} for a in AGENTS]}), encoding="utf-8")
    personas = []
    for aid, gu, dong, age, sex in ((AGENTS[0], "노원구", "월계1동", 34, "M"), (AGENTS[1], "서초구", "서초1동", 45, "F"), (AGENTS[2], "송파구", "잠실동", 27, "M")):
        personas.append({"agent_id": aid, "residence": {"gu": gu, "dong": dong, "dong_code": aid.split("_")[1]},
                         "personal": {"age": age, "gender": sex, "age_group": f"{age // 10 * 10}대", "job": "회사원", "income_level": "중", "life_stage": "직장인"}})
    (bundle / "personas.json").write_text(json.dumps(personas, ensure_ascii=False), encoding="utf-8")


class _MirrorFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="exp-")
        self.root = Path(self.temp.name)
        build_fixture(self.root)
        self.env = patch.dict(os.environ, {"SIM_EXPERIMENTS_ROOT": str(self.root / "mirror")})
        self.env.start()
        self.store = ExperimentStore(registry_dir=self.root / "registry")

    def tearDown(self) -> None:
        self.env.stop()
        self.temp.cleanup()


class ExperimentStoreTests(_MirrorFixture):
    def test_rows_are_seal_checked_and_partial_tail_is_deferred(self) -> None:
        parsed = read_day_rows(self.root / "mirror" / EXP / "arms" / "post-on" / "metrics" / "day_2017-11-21.jsonl")
        self.assertEqual(sorted(parsed["rows"]), sorted(AGENTS[:2]))
        self.assertEqual(parsed["integrity_failures"], 1)
        self.assertGreater(parsed["partial_tail_bytes"], 0)

    def test_arm_status_phases_and_counts(self) -> None:
        detail = self.store.detail(EXP)
        arms = {arm["arm_id"]: arm for arm in detail["arms"]}
        self.assertEqual(arms["shared-pre"]["phase"], "complete")
        self.assertEqual(arms["shared-pre"]["skipped_total"], 1)
        self.assertEqual(arms["post-off"]["days_complete"], 2)
        self.assertEqual(arms["post-on"]["phase"], "running")
        self.assertEqual(arms["post-on"]["integrity_failures"], 1)
        self.assertTrue(arms["post-on"]["current_day"]["partial_tail"])
        self.assertTrue(detail["bundle_present"])
        listing = self.store.list()
        self.assertEqual(listing["total"], 1)
        self.assertNotIn("day_status", listing["items"][0]["arms"][0])

    def test_daily_totals_and_target_groups(self) -> None:
        series = {s["arm_id"]: s for s in self.store.daily(EXP)["series"]}
        first = series["shared-pre"]["points"][0]
        self.assertEqual(first["status"], {"ok": 3})
        self.assertEqual(first["totals"]["receipt_count"], 3)
        self.assertEqual(first["target"]["total"], {"revenue_krw": 32000, "payment_count": 3})
        # screen_golf 는 indoor_golf 로 묶인다
        self.assertEqual(first["target"]["groups"]["indoor_golf:smoker"], {"revenue_krw": 20000, "payment_count": 2})
        self.assertNotIn("agents", first)

    def test_comparison_uses_only_days_observed_in_both_arms(self) -> None:
        result = self.store.comparison(EXP)
        self.assertTrue(result["available"])
        self.assertEqual(result["status"], "interim_preview")
        self.assertEqual(result["post"]["days"], ["2017-11-21"])
        # 둘째 사람(ON skipped)과 셋째 사람(ON 봉인 불일치)은 빠지고 첫째만 짝이 된다
        self.assertEqual(result["post"]["observed_paired_agent_days"], 1)
        rows = {(c["group"], c["metric"]): c for c in result["comparisons"]}
        billiard = rows[("billiard", "revenue_krw")]
        self.assertEqual((billiard["off"], billiard["on"], billiard["on_minus_off"]), (10000, 15000, 5000))
        self.assertEqual(billiard["percent_change"], 50.0)
        self.assertEqual(rows[("indoor_golf", "payment_count")]["status"], "zero_baseline_not_estimable")
        self.assertEqual(result["pre"]["totals"]["billiard"], {"revenue_krw": 12000, "payment_count": 1})

    def test_roster_prefers_bundle_and_falls_back_to_metrics(self) -> None:
        roster = self.store.roster(EXP)
        self.assertEqual(roster["source"], "bundle")
        first = roster["items"][0]
        self.assertEqual((first["gu"], first["smoking_status"], first["district_code"]), ("노원구", "smoker", "11350"))
        (self.root / "mirror" / EXP / "bundle" / "personas.json").unlink()
        fallback = ExperimentStore(registry_dir=self.root / "registry").roster(EXP)
        self.assertEqual(fallback["source"], "metrics")
        self.assertEqual({a["gu"] for a in fallback["items"]}, {"노원구", "서초구", "송파구"})

    def test_agent_view_spans_arms(self) -> None:
        view = self.store.agent(EXP, AGENTS[0])
        arms = {arm["arm_id"]: arm for arm in view["arms"]}
        self.assertEqual(len(arms["shared-pre"]["days"]), 2)
        self.assertEqual(arms["post-on"]["days"][0]["no_smoking"]["by_poi"][0]["revenue_krw"], 15000)
        with self.assertRaises(StoreError):
            self.store.agent(EXP, "AGT_nobody")

    def test_registry_rejects_bad_definitions(self) -> None:
        (self.root / "registry" / "broken.json").write_text(json.dumps({"experiment_id": "Bad Id"}), encoding="utf-8")
        listing = ExperimentStore(registry_dir=self.root / "registry").list()
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["problems"][0]["file"], "broken.json")


class InterviewTests(_MirrorFixture):
    def test_packet_exposes_records_up_to_the_chosen_day(self) -> None:
        experiment = self.store.experiment(EXP)
        packet = experiment_interview.build_packet(self.store, experiment, "post-off", AGENTS[1], "2017-11-21")
        self.assertEqual(packet["lineage"], ["shared-pre", "post-off"])
        self.assertEqual(packet["days"], ["2017-11-19", "2017-11-20", "2017-11-21"])
        self.assertEqual(packet["skipped_days"], ["2017-11-20"])
        kinds = [item.kind for item in packet["items"]]
        self.assertIn("skipped_day", kinds)
        self.assertNotIn("2017-11-22", {item.day for item in packet["items"]})
        with self.assertRaises(StoreError):
            experiment_interview.build_packet(self.store, experiment, "post-off", AGENTS[1], "2017-12-31")

    def test_journal_statements_are_marked_and_source_refs_rewritten(self) -> None:
        experiment = self.store.experiment(EXP)
        packet = experiment_interview.build_packet(self.store, experiment, "shared-pre", AGENTS[0])
        rationale = next(item for item in packet["items"] if item.kind == "stated_rationale")
        self.assertIn("확인되지 않은 진술", rationale.text)
        self.assertNotIn("[E0012]", rationale.text)
        self.assertIn("(입력 사실 12)", rationale.text)
        self.assertEqual(packet["journal_status"]["2017-11-19"], "used")
        event = next(item for item in packet["items"] if item.kind == "executed_event")
        self.assertIn("정책 대상", event.text)

    def test_answer_checks_flag_unexposed_citations_and_numbers(self) -> None:
        experiment = self.store.experiment(EXP)
        packet = experiment_interview.build_packet(self.store, experiment, "shared-pre", AGENTS[0])
        exposed = experiment_interview.select_items(packet, "당구장 갔어요?")
        target = next(n for n, item in enumerate(exposed, start=1) if item.kind == "target_facility")
        checked = experiment_interview.check_answer(f"당구장에서 12,000원 썼어요 [E{target}] 그리고 55,555원도 [E999] [E0012]", exposed)
        self.assertEqual([c["kind"] for c in checked["citations"]], ["target_facility"])
        self.assertEqual(checked["unsupported_numbers"], ["55,555"])
        self.assertEqual(len(checked["warnings"]), 3)

    def test_interview_calls_the_model_and_logs_console_only(self) -> None:
        seen = {}

        def fake_chat(system, question, history=None, **kwargs):
            seen["system"] = system
            return {"answer": "그날 당구장에 갔어요 [E2]", "model_label": "fake", "tokens": {}}

        with patch.object(interview, "chat", fake_chat):
            result = experiment_interview.interview_agent(self.store, EXP, "shared-pre", AGENTS[0], "당구장 갔어요?", fetch_evidence=False)
        self.assertTrue(result["console_only"])
        self.assertIn("[내 기록]", seen["system"])
        self.assertIn("오늘은 2017-11-20입니다", seen["system"])
        log = (self.root / "mirror" / EXP / "_console" / "interviews.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(log[-1])["agent_id"], AGENTS[0])
        self.assertEqual(experiment_interview.recent_interviews(self.store, EXP)["total"], 1)

    def test_group_stats_filters_and_breakdowns(self) -> None:
        stats = experiment_query.group_stats(self.store, EXP, "post-off", {"smoking_status": ["smoker"]})
        self.assertEqual(stats["overall"]["agents"], 2)
        self.assertEqual(stats["overall"]["skipped_agent_days"], 0)
        self.assertEqual(stats["overall"]["target_payments"], 1 + 2 + 1)
        self.assertEqual(set(stats["breakdowns"]["gu"]), {"노원구", "송파구"})
        with self.assertRaises(StoreError):
            experiment_query.group_stats(self.store, EXP, "post-off", {"password": "x"})


class FakeRemote(experiment_sync.Remote):
    def __init__(self, files: dict[str, bytes]) -> None:
        super().__init__(["ssh", "host"], runner=self._never)
        self.files = files
        self.calls: list[str] = []

    @staticmethod
    def _never(*_args, **_kwargs):
        raise AssertionError("no real ssh in tests")

    def list_files(self, directory, patterns, subdir=None):
        import fnmatch

        prefix = f"{directory}/{subdir}/" if subdir else f"{directory}/"
        found = {}
        for path, data in self.files.items():
            if path.startswith(prefix) and "/" not in path[len(prefix):]:
                name = path[len(prefix):]
                if any(fnmatch.fnmatch(name, p) for p in patterns):
                    found[f"{subdir}/{name}" if subdir else name] = (len(data), 1.0 + len(data))
        return found

    def cat(self, path):
        self.calls.append(f"cat {path}")
        self.bytes_received += len(self.files[path])
        return self.files[path]

    def tail_from(self, path, offset):
        self.calls.append(f"tail {path} {offset}")
        return self.files[path][offset:]

    def head_sha256(self, path, length):
        return hashlib.sha256(self.files[path][:length]).hexdigest()

    def tar(self, directory, members):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for member in members:
                data = self.files[f"{directory}/{member}"]
                info = tarfile.TarInfo(member)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return buffer.getvalue()


class SyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="sync-")
        self.root = Path(self.temp.name)
        registry = self.root / "registry"
        registry.mkdir()
        (registry / f"{EXP}.json").write_text(json.dumps(DEFINITION, ensure_ascii=False), encoding="utf-8")
        self.env = patch.dict(os.environ, {"SIM_EXPERIMENTS_ROOT": str(self.root / "mirror")})
        self.env.start()
        self.store = ExperimentStore(registry_dir=registry)
        self.experiment = self.store.experiment(EXP)

    def tearDown(self) -> None:
        self.env.stop()
        self.temp.cleanup()

    def test_mirror_downloads_then_appends_growing_day_file(self) -> None:
        line = lambda aid: json.dumps(_row(aid, "2017-11-21"), ensure_ascii=False).encode("utf-8") + b"\n"
        files = {
            "/srv/results/off/experiment_run.json": b'{"status":"running","cohort_ids":[]}',
            "/srv/results/off/metrics/day_2017-11-21.jsonl": line(AGENTS[0]),
            "/srv/results/off/notes.txt": b"ignored",
            "/srv/bundle/runtime.json": b"{}",
            "/srv/bundle/personas.json": b"[]",
        }
        remote = FakeRemote(files)
        report = experiment_sync.sync_experiment(self.experiment, remote=remote, store=self.store)
        self.assertEqual(report["arms"]["post-off"]["downloaded"], 2)
        local = self.root / "mirror" / EXP / "arms" / "post-off" / "metrics" / "day_2017-11-21.jsonl"
        self.assertEqual(local.read_bytes(), files["/srv/results/off/metrics/day_2017-11-21.jsonl"])
        self.assertFalse((self.root / "mirror" / EXP / "arms" / "post-off" / "notes.txt").exists())

        files["/srv/results/off/metrics/day_2017-11-21.jsonl"] += line(AGENTS[1])
        report = experiment_sync.sync_experiment(self.experiment, remote=remote)
        self.assertEqual(report["arms"]["post-off"]["appended"], 1)
        self.assertEqual(report["arms"]["post-off"]["unchanged"], 1)
        self.assertEqual(local.read_bytes(), files["/srv/results/off/metrics/day_2017-11-21.jsonl"])
        self.assertTrue(any(call.startswith("tail ") for call in remote.calls))

        # 앞부분이 달라지면(서버에서 다시 쓴 경우) 통째로 다시 받는다
        files["/srv/results/off/metrics/day_2017-11-21.jsonl"] = line(AGENTS[2]) + line(AGENTS[0]) + line(AGENTS[1])
        report = experiment_sync.sync_experiment(self.experiment, remote=remote)
        self.assertEqual(report["arms"]["post-off"]["downloaded"], 1)
        self.assertEqual(local.read_bytes(), files["/srv/results/off/metrics/day_2017-11-21.jsonl"])

    def test_agent_evidence_is_fetched_once(self) -> None:
        build_fixture(self.root / "fixture")
        source = self.root / "fixture" / "mirror" / EXP / "arms" / "shared-pre"
        target = self.root / "mirror" / EXP / "arms" / "shared-pre" / "metrics"
        target.mkdir(parents=True)
        for path in (source / "metrics").iterdir():
            (target / path.name).write_bytes(path.read_bytes())
        journal = (source / "evidence/v1/2017-11-19/abc.jsonl").read_bytes()
        remote = FakeRemote({"/srv/results/pre/evidence/v1/2017-11-19/abc.jsonl": journal})
        first = experiment_sync.fetch_agent_evidence(self.store, self.experiment, AGENTS[0], remote=remote)
        self.assertEqual(first["fetched"], 1)
        second = experiment_sync.fetch_agent_evidence(self.store, self.experiment, AGENTS[0], remote=remote)
        self.assertEqual(second["fetched"], 0)

    def test_remote_settings_and_paths_are_guarded(self) -> None:
        with patch.dict(os.environ, {"SIM_REMOTE_TEST_REMOTE": ""}):
            with self.assertRaises(experiment_sync.SyncError):
                experiment_sync.ssh_base("TEST_REMOTE")
        with patch.dict(os.environ, {"SIM_REMOTE_TEST_REMOTE": "rm -rf /"}):
            with self.assertRaises(experiment_sync.SyncError):
                experiment_sync.ssh_base("TEST_REMOTE")
        with patch.dict(os.environ, {"SIM_REMOTE_TEST_REMOTE": "ssh -p 22 user@host"}):
            base = experiment_sync.ssh_base("TEST_REMOTE")
        self.assertEqual(base[0], "ssh")
        self.assertIn("BatchMode=yes", base)
        self.assertEqual(base[-1], "user@host")
        with self.assertRaises(experiment_sync.SyncError):
            experiment_sync._remote_path("/srv/results", "../../etc")


class ExperimentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="exp-api-")
        self.root = Path(self.temp.name)
        build_fixture(self.root)
        self.env = patch.dict(os.environ, {"SIM_EXPERIMENTS_ROOT": str(self.root / "mirror")})
        self.env.start()
        store = ArtifactStore(repo_root=ROOT, data_root=self.root / "data", fixture_dir=FIXTURES)
        runner = Runner(repo_root=ROOT, lock=RunLock(self.root / "run.lock"))
        with patch("web.api.experiments.REGISTRY_DIR", self.root / "registry"):
            self.app = create_app(store=store, runner=runner)
        self.app.state.experiments.registry_dir = self.root / "registry"
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.env.stop()
        self.temp.cleanup()

    def test_read_routes(self) -> None:
        listing = self.client.get("/api/experiments").json()
        self.assertEqual([item["experiment_id"] for item in listing["items"]], [EXP])
        self.assertNotIn("/srv", json.dumps(listing))  # 원격 위치는 내보내지 않는다
        self.assertEqual(self.client.get(f"/api/experiments/{EXP}").json()["arms"][0]["phase"], "complete")
        self.assertEqual(len(self.client.get(f"/api/experiments/{EXP}/daily").json()["series"]), 3)
        self.assertTrue(self.client.get(f"/api/experiments/{EXP}/comparison").json()["available"])
        self.assertEqual(self.client.get(f"/api/experiments/{EXP}/agents").json()["total"], 3)
        self.assertEqual(self.client.get(f"/api/experiments/{EXP}/agents/{AGENTS[0]}").status_code, 200)
        self.assertEqual(self.client.get("/api/experiments/NOPE").status_code, 400)
        self.assertEqual(self.client.get("/api/experiments/nope").status_code, 404)
        sync = self.client.get(f"/api/experiments/{EXP}/sync").json()
        self.assertEqual((sync["state"], sync["remote_configured"]), ("idle", False))

    def test_interview_and_stats_routes(self) -> None:
        stats = self.client.post(f"/api/experiments/{EXP}/query/stats", json={"arm_id": "post-off", "filters": {"gu": ["노원구"]}})
        self.assertEqual(stats.json()["overall"]["agents"], 1)
        with patch.object(interview, "chat", lambda *a, **k: {"answer": "기억에 없어요", "model_label": "fake", "tokens": {}}):
            reply = self.client.post(f"/api/experiments/{EXP}/agents/{AGENTS[1]}/interview",
                                     json={"arm_id": "post-off", "question": "요즘 어때요?"})
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.json()["warnings"], ["답변에 근거 번호가 없습니다"])
        with patch.dict(os.environ, {"INTERVIEW_BASE_URL": "", "GEMINI_API_KEY": "", "REPORT_LLM_BASE_URL": ""}):
            with patch.object(interview, "load_dotenv", lambda *a, **k: None):
                blocked = self.client.post(f"/api/experiments/{EXP}/query/interviews",
                                           json={"arm_id": "post-off", "question": "?", "sample": 2})
        self.assertEqual(blocked.status_code, 503)


if __name__ == "__main__":
    unittest.main()
