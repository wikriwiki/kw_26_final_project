"""집단 질의 — 조건으로 대상자를 고르고, 기록 통계를 보거나 여러 명에게 같은 질문을 한다.

* 통계(``group_stats``)는 LLM 없이 봉인된 하루 집계만으로 바로 계산한다.
* 집단 인터뷰(``QueryJobs``)는 표본마다 1대1 인터뷰와 같은 근거 규칙으로 묻는다. 배경에서 돌고 진행을 알린다.
  끝나면 답들을 모델이 요약할 수 있다 — 요약은 확인되지 않은 정리이므로 개별 답과 따로 표시한다.
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from . import interview
from .experiment_interview import interview_agent, spoken
from .experiments import CONSOLE_DIR, DAY, Experiment, ExperimentStore, paired_subset_effect
from .store import StoreError

FILTER_KEYS = ("gu", "sex", "age_group", "smoking_status", "income_level", "life_stage")
MAX_SAMPLE = 30
WORKERS = 3

SUMMARY_SYSTEM = """아래는 같은 질문에 대한 시뮬레이션 속 여러 인물의 답변입니다.
공통으로 나온 내용과 서로 다른 내용을 3~4문장의 한 문단으로 정리하세요.
- 답변에 없는 내용을 더하지 마세요. 숫자는 답변에 적힌 그대로만 쓰세요.
- 특정 인물 id 를 나열하지 말고, 괄호 안의 특성(연령·성별·소득·사는 구)으로 묶어 말로 풀어 쓰세요. 괄호를 그대로 옮겨 적지 마세요.
- 괄호에 없는 연령·지역·특성은 쓰지 마세요.
- 요약 문장만 쓰세요. 머리말이나 안내 문구, 빈 줄이나 끝나지 않은 문장을 남기지 마세요.
"""


SEX = {"F": "여성", "M": "남성"}


def _traits(row: dict[str, Any]) -> str:
    """요약에 넘길 응답자 특성. 흡연 여부는 금연구역 실험에만 있다 — 없는 특성은 넣지 않는다."""
    income = f"소득 {row['income_level']}" if row.get("income_level") else ""
    values = (row.get("smoking_status"), row.get("gu"), row.get("age_group"), SEX.get(str(row.get("sex")), ""), income)
    return ", ".join(str(v) for v in values if v)


def tidy_summary(text: str) -> str:
    """요약에서 끝나지 않은 줄('70대 이상 연령대.  .')과 다른 문자 조각을 걷어 낸다."""
    text = spoken(text)
    kept = []
    for line in text.splitlines():
        line = re.sub(r"(?:\s*\.){2,}", ".", line).strip()
        line = re.sub(r"\s+(?:and|or)\s+", ", ", line)  # 한국어 요약에 섞인 영어 접속사
        if len(re.sub(r"[^가-힣]", "", line)) >= 12:
            kept.append(line)
    return "\n".join(kept)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def select_agents(store: ExperimentStore, experiment_id: str, filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    filters = filters or {}
    unknown = set(filters) - set(FILTER_KEYS)
    if unknown:
        raise StoreError(400, "알 수 없는 조건: " + ", ".join(sorted(unknown)))
    agents = store.roster(experiment_id)["items"]
    for key, allowed in filters.items():
        if allowed in (None, "", []):
            continue
        values = {str(value) for value in (allowed if isinstance(allowed, list) else [allowed])}
        agents = [agent for agent in agents if str(agent.get(key)) in values]
    return agents


def group_effect(store: ExperimentStore, experiment_id: str, filters: dict[str, Any] | None) -> dict[str, Any]:
    """고른 집단의 정책 효과(같은 시민·같은 날 있음 − 없음). 정책 실험에서만 뜻이 있다."""
    experiment = store.experiment(experiment_id)
    if experiment.kind != "policy_paired":
        raise StoreError(400, "정책 있음·없음을 함께 돌린 실험에서만 집단 효과를 낼 수 있습니다")
    agents = select_agents(store, experiment_id, filters)
    return {"filters": filters or {}, "matched": len(agents), **paired_subset_effect(store, experiment, [a["id"] for a in agents])}


def _days_through(experiment: Experiment, arm_id: str, through_day: str | None) -> list[tuple[Any, str]]:
    arm = experiment.arm(arm_id)
    if through_day is not None and not DAY.match(through_day):
        raise StoreError(400, "날짜 형식은 YYYY-MM-DD 입니다")
    return [(step, day) for step in experiment.lineage(arm) for day in step.day_list() if not through_day or day <= through_day]


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def group_stats(store: ExperimentStore, experiment_id: str, arm_id: str, filters: dict[str, Any] | None = None,
                through_day: str | None = None) -> dict[str, Any]:
    experiment = store.experiment(experiment_id)
    agents = select_agents(store, experiment_id, filters)
    wanted = {agent["id"]: agent for agent in agents}
    per_agent: dict[str, dict[str, Any]] = {
        aid: {"ok": 0, "skipped": 0, "spend": 0.0, "receipts": 0, "amount": 0, "target_payments": 0, "target_revenue": 0, "mood": []}
        for aid in wanted
    }
    days_used = []
    for step, day in _days_through(experiment, arm_id, through_day):
        loaded = store.cache.get(experiment, step, day)
        if loaded is None:
            continue
        days_used.append(day)
        for aid, compact in loaded["summary"]["agents"].items():
            record = per_agent.get(aid)
            if record is None:
                continue
            if compact.get("s") != "ok":
                record["skipped"] += 1
                continue
            record["ok"] += 1
            record["spend"] += compact.get("spend") or 0
            record["receipts"] += compact.get("n") or 0
            record["amount"] += compact.get("amt") or 0
            if compact.get("mood") is not None:
                record["mood"].append(compact["mood"])
            for _facility, _district, revenue, count in compact.get("t") or []:
                record["target_payments"] += count
                record["target_revenue"] += revenue

    def describe(ids: list[str]) -> dict[str, Any]:
        rows = [per_agent[aid] for aid in ids]
        ok_days = sum(row["ok"] for row in rows)
        return {
            "agents": len(ids),
            "observed_agent_days": ok_days,
            "skipped_agent_days": sum(row["skipped"] for row in rows),
            "spend_per_observed_day": round(sum(row["spend"] for row in rows) / ok_days, 1) if ok_days else None,
            "receipts_per_observed_day": round(sum(row["receipts"] for row in rows) / ok_days, 3) if ok_days else None,
            "target_payments": sum(row["target_payments"] for row in rows),
            "target_revenue_krw": sum(row["target_revenue"] for row in rows),
            "agents_with_target_visit": sum(1 for row in rows if row["target_payments"] > 0),
            "mood_mean": _mean([value for row in rows for value in row["mood"]]),
        }

    breakdowns: dict[str, dict[str, Any]] = {}
    for key in ("smoking_status", "gu", "sex", "age_group"):
        groups: dict[str, list[str]] = defaultdict(list)
        for aid, agent in wanted.items():
            groups[str(agent.get(key) or "미상")].append(aid)
        breakdowns[key] = {value: describe(ids) for value, ids in sorted(groups.items())}
    return {
        "experiment_id": experiment_id,
        "arm_id": arm_id,
        "filters": filters or {},
        "through_day": through_day or (days_used[-1] if days_used else None),
        "days": len(days_used),
        "overall": describe(list(wanted)),
        "breakdowns": breakdowns,
        "notes": [
            "건너뛴 대상자-일은 결측으로 빼고, 관측된 날 기준으로 나눈다.",
            "정책 대상 시설 결제 건수는 방문자 수가 아니다.",
        ],
    }


class QueryJobs:
    def __init__(self, store: ExperimentStore) -> None:
        self.store = store
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def get(self, experiment_id: str, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job["experiment_id"] == experiment_id:
                return json.loads(json.dumps(job, ensure_ascii=False))
        experiment = self.store.experiment(experiment_id)
        path = experiment.local_dir / CONSOLE_DIR / "queries" / f"{job_id}.json"
        if len(job_id) == 32 and job_id.isalnum() and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        raise StoreError(404, "질의 작업을 찾을 수 없습니다")

    def start(self, experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        experiment = self.store.experiment(experiment_id)
        arm_id = str(payload.get("arm_id") or "")
        experiment.arm(arm_id)
        question = str(payload.get("question") or "").strip()
        if not question or len(question) > 500:
            raise StoreError(400, "질문은 1~500자로 적어 주세요")
        sample = int(payload.get("sample") or 10)
        if not 1 <= sample <= MAX_SAMPLE:
            raise StoreError(400, f"한 번에 1~{MAX_SAMPLE}명까지 물을 수 있습니다")
        seed = int(payload.get("seed") or 0)
        through_day = payload.get("through_day")
        if not interview.settings()["ready"]:
            raise StoreError(503, "대화 모델이 연결되지 않아 집단 인터뷰를 할 수 없습니다. 통계 질의는 쓸 수 있습니다.")
        agents = select_agents(self.store, experiment_id, payload.get("filters"))
        if not agents:
            raise StoreError(409, "조건에 맞는 대상자가 없습니다")
        ids = sorted(agent["id"] for agent in agents)
        chosen = random.Random(seed).sample(ids, min(sample, len(ids)))
        by_id = {agent["id"]: agent for agent in agents}
        job = {
            "job_id": uuid.uuid4().hex,
            "experiment_id": experiment_id,
            "arm_id": arm_id,
            "question": question,
            "filters": payload.get("filters") or {},
            "through_day": through_day,
            "seed": seed,
            "matched": len(ids),
            "sampled": chosen,
            "state": "running",
            "done": 0,
            "results": [],
            "summary": None,
            "started_at": now(),
        }
        with self.lock:
            self.jobs[job["job_id"]] = job

        def ask(agent_id: str) -> dict[str, Any]:
            profile = by_id.get(agent_id) or {}
            base = {"agent_id": agent_id, **{key: profile.get(key) for key in FILTER_KEYS}}
            try:
                reply = interview_agent(self.store, experiment_id, arm_id, agent_id, question, None, through_day,
                                        fetch_evidence=bool(payload.get("fetch_evidence")))
            except StoreError as exc:
                return {**base, "error": exc.message}
            return {**base, "answer": reply["answer"], "citations": reply["citations"], "warnings": reply["warnings"],
                    "through_day": reply["through_day"], "model_label": reply["model_label"]}

        def work() -> None:
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for result in pool.map(ask, chosen):
                    with self.lock:
                        job["results"].append(result)
                        job["done"] += 1
            answers = [r for r in job["results"] if r.get("answer")]
            if payload.get("summarize", True) and len(answers) >= 2:
                joined = "\n".join(
                    # 흡연 여부는 금연구역 실험에만 있다. 없는 특성은 괄호에 넣지 않는다
                    f"- ({_traits(r)}) {r['answer']}"
                    for r in answers
                )
                try:
                    summary = interview.chat(SUMMARY_SYSTEM, f"질문: {question}\n\n답변들:\n{joined}", None, max_tokens=600, temperature=0.3)
                    job["summary"] = {"text": tidy_summary(summary["answer"]), "model_label": summary.get("model_label"), "verified": False}
                except interview.InterviewError as exc:
                    job["summary"] = {"error": exc.message}
            with self.lock:
                job["state"] = "done"
                job["finished_at"] = now()
            path = experiment.local_dir / CONSOLE_DIR / "queries" / f"{job['job_id']}.json"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError:
                pass

        if os.environ.get("WEB_INLINE_JOBS", "").strip().lower() in {"1", "true", "yes", "on"}:
            # 서버리스 함수(버셀)는 응답을 돌려준 뒤 배경 스레드가 멈추고, 다음 조회가 다른 인스턴스로 갈 수 있다.
            # 그래서 끝까지 묻고 'done' 으로 돌려준다 — 화면은 done 을 받으면 다시 묻지 않는다
            work()
        else:
            threading.Thread(target=work, name=f"query-{job['job_id'][:8]}", daemon=True).start()
        return self.get(experiment_id, job["job_id"])
