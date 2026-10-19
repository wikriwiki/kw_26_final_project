import argparse
import copy

import pytest

from scripts.experiments import rebuild_no_smoking_baseline as rebuild


def arguments():
    return argparse.Namespace(uri="bolt://127.0.0.1:17687", database="neo4j",
                              source_sha256="a" * 64, day_zero="2017-11-25",
                              source_cohort_label="reviewed original 14881 persona generation",
                              confirm_target="bolt://127.0.0.1:17687|neo4j")


def source_marker(args):
    return {"id": rebuild.SOURCE_ID, "isolated": True,
            "source_sha256": args.source_sha256, "intended_day_zero": args.day_zero,
            "target_uri": args.uri, "target_database": args.database}


def reviewed_report(args):
    return {"blockers": [], "source_markers": [source_marker(args)]}


def test_apply_needs_independent_isolation_target_and_source_guards():
    args = arguments()
    report = reviewed_report(args)
    with pytest.raises(ValueError, match="ISOLATED"):
        rebuild.validate_apply(args, report, {})
    args.confirm_target = "bolt://127.0.0.1:7687|neo4j"
    with pytest.raises(ValueError, match="confirm-target"):
        rebuild.validate_apply(args, report, {"NO_SMOKING_REBUILD_ISOLATED": "1"})
    args = arguments()
    report["source_markers"][0]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="marker"):
        rebuild.validate_apply(args, report, {"NO_SMOKING_REBUILD_ISOLATED": "1"})


@pytest.mark.parametrize("change", [
    {"status": "in_progress"}, {"status": "complete"}, {"isolated": 1},
    {"intended_day_zero": "2025-07-13"}, {"target_database": "production"},
])
def test_source_marker_cannot_be_reused_or_retargeted(change):
    args = arguments()
    report = reviewed_report(args)
    report["source_markers"][0].update(change)
    with pytest.raises(ValueError):
        rebuild.validate_apply(args, report, {"NO_SMOKING_REBUILD_ISOLATED": "1"})


def test_guard_accepts_reviewed_staging_and_rejects_unknown_schema():
    args = arguments()
    report = reviewed_report(args)
    rebuild.validate_apply(args, report, {"NO_SMOKING_REBUILD_ISOLATED": "1"})
    report["blockers"] = ["Unreviewed node labels"]
    with pytest.raises(ValueError, match="audit blocked"):
        rebuild.validate_apply(args, report, {"NO_SMOKING_REBUILD_ISOLATED": "1"})


def test_remote_production_endpoint_cannot_apply_even_with_matching_marker():
    args = arguments()
    args.uri = "bolt://production.example:7687"
    args.confirm_target = args.uri + "|neo4j"
    with pytest.raises(ValueError, match="loopback"):
        rebuild.validate_apply(args, reviewed_report(args), {"NO_SMOKING_REBUILD_ISOLATED": "1"})


def test_unknown_labels_mixed_static_dynamic_and_relationships_fail_closed():
    assert rebuild.schema_blockers([{"labels": ["Agent"]}], []) == []
    for labels in (["Unknown"], ["Agent", "Memory"], []):
        assert rebuild.schema_blockers([{"labels": labels}], [])
    for relation in (
        {"type": "unknown", "source_labels": ["Agent"], "target_labels": ["POI"]},
        {"type": "HAS_PLAN", "source_labels": ["Agent"], "target_labels": ["POI"]},
    ):
        assert rebuild.schema_blockers([], [relation])


def test_fresh_state_does_not_reuse_old_balance_or_policy_memory():
    agents = [{"id": "a", "weekday": 10000, "weekend": 17000,
               "balance": 12, "policy_lifecycle": '{"P010":true}'}]
    row = rebuild.state_rows(agents, "2017-11-25")[0]
    assert row["balance"] == 468000
    assert row["id"] == "a_2017-11-25" and row["day"] == "2017-11-25"
    assert row["policy_lifecycle"] == "{}" and row["month_spent"] == 0
    assert row["sangsaeng_month_spent"] == 0
    assert row["mood"] == 0.5 and row["fatigue"] == 0.3
    assert "experience_run_id" not in row


@pytest.mark.parametrize("anchors", [(float("nan"), 10), (float("inf"), 10), (-1, 10), (True, 10), ("100", 10)])
def test_corrupt_spending_anchors_are_not_silently_replaced(anchors):
    with pytest.raises(ValueError, match="anchor"):
        rebuild.initial_balance(*anchors)


def test_missing_spending_anchors_use_existing_loader_fallback():
    assert rebuild.initial_balance(None, None) == 1500000
    assert rebuild.initial_balance(None, 10000) == 390000


def test_same_id_different_persona_generation_fails_cohort_alignment():
    graph = [{"id": "same", "age": 42, "sex": "F"}]
    runtime = {"experiment_id": "no_smoking_zone", "cohort": [{"id": "same", "age": 43, "sex": "female"}]}
    blockers, report = rebuild.cohort_blockers(graph, runtime)
    assert blockers and report["age_sex_mismatch"] == 1
    runtime["cohort"][0]["age"] = 42
    assert rebuild.cohort_blockers(graph, runtime)[0] == []


class Result(list):
    def consume(self):
        return None

    def single(self):
        assert len(self) == 1
        return self[0]


class FingerprintSession:
    def __init__(self, nodes):
        self.nodes = nodes

    def run(self, query, **params):
        assert "ORDER BY" not in query  # No graph-sized sort on a small staging server.
        return Result(copy.deepcopy(self.nodes) if "labels(n)" in query else [])


def test_streamed_fingerprint_is_order_independent_but_detects_static_changes():
    nodes = [
        {"element_id": "1", "labels": ["Agent"], "properties": {"id": "a", "execution_lock": 8}},
        {"element_id": "2", "labels": ["POI"], "properties": {"id": "p", "name": "당구장"}},
    ]
    a = rebuild.static_fingerprint(FingerprintSession(nodes))
    assert a == rebuild.static_fingerprint(FingerprintSession(list(reversed(nodes))))
    nodes[0]["properties"].pop("execution_lock")
    assert a == rebuild.static_fingerprint(FingerprintSession(nodes))
    nodes[1]["properties"]["name"] = "다른 상점"
    assert a != rebuild.static_fingerprint(FingerprintSession(nodes))


def test_default_audit_executes_only_reads_and_unknown_awareness_blocks():
    class AuditSession:
        def run(self, query, **params):
            assert not any(word in query for word in ("DELETE", "CREATE", "MERGE", " SET ", "REMOVE"))
            if "RETURN labels(n)" in query:
                return Result([{"labels": ["Agent"], "count": 1}])
            if "source_labels" in query:
                return Result([])
            if "a.personal_age" in query:
                return Result([{"id": "a", "age": 40, "sex": "M", "weekday": 10000, "weekend": 10000}])
            if "kp.source AS source" in query:
                return Result([{"source": "initial", "count": 3}, {"source": "unreviewed", "count": 1}])
            if "keys(kp)" in query:
                return Result([{"key": "source"}])
            return Result([])
    report, agents = rebuild.audit_graph(AuditSession())
    assert len(agents) == 1 and any("Unreviewed KNOWS_POI source" in b for b in report["blockers"])


def test_rebuild_resets_initial_affinity_removes_learned_edges_and_checks_static(monkeypatch):
    queries = []

    class WriteSession:
        def run(self, query, **params):
            queries.append((query, params))
            if "RETURN count(a) AS count" in query:
                return Result([{"count": 0}])
            return Result([])

    monkeypatch.setattr(rebuild, "static_fingerprint", lambda session, **kwargs: {"unchanged": True})
    result = rebuild.rebuild(WriteSession(), arguments(), [{"id": "a", "weekday": 1000, "weekend": 1000}])
    assert result["initial_states"] == 1
    assert any("WHERE kp.source <> 'initial'" in q and "DELETE kp" in q for q, _ in queries)
    assert any("SET kp = {source:'initial', since:date($day), affinity:0.5, visit_count:0}" in q for q, _ in queries)
    assert any("MATCH (n:Memory)" in q and "DETACH DELETE n" in q for q, _ in queries)
    assert not any("MATCH (n:Agent)" in q and "DELETE" in q for q, _ in queries)
    assert any("CREATE (s:State)" in q and p["rows"][0]["balance"] == 39000 for q, p in queries if "rows" in p)


@pytest.mark.parametrize("value", [0, -1, 1001, 5000, True, 1.5, "invalid"])
def test_write_batch_size_rejects_unbounded_or_non_integer_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="1 to 1000"):
        rebuild.batch_size_value(value)


def test_small_batch_size_applies_to_deletes_awareness_and_state_creation(monkeypatch):
    queries, progress = [], []

    class WriteSession:
        def run(self, query, **params):
            queries.append((query, params))
            return Result([{"count": 0}]) if "RETURN count(a) AS count" in query else Result([])

    monkeypatch.setattr(rebuild, "static_fingerprint", lambda session, **kwargs: {"unchanged": True})
    args = arguments()
    args.batch_size = 2
    agents = [{"id": f"a{i}", "weekday": 1000, "weekend": 1000} for i in range(5)]
    report = rebuild.rebuild(WriteSession(), args, agents, progress=progress.append)
    batched = [q for q, _ in queries if "IN TRANSACTIONS" in q]
    assert batched and all("IN TRANSACTIONS OF 2 ROWS" in q for q in batched)
    assert all("CALL (" in q and "RETURN touched" in q for q in batched)
    assert [len(p["rows"]) for q, p in queries if "CREATE (s:State)" in q] == [2, 2, 1]
    assert report["batch_size"] == 2
    assert any(e["phase"] == "delete_Memory" and e["status"] == "started" for e in progress)
    assert progress[-1]["phase"] == "finalize_baseline" and progress[-1]["status"] == "complete"


def test_failed_batch_reports_phase_without_finalizing_or_retrying(monkeypatch):
    progress, queries = [], []

    class FailingSession:
        def run(self, query, **params):
            queries.append(query)
            if "MATCH (n:Memory)" in query:
                raise RuntimeError("transaction memory exceeded")
            return Result([])

    monkeypatch.setattr(rebuild, "static_fingerprint", lambda session, **kwargs: {"unchanged": True})
    with pytest.raises(RuntimeError, match="memory exceeded"):
        rebuild.rebuild(FailingSession(), arguments(), [], progress=progress.append)
    assert progress[-1]["phase"] == "delete_Memory"
    assert sum("MATCH (n:Memory)" in q for q in queries) == 1
    assert not any("SET s.status='complete'" in q for q in queries)


@pytest.mark.parametrize("dirty,learned,error", [
    (1, 0, "simulation output"), (0, 1, "learned/future"), (0, 0, None),
])
def test_runtime_preflight_rejects_residual_memories_and_learned_awareness(monkeypatch, dirty, learned, error):
    import neo4j
    from scripts.experiments import no_smoking_zone as experiment

    queries = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, query, **params):
            queries.append(query)
            if "ExperimentSnapshot" in query:
                return Result([{"sha256": "a" * 64}])
            if "RETURN DISTINCT a.id" in query:
                return Result([{"id": "agent"}])
            if "d.code AS district" in query:
                return Result([{"id": "poi", "district": "11650"}])
            if "c.parent='여가'" in query:
                return Result([{"id": "poi"}])
            if "s.balance AS balance" in query:
                return Result([{"id": "agent", "balance": 500000}])
            if "MATCH (n) WHERE" in query:
                assert "n:Memory" in query and "n.day <> date($day)" in query
                return Result([{"n": dirty}])
            if "MATCH ()-[kp:KNOWS_POI]" in query:
                return Result([{"n": learned}])
            if "MATCH (p:Policy)" in query:
                return Result([{"n": 0}])
            raise AssertionError(query)

    class Driver(Session):
        def session(self, **kwargs):
            return Session()

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **kw: Driver())
    monkeypatch.setenv("NO_SMOKING_OFF_NEO4J_URI", "bolt://127.0.0.1:1")
    monkeypatch.setenv("NO_SMOKING_ON_NEO4J_URI", "bolt://127.0.0.1:2")
    monkeypatch.setenv("NEO4J_PASSWORD", "fixture")
    runtime = {"cohort": [{"id": "agent"}], "pois": [{"poi_id": "poi", "district_code": "11650"}]}
    if error:
        with pytest.raises(ValueError, match=error):
            experiment.graph_preflight(runtime, "off", "a" * 64, "2017-11-26")
    else:
        assert experiment.graph_preflight(runtime, "off", "a" * 64, "2017-11-26")["NEO4J_URI"].endswith(":1")
