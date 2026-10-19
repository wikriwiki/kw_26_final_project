"""Protect baseline inputs, existing databases, credentials and network isolation."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("vast_neo4j_pair", ROOT / "deploy/vast/neo4j_pair.py")
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)


class Neo4jPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve()
        self.dump = self.workspace / "baseline.dump"
        self.dump.write_bytes(b"verified clean baseline fixture")
        self.env = {"CLEAN_BASELINE_DUMP": str(self.dump),
                    "NO_SMOKING_SNAPSHOT_SHA256": hashlib.sha256(self.dump.read_bytes()).hexdigest(),
                    "NEO4J_PASSWORD": "private-test-password"}

    def config(self, **overrides):
        return PAIR.settings({**self.env, **overrides}, workspace=self.workspace)

    def test_summary_contains_no_password_and_no_mutation(self):
        before = self.dump.read_bytes()
        cfg = self.config()
        report = json.dumps(PAIR.summary(cfg))
        self.assertNotIn(self.env["NEO4J_PASSWORD"], report)
        self.assertFalse(cfg["root"].exists())
        self.assertEqual(before, self.dump.read_bytes())
        self.assertFalse(PAIR.summary(cfg)["load_overwrite_destination"])

    def test_existing_directory_is_never_reused(self):
        root = self.workspace / "existing"
        root.mkdir()
        sentinel = root / "do-not-change"
        sentinel.write_text("original")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.config(NO_SMOKING_NEO4J_ROOT=str(root))
        self.assertEqual("original", sentinel.read_text())

    def test_root_outside_workspace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside /workspace"):
            self.config(NO_SMOKING_NEO4J_ROOT=str(self.workspace.parent / "unowned"))

    def test_dotdot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "symlinks or"):
            self.config(NO_SMOKING_NEO4J_ROOT=str(self.workspace / ".." / "unexpected"))

    def test_workspace_root_itself_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NEW child"):
            self.config(NO_SMOKING_NEO4J_ROOT=str(self.workspace))

    def test_duplicate_ports_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "different"):
            self.config(NO_SMOKING_OFF_BOLT_PORT="17687", NO_SMOKING_ON_BOLT_PORT="17687")

    def test_missing_or_short_password_is_rejected_without_leak(self):
        for password in ("", "tiny"):
            with self.assertRaisesRegex(ValueError, "at least 8") as context:
                self.config(NEO4J_PASSWORD=password)
            if password:
                self.assertNotIn(password, str(context.exception))

    def test_hash_mismatch_stops_before_creating_root(self):
        cfg = self.config(NO_SMOKING_SNAPSHOT_SHA256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            PAIR.check_input(cfg)
        self.assertFalse(cfg["root"].exists())

    def test_occupied_port_stops_before_creating_root(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            cfg = self.config(NO_SMOKING_OFF_BOLT_PORT=str(port), NO_SMOKING_NEO4J_MIN_FREE_GB="4")
            with patch.object(PAIR.shutil, "disk_usage", return_value=SimpleNamespace(free=5 * 1024**3)), \
                 self.assertRaisesRegex(ValueError, "occupied"):
                PAIR.check_input(cfg)
            self.assertFalse(cfg["root"].exists())

    def test_insufficient_disk_stops_before_creating_root(self):
        cfg = self.config(NO_SMOKING_NEO4J_MIN_FREE_GB="4")
        with patch.object(PAIR.shutil, "disk_usage", return_value=SimpleNamespace(free=3 * 1024**3)), \
             self.assertRaisesRegex(ValueError, "At least 4 GiB"):
            PAIR.check_input(cfg)
        self.assertFalse(cfg["root"].exists())

    def test_dry_run_validates_without_writing_or_revealing_password(self):
        ports = []
        for _ in range(2):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                ports.append(sock.getsockname()[1])
        env = {**self.env, "NO_SMOKING_OFF_BOLT_PORT": str(ports[0]),
               "NO_SMOKING_ON_BOLT_PORT": str(ports[1]), "NO_SMOKING_NEO4J_MIN_FREE_GB": "4"}
        output = io.StringIO()
        with patch.object(PAIR, "WORKSPACE", self.workspace), \
             patch.object(PAIR.shutil, "disk_usage", return_value=SimpleNamespace(free=5 * 1024**3)), \
             patch.dict(PAIR.os.environ, env, clear=True), \
             patch.object(PAIR.sys, "argv", ["neo4j_pair.py", "--dry-run"]), patch("sys.stdout", output):
            result = PAIR.main()
        self.assertEqual(0, result)
        self.assertEqual("dry_run_validated_no_changes", json.loads(output.getvalue())["status"])
        self.assertNotIn(self.env["NEO4J_PASSWORD"], output.getvalue())
        self.assertEqual([self.dump], list(self.workspace.iterdir()))

    def test_configuration_keeps_auth_and_bolt_local(self):
        cfg = self.config()
        for arm in ("off", "on"):
            content = PAIR.configuration(cfg, arm)
            self.assertIn("server.bolt.listen_address=127.0.0.1:", content)
            self.assertIn("dbms.security.auth_enabled=true", content)
            self.assertIn("server.http.enabled=false", content)
            self.assertIn("server.memory.heap.max_size=2048m", content)
            self.assertIn("server.memory.pagecache.size=512m", content)
            self.assertNotIn("0.0.0.0", content)

    def test_subprocess_environment_does_not_inherit_passwords(self):
        with patch.dict(PAIR.os.environ, {"NEO4J_PASSWORD": "hidden", "HF_TOKEN": "hidden", "JAVA_OPTS": "unsafe"}):
            result = PAIR.neo_env(self.workspace / "off")
        self.assertNotIn("NEO4J_PASSWORD", result)
        self.assertNotIn("HF_TOKEN", result)
        self.assertNotIn("JAVA_OPTS", result)
        self.assertEqual(str(self.workspace / "off"), result["NEO4J_HOME"])

    def test_vendor_jvm_options_survive_configuration(self):
        cfg = self.config()
        home = self.workspace / "fixture-home"
        (home / "conf").mkdir(parents=True)
        path = home / "conf/neo4j.conf"
        path.write_text("server.jvm.additional=-XX:+UseG1GC\nserver.http.enabled=true\n")
        PAIR.write_configuration(home, cfg, "off")
        result = path.read_text()
        self.assertIn("server.jvm.additional=-XX:+UseG1GC", result)
        self.assertNotIn("server.http.enabled=true", result)
        self.assertEqual(1, result.count("server.http.enabled="))

    def test_archive_traversal_is_rejected(self):
        archive = self.workspace / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("neo4j-community-5.26.0/../escaped")
            member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
        home = self.workspace / "off"
        home.mkdir()
        with self.assertRaisesRegex(ValueError, "Unexpected path"):
            PAIR.extract(archive, home)
        self.assertFalse((self.workspace / "escaped").exists())

    def test_archive_links_are_rejected(self):
        archive = self.workspace / "bad-link.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("neo4j-community-5.26.0/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            tar.addfile(member)
        home = self.workspace / "off"
        home.mkdir()
        with self.assertRaisesRegex(ValueError, "Links/devices"):
            PAIR.extract(archive, home)


if __name__ == "__main__":
    unittest.main()
