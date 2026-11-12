from pathlib import Path
import sqlite3
import tempfile
import unittest

from cohorts.records import InputError
from releases.ledger import ConflictError, Ledger, canonical, digest


class LedgerRecovery(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "runs.sqlite3"
        self.ledger = Ledger(self.path)
        self.addCleanup(self.ledger.close)
        self.batch_id, _ = self.ledger.put_batch({"source": "synthetic"})
        self.run_id, _ = self.ledger.create_run(self.batch_id, "baseline", "request-1")

    def result(self):
        return {"scenario": "baseline", "demand": [{"age_counts": {"6": 3, "10": 4}}]}

    def complete(self):
        revision = self.ledger.claim(self.run_id, "worker", clock=100, duration=100)
        fingerprint = self.ledger.complete(self.run_id, "worker", revision, self.result(), clock=101)
        return revision, fingerprint

    def test_import_is_idempotent(self):
        second_id, created = self.ledger.put_batch({"source": "synthetic"})
        self.assertEqual(second_id, self.batch_id)
        self.assertFalse(created)
        history = self.ledger.history(self.batch_id)
        self.assertEqual([row["action"] for row in history], ["batch_imported"])

    def test_request_key_is_bound_to_batch_and_scenario(self):
        run_id, created = self.ledger.create_run(self.batch_id, "baseline", "request-1")
        self.assertEqual(run_id, self.run_id)
        self.assertFalse(created)
        with self.assertRaises(ConflictError):
            self.ledger.create_run(self.batch_id, "alternative", "request-1")

    def test_active_lease_excludes_other_worker(self):
        revision = self.ledger.claim(self.run_id, "first", clock=100, duration=10)
        self.assertEqual(revision, 1)
        with self.assertRaises(ConflictError):
            self.ledger.claim(self.run_id, "second", clock=109, duration=10)

    def test_expired_worker_is_fenced_after_recovery(self):
        first = self.ledger.claim(self.run_id, "first", clock=100, duration=10)
        second = self.ledger.claim(self.run_id, "second", clock=110, duration=10)
        self.assertEqual(second, first + 1)
        with self.assertRaises(ConflictError):
            self.ledger.complete(self.run_id, "first", first, self.result(), clock=111)
        self.ledger.complete(self.run_id, "second", second, self.result(), clock=111)
        self.assertEqual(self.ledger.get_run(self.run_id)["state"], "complete")

    def test_exact_expiry_is_not_valid(self):
        revision = self.ledger.claim(self.run_id, "worker", clock=100, duration=10)
        with self.assertRaises(ConflictError):
            self.ledger.complete(self.run_id, "worker", revision, self.result(), clock=110)

    def test_heartbeat_extends_but_does_not_change_fencing_revision(self):
        revision = self.ledger.claim(self.run_id, "worker", clock=100, duration=10)
        self.ledger.heartbeat(self.run_id, "worker", revision, clock=109, duration=20)
        run = self.ledger.get_run(self.run_id)
        self.assertEqual(run["lease_until"], 129)
        self.assertEqual(run["revision"], revision)
        self.ledger.complete(self.run_id, "worker", revision, self.result(), clock=120)

    def test_heartbeat_does_not_resurrect_expired_lease(self):
        revision = self.ledger.claim(self.run_id, "worker", clock=100, duration=10)
        with self.assertRaises(ConflictError):
            self.ledger.heartbeat(self.run_id, "worker", revision, clock=110)

    def test_failed_run_is_retryable(self):
        revision = self.ledger.claim(self.run_id, "worker", clock=100, duration=10)
        self.ledger.fail(self.run_id, "worker", revision, "invalid migration", clock=101)
        run = self.ledger.get_run(self.run_id)
        self.assertEqual(run["state"], "failed")
        self.assertEqual(run["error"], "invalid migration")
        retry = self.ledger.claim(self.run_id, "retry", clock=102, duration=10)
        self.assertEqual(retry, revision + 1)
        self.assertIsNone(self.ledger.get_run(self.run_id)["error"])
        self.ledger.complete(self.run_id, "retry", retry, self.result(), clock=103)

    def test_completed_run_cannot_be_reclaimed(self):
        self.complete()
        with self.assertRaises(ConflictError):
            self.ledger.claim(self.run_id, "another", clock=1000)

    def test_scenario_mismatch_rolls_back_completion(self):
        revision = self.ledger.claim(self.run_id, "worker", clock=100, duration=100)
        with self.assertRaises(InputError):
            self.ledger.complete(self.run_id, "worker", revision, {"scenario": "wrong"}, clock=101)
        run = self.ledger.get_run(self.run_id)
        self.assertEqual(run["state"], "running")
        self.assertIsNone(run["result"])

    def test_digest_survives_database_roundtrip(self):
        _, fingerprint = self.complete()
        run = self.ledger.get_run(self.run_id)
        self.assertEqual(run["result"], self.result())
        self.assertEqual(digest(run["result"]), fingerprint)
        self.ledger.close()
        with Ledger(self.path) as reopened:
            self.assertEqual(reopened.get_run(self.run_id)["result_digest"], fingerprint)

    def test_release_is_idempotent_and_readable(self):
        _, fingerprint = self.complete()
        first, created = self.ledger.publish(self.run_id, "2026-plan")
        second, repeated = self.ledger.publish(self.run_id, "2026-plan")
        self.assertTrue(created)
        self.assertFalse(repeated)
        self.assertEqual(first, second)
        release = self.ledger.get_release("2026-plan")
        self.assertEqual(release["digest"], fingerprint)
        self.assertEqual(release["result"], self.result())

    def test_pending_run_cannot_be_released(self):
        with self.assertRaises(ConflictError):
            self.ledger.publish(self.run_id, "premature")
        self.assertEqual(self.ledger.verify()["releases"], 0)

    def test_release_label_cannot_be_reassigned(self):
        self.complete()
        self.ledger.publish(self.run_id, "frozen")
        other_id, _ = self.ledger.create_run(self.batch_id, "baseline", "other-request")
        revision = self.ledger.claim(other_id, "other", clock=100, duration=10)
        self.ledger.complete(other_id, "other", revision, self.result(), clock=101)
        with self.assertRaises(ConflictError):
            self.ledger.publish(other_id, "frozen")
        self.assertEqual(self.ledger.get_release("frozen")["run_id"], self.run_id)

    def test_payload_corruption_is_detected(self):
        self.complete()
        self.ledger.connection.execute("UPDATE runs SET result=? WHERE id=?", ('{"scenario":"baseline","corrupted":true}', self.run_id))
        with self.assertRaises(ConflictError):
            self.ledger.get_run(self.run_id)

    def test_batch_corruption_is_detected(self):
        self.ledger.connection.execute("UPDATE batches SET payload=? WHERE id=?", ('{"source":"changed"}', self.batch_id))
        with self.assertRaises(ConflictError):
            self.ledger.get_batch(self.batch_id)

    def test_transaction_rolls_back_journal_together_with_state(self):
        before = self.ledger.history(self.run_id)
        with self.assertRaises(RuntimeError):
            with self.ledger.transaction():
                self.ledger._journal(self.run_id, "uncommitted", {"value": 1})
                self.ledger.connection.execute("UPDATE runs SET state='failed' WHERE id=?", (self.run_id,))
                raise RuntimeError("synthetic interruption")
        self.assertEqual(self.ledger.get_run(self.run_id)["state"], "pending")
        self.assertEqual(self.ledger.history(self.run_id), before)

    def test_nested_transactions_are_rejected(self):
        with self.ledger.transaction():
            with self.assertRaises(ConflictError):
                with self.ledger.transaction():
                    self.fail("nested transaction unexpectedly opened")

    def test_journal_pagination_has_no_duplicate_sequence(self):
        self.complete()
        self.ledger.publish(self.run_id, "published")
        first = self.ledger.history(self.run_id, limit=2)
        second = self.ledger.history(self.run_id, after=first[-1]["sequence"], limit=2)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertLess(first[-1]["sequence"], second[0]["sequence"])
        self.assertEqual(second[-1]["action"], "release_published")

    def test_second_connection_observes_committed_state(self):
        with Ledger(self.path) as observer:
            self.assertEqual(observer.get_run(self.run_id)["state"], "pending")
            self.complete()
            self.assertEqual(observer.get_run(self.run_id)["state"], "complete")
            self.assertEqual(observer.verify(), {"batches": 1, "runs": 1, "releases": 0, "integrity": "ok"})

    def test_foreign_key_prevents_orphan_run(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.connection.execute("DELETE FROM batches WHERE id=?", (self.batch_id,))
        self.assertEqual(self.ledger.get_run(self.run_id)["batch_id"], self.batch_id)

    def test_expected_revision_prevents_stale_claim(self):
        revision = self.ledger.claim(self.run_id, "first", clock=100, duration=10)
        with self.assertRaises(ConflictError):
            self.ledger.claim(self.run_id, "stale", clock=110, expected_revision=0)
        recovered = self.ledger.claim(self.run_id, "fresh", clock=110, expected_revision=revision)
        self.assertEqual(recovered, revision + 1)
