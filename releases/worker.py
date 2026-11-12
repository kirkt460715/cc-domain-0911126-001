from __future__ import annotations

from dataclasses import dataclass
import time
import uuid

from cohorts.records import InputError, integer
from forecast.batches import Batch
from forecast.quality import audit
from .ledger import ConflictError, Ledger


@dataclass(frozen=True)
class WorkResult:
    run_id: str
    state: str
    reused: bool
    result_digest: str | None
    error: str | None

    def as_dict(self):
        return {
            "run_id": self.run_id,
            "state": self.state,
            "reused": self.reused,
            "result_digest": self.result_digest,
            "error": self.error,
        }


class Worker:
    def __init__(self, database, *, clock=None, lease_seconds=300):
        integer(lease_seconds, "lease_seconds", 10, 86400)
        self.database = database
        self.clock = clock or (lambda: int(time.time()))
        self.lease_seconds = lease_seconds
        self.owner = "worker-" + uuid.uuid4().hex

    def execute(self, run_id):
        with Ledger(self.database) as ledger:
            run = ledger.get_run(run_id)
            if run["state"] == "complete":
                return WorkResult(run_id, "complete", True, run["result_digest"], None)
            now = self.clock()
            revision = ledger.claim(run_id, self.owner, clock=now, duration=self.lease_seconds, expected_revision=run["revision"])
            last_heartbeat = now

            def progress(year, first, last):
                nonlocal last_heartbeat
                current = self.clock()
                if current < last_heartbeat:
                    raise ConflictError("worker clock moved backwards")
                if current - last_heartbeat >= self.lease_seconds // 2:
                    ledger.heartbeat(run_id, self.owner, revision, clock=current, duration=self.lease_seconds)
                    last_heartbeat = current

            try:
                batch = Batch.parse(ledger.get_batch(run["batch_id"])["payload"])
                result = batch.run(run["scenario"], progress=progress)
                quality = audit(batch, [result])
                if not quality["passed"]:
                    raise InputError("quality", "forecast output failed arithmetic audit")
                result["quality"] = quality
                fingerprint = ledger.complete(run_id, self.owner, revision, result, clock=self.clock())
                return WorkResult(run_id, "complete", False, fingerprint, None)
            except (InputError, ArithmeticError) as exc:
                ledger.fail(run_id, self.owner, revision, str(exc), clock=self.clock())
                return WorkResult(run_id, "failed", False, None, str(exc))

    def candidates(self, *, include_failed=False, limit=100):
        integer(limit, "limit", 1, 1000)
        states = ["pending", "running"]
        if include_failed:
            states.append("failed")
        now = self.clock()
        with Ledger(self.database) as ledger:
            selected = []
            for state in states:
                after = ""
                while len(selected) < limit:
                    page = ledger.list_runs(state=state, after=after, limit=min(100, limit))
                    if not page:
                        break
                    for run in page:
                        if run["state"] == "running" and run["lease_until"] > now:
                            continue
                        selected.append(run["id"])
                        if len(selected) == limit:
                            break
                    after = page[-1]["id"]
                    if len(page) < min(100, limit):
                        break
                if len(selected) == limit:
                    break
        return tuple(selected)

    def drain(self, *, include_failed=False, limit=100):
        candidates = self.candidates(include_failed=include_failed, limit=limit)
        completed = []
        failed = []
        conflicted = []
        for run_id in candidates:
            try:
                outcome = self.execute(run_id)
            except ConflictError as exc:
                conflicted.append({"run_id": run_id, "message": str(exc)})
                continue
            if outcome.state == "complete":
                completed.append(outcome.as_dict())
            else:
                failed.append(outcome.as_dict())
        return {
            "candidate_count": len(candidates),
            "completed": completed,
            "failed": failed,
            "conflicted": conflicted,
        }


def queue_status(database, *, clock=None):
    now = int(time.time()) if clock is None else clock
    counts = {state: 0 for state in ("pending", "running", "complete", "failed")}
    expired = 0
    after = ""
    with Ledger(database) as ledger:
        while True:
            rows = ledger.list_runs(after=after, limit=1000)
            if not rows:
                break
            for row in rows:
                counts[row["state"]] += 1
                if row["state"] == "running" and row["lease_until"] <= now:
                    expired += 1
            after = rows[-1]["id"]
    return {"counts": counts, "expired_running": expired, "observed_at": now}
