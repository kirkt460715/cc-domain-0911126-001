from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

from cohorts.records import InputError, integer, name


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS batches(
 id TEXT PRIMARY KEY,
 digest TEXT NOT NULL UNIQUE,
 payload TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs(
 id TEXT PRIMARY KEY,
 batch_id TEXT NOT NULL REFERENCES batches(id),
 request_key TEXT NOT NULL UNIQUE,
 scenario TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('pending','running','complete','failed')),
 revision INTEGER NOT NULL DEFAULT 0,
 result TEXT,
 result_digest TEXT,
 error TEXT,
 lease_owner TEXT,
 lease_until INTEGER,
 created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS releases(
 sequence INTEGER PRIMARY KEY AUTOINCREMENT,
 label TEXT NOT NULL UNIQUE,
 run_id TEXT NOT NULL REFERENCES runs(id),
 digest TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS journal(
 sequence INTEGER PRIMARY KEY AUTOINCREMENT,
 entity TEXT NOT NULL,
 action TEXT NOT NULL,
 payload TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_state ON runs(state,lease_until);
CREATE INDEX IF NOT EXISTS journal_entity ON journal(entity,sequence);
"""


class ConflictError(RuntimeError):
    pass


def canonical(value):
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=10000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(SCHEMA)
        versions = list(self.connection.execute("SELECT version FROM schema_version"))
        if not versions:
            self.connection.execute("INSERT INTO schema_version VALUES(1)")
        elif len(versions) != 1 or versions[0][0] != 1:
            self.close()
            raise InputError("database", "unsupported schema version")

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @contextmanager
    def transaction(self):
        if self.connection.in_transaction:
            raise ConflictError("nested transactions are not supported")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def _journal(self, entity, action, payload):
        self.connection.execute(
            "INSERT INTO journal(entity,action,payload,created_at) VALUES(?,?,?,?)",
            (entity, action, canonical(payload), timestamp()),
        )

    def put_batch(self, payload):
        encoded = canonical(payload)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        with self.transaction():
            existing = self.connection.execute("SELECT id,payload FROM batches WHERE digest=?", (fingerprint,)).fetchone()
            if existing is not None:
                if existing["payload"] != encoded:
                    raise ConflictError("batch digest collision")
                return existing["id"], False
            batch_id = "batch-" + uuid.uuid4().hex
            self.connection.execute(
                "INSERT INTO batches(id,digest,payload,created_at) VALUES(?,?,?,?)",
                (batch_id, fingerprint, encoded, timestamp()),
            )
            self._journal(batch_id, "batch_imported", {"digest": fingerprint})
        return batch_id, True

    def get_batch(self, batch_id):
        batch_id = name(batch_id, "batch_id")
        row = self.connection.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if row is None:
            raise InputError("batch_id", "batch not found")
        payload = json.loads(row["payload"])
        if digest(payload) != row["digest"]:
            raise ConflictError("batch integrity check failed")
        return {"id": row["id"], "digest": row["digest"], "created_at": row["created_at"], "payload": payload}

    def create_run(self, batch_id, scenario, request_key):
        scenario = name(scenario, "scenario")
        request_key = name(request_key, "request_key")
        with self.transaction():
            self.get_batch(batch_id)
            existing = self.connection.execute("SELECT * FROM runs WHERE request_key=?", (request_key,)).fetchone()
            if existing is not None:
                if existing["batch_id"] != batch_id or existing["scenario"] != scenario:
                    raise ConflictError("request key already used for a different run")
                return existing["id"], False
            run_id = "run-" + uuid.uuid4().hex
            now = timestamp()
            self.connection.execute(
                "INSERT INTO runs(id,batch_id,request_key,scenario,state,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (run_id, batch_id, request_key, scenario, "pending", now, now),
            )
            self._journal(run_id, "run_created", {"batch_id": batch_id, "scenario": scenario})
        return run_id, True

    def get_run(self, run_id):
        run_id = name(run_id, "run_id")
        row = self.connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise InputError("run_id", "run not found")
        result = dict(row)
        if result["result"] is not None:
            result["result"] = json.loads(result["result"])
            if digest(result["result"]) != result["result_digest"]:
                raise ConflictError("run result integrity check failed")
        return result

    def claim(self, run_id, owner, *, clock, duration=300, expected_revision=None):
        owner = name(owner, "owner")
        integer(clock, "clock")
        integer(duration, "duration", 1, 86400)
        with self.transaction():
            row = self.get_run(run_id)
            if expected_revision is not None and row["revision"] != expected_revision:
                raise ConflictError("run revision changed")
            if row["state"] == "complete":
                raise ConflictError("completed run is immutable")
            if row["state"] == "running" and row["lease_until"] > clock:
                raise ConflictError("run already has an active worker")
            revision = row["revision"] + 1
            self.connection.execute(
                "UPDATE runs SET state='running',revision=?,lease_owner=?,lease_until=?,error=NULL,updated_at=? WHERE id=?",
                (revision, owner, clock + duration, timestamp(), run_id),
            )
            self._journal(run_id, "run_claimed", {"owner": owner, "revision": revision, "recovered": row["state"] == "running"})
        return revision

    def heartbeat(self, run_id, owner, revision, *, clock, duration=300):
        integer(clock, "clock")
        integer(duration, "duration", 1, 86400)
        with self.transaction():
            row = self.get_run(run_id)
            self._check_lease(row, owner, revision, clock)
            self.connection.execute(
                "UPDATE runs SET lease_until=?,updated_at=? WHERE id=?",
                (clock + duration, timestamp(), run_id),
            )
            self._journal(run_id, "lease_extended", {"owner": owner, "revision": revision, "until": clock + duration})

    def _check_lease(self, row, owner, revision, clock):
        if row["state"] != "running":
            raise ConflictError("run is not running")
        if row["lease_owner"] != owner or row["revision"] != revision:
            raise ConflictError("worker fencing token no longer valid")
        if row["lease_until"] <= clock:
            raise ConflictError("worker lease has expired")

    def complete(self, run_id, owner, revision, result, *, clock):
        encoded = canonical(result)
        fingerprint = digest(result)
        with self.transaction():
            row = self.get_run(run_id)
            self._check_lease(row, owner, revision, clock)
            if result.get("scenario") != row["scenario"]:
                raise InputError("result.scenario", "does not match requested scenario")
            self.connection.execute(
                "UPDATE runs SET state='complete',result=?,result_digest=?,lease_owner=NULL,lease_until=NULL,updated_at=? WHERE id=?",
                (encoded, fingerprint, timestamp(), run_id),
            )
            self._journal(run_id, "run_completed", {"digest": fingerprint, "revision": revision})
        return fingerprint

    def fail(self, run_id, owner, revision, error, *, clock):
        if not isinstance(error, str) or not error or len(error) > 4000:
            raise InputError("error", "must contain 1..4000 characters")
        with self.transaction():
            row = self.get_run(run_id)
            self._check_lease(row, owner, revision, clock)
            self.connection.execute(
                "UPDATE runs SET state='failed',error=?,lease_owner=NULL,lease_until=NULL,updated_at=? WHERE id=?",
                (error, timestamp(), run_id),
            )
            self._journal(run_id, "run_failed", {"error": error, "revision": revision})

    def publish(self, run_id, label):
        label = name(label, "label")
        with self.transaction():
            row = self.get_run(run_id)
            if row["state"] != "complete":
                raise ConflictError("only complete runs may be released")
            existing = self.connection.execute("SELECT * FROM releases WHERE label=?", (label,)).fetchone()
            if existing is not None:
                if existing["run_id"] != run_id or existing["digest"] != row["result_digest"]:
                    raise ConflictError("release label is immutable")
                return dict(existing), False
            self.connection.execute(
                "INSERT INTO releases(label,run_id,digest,created_at) VALUES(?,?,?,?)",
                (label, run_id, row["result_digest"], timestamp()),
            )
            self._journal(run_id, "release_published", {"label": label, "digest": row["result_digest"]})
            release = self.connection.execute("SELECT * FROM releases WHERE label=?", (label,)).fetchone()
        return dict(release), True

    def get_release(self, label):
        label = name(label, "label")
        row = self.connection.execute("SELECT * FROM releases WHERE label=?", (label,)).fetchone()
        if row is None:
            raise InputError("label", "release not found")
        run = self.get_run(row["run_id"])
        if run["state"] != "complete" or run["result_digest"] != row["digest"]:
            raise ConflictError("release integrity check failed")
        return {**dict(row), "result": run["result"]}

    def history(self, entity, *, after=0, limit=100):
        integer(after, "after")
        integer(limit, "limit", 1, 1000)
        rows = self.connection.execute(
            "SELECT * FROM journal WHERE entity=? AND sequence>? ORDER BY sequence LIMIT ?",
            (entity, after, limit),
        )
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def list_runs(self, *, state=None, after="", limit=100):
        integer(limit, "limit", 1, 1000)
        if state is not None and state not in ("pending", "running", "complete", "failed"):
            raise InputError("state", "unknown run state")
        query = "SELECT id FROM runs WHERE id>?"
        parameters = [after]
        if state is not None:
            query += " AND state=?"
            parameters.append(state)
        query += " ORDER BY id LIMIT ?"
        parameters.append(limit)
        return [self.get_run(row["id"]) for row in self.connection.execute(query, parameters)]

    def verify(self):
        check = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise ConflictError(f"SQLite integrity check failed: {check}")
        foreign_keys = list(self.connection.execute("PRAGMA foreign_key_check"))
        if foreign_keys:
            raise ConflictError("foreign key integrity check failed")
        batches = list(self.connection.execute("SELECT id FROM batches"))
        runs = list(self.connection.execute("SELECT id FROM runs"))
        releases = list(self.connection.execute("SELECT label FROM releases"))
        for row in batches:
            self.get_batch(row["id"])
        for row in runs:
            self.get_run(row["id"])
        for row in releases:
            self.get_release(row["label"])
        return {"batches": len(batches), "runs": len(runs), "releases": len(releases), "integrity": "ok"}
