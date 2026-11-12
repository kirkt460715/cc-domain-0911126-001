from __future__ import annotations

from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile

from cohorts.records import InputError
from .ledger import ConflictError, Ledger


def backup(ledger, destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ConflictError("backup destination already exists")
    if ledger.connection.in_transaction:
        raise ConflictError("backup cannot run inside a transaction")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=".backup-", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(filename)
    try:
        with closing(sqlite3.connect(temporary)) as target:
            ledger.connection.backup(target, pages=100)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ConflictError("backup failed SQLite integrity check")
        with Ledger(temporary) as verifier:
            counts = verifier.verify()
            verifier.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            verifier.connection.execute("PRAGMA journal_mode=DELETE")
        raw = temporary.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        return {"destination": str(destination), "bytes": len(raw), "sha256": fingerprint, "integrity": counts}
    finally:
        if temporary.exists():
            temporary.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(temporary) + suffix)
            if sidecar.exists():
                sidecar.unlink()


def restore(source, destination, *, expected_sha256):
    source = Path(source)
    destination = Path(destination)
    if source.is_symlink() or not source.is_file():
        raise InputError("backup", "source must be a regular file")
    if destination.exists() or destination.is_symlink():
        raise ConflictError("restore destination already exists")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ConflictError("backup checksum mismatch")
    with closing(sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ConflictError("backup has invalid SQLite structure")
        if list(connection.execute("PRAGMA foreign_key_check")):
            raise ConflictError("backup contains orphan references")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=".restore-", dir=destination.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        with Ledger(temporary) as ledger:
            integrity = ledger.verify()
            ledger.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            ledger.connection.execute("PRAGMA journal_mode=DELETE")
        os.link(temporary, destination)
        temporary.unlink()
        return {"destination": str(destination), "source_sha256": expected_sha256, "integrity": integrity}
    finally:
        if temporary.exists():
            temporary.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(temporary) + suffix)
            if sidecar.exists():
                sidecar.unlink()
