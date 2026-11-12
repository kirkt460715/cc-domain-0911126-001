from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from cohorts.records import InputError
from .ledger import ConflictError, canonical, digest


PRESSURE_COLUMNS = (
    "year", "region", "settlement", "stage", "pupils", "seats", "classrooms",
    "accessible_seats", "utilization", "seat_shortage", "spare_seats",
    "target_class_shortage", "legal_class_shortage", "status",
)
ACCOUNT_COLUMNS = (
    "from_year", "to_year", "opening", "internal_moved", "external_arrivals",
    "external_departures", "deaths", "aged_out", "births", "closing", "balance",
)
POPULATION_COLUMNS = ("year", "region", "settlement", "age", "count")


def safe_component(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise InputError("filename", "expected a safe file component")
    if value in (".", ".."):
        raise InputError("filename", "relative path components are forbidden")
    return value


def atomic_text(path, text):
    path = Path(path)
    if path.is_symlink():
        raise InputError("output", "output path cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def csv_text(rows, columns):
    import io
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        absent = set(columns) - set(row)
        if absent:
            raise InputError("export", f"missing columns {sorted(absent)}")
        writer.writerow({key: "" if row[key] is None else row[key] for key in columns})
    return output.getvalue()


def verify_export(directory, *, maximum_file_bytes=64 * 1024 * 1024):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise InputError("export", "export must be a regular directory")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise InputError("manifest", "manifest missing or not a regular file")
    if manifest_path.stat().st_size > 1024 * 1024:
        raise InputError("manifest", "manifest too large")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise InputError("manifest", "manifest is not valid UTF-8 JSON") from exc
    if manifest.get("schema_version") != 1:
        raise InputError("manifest", "unsupported manifest version")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise InputError("manifest", "missing file inventory")
    observed = {path.name for path in directory.iterdir()}
    if observed != set(files) | {"manifest.json"}:
        raise ConflictError("export contains missing or unexpected files")
    for filename, specification in files.items():
        safe_component(filename)
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise InputError("export", f"file {filename} is not regular")
        if path.stat().st_size > maximum_file_bytes:
            raise InputError("export", f"file {filename} exceeds limit")
        content = path.read_bytes()
        fingerprint = hashlib.sha256(content).hexdigest()
        if fingerprint != specification.get("sha256") or len(content) != specification.get("bytes"):
            raise ConflictError(f"export integrity check failed for {filename}")
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if digest(result) != manifest.get("result_digest"):
        raise ConflictError("result digest does not match export manifest")
    return manifest


def export_release(ledger, label, destination):
    release = ledger.get_release(label)
    result = release["result"]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        manifest = verify_export(destination)
        if manifest.get("release_label") != label or manifest.get("result_digest") != release["digest"]:
            raise ConflictError("existing export belongs to a different release")
        return manifest, False
    temporary = Path(tempfile.mkdtemp(prefix=".release-", dir=destination.parent))
    try:
        contents = {
            "result.json": canonical(result) + "\n",
            "pressure.csv": csv_text(result["pressure"], PRESSURE_COLUMNS),
            "population.csv": csv_text(result["population"], POPULATION_COLUMNS),
            "accounting.csv": csv_text(result["accounting"], ACCOUNT_COLUMNS),
        }
        files = {}
        for filename, content in contents.items():
            atomic_text(temporary / filename, content)
            raw = content.encode("utf-8")
            files[filename] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        manifest = {
            "schema_version": 1,
            "release_label": label,
            "release_sequence": release["sequence"],
            "run_id": release["run_id"],
            "result_digest": release["digest"],
            "batch_digest": result["batch_digest"],
            "scenario": result["scenario"],
            "files": files,
        }
        atomic_text(temporary / "manifest.json", canonical(manifest) + "\n")
        verify_export(temporary)
        os.rename(temporary, destination)
        return manifest, True
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_export_result(directory):
    manifest = verify_export(directory)
    result = json.loads((Path(directory) / "result.json").read_text(encoding="utf-8"))
    if result["scenario"] != manifest["scenario"] or result["batch_digest"] != manifest["batch_digest"]:
        raise ConflictError("result identity differs from manifest")
    return result
