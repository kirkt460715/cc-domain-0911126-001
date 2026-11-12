from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re

from .records import InputError, integer


TABLES = {
    "regions": {
        "required": ("code", "label"),
        "optional": ("parent",),
        "integers": (),
        "decimals": (),
    },
    "population": {
        "required": ("year", "region", "settlement", "age", "count"),
        "optional": (),
        "integers": ("year", "age", "count"),
        "decimals": (),
    },
    "stages": {
        "required": ("code", "min_age", "max_age", "class_size"),
        "optional": ("max_class_size", "participation"),
        "integers": ("min_age", "max_age", "class_size", "max_class_size"),
        "decimals": ("participation",),
    },
    "capacities": {
        "required": ("facility", "region", "settlement", "stage", "from_year", "through_year", "seats", "classrooms"),
        "optional": ("accessible_seats",),
        "integers": ("from_year", "through_year", "seats", "classrooms", "accessible_seats"),
        "decimals": (),
    },
    "migrations": {
        "required": ("scenario", "year", "age", "source", "source_settlement", "destination", "destination_settlement", "count"),
        "optional": (),
        "integers": ("year", "age", "count"),
        "decimals": (),
    },
    "births": {
        "required": ("scenario", "year", "region", "settlement", "count"),
        "optional": (),
        "integers": ("year", "count"),
        "decimals": (),
    },
    "adjustments": {
        "required": ("scenario", "year", "region", "settlement", "age", "reason"),
        "optional": ("arrivals", "departures"),
        "integers": ("year", "age", "arrivals", "departures"),
        "decimals": (),
    },
    "survival": {
        "required": ("scenario", "age", "rate"),
        "optional": (),
        "integers": ("age",),
        "decimals": ("rate",),
    },
    "participation": {
        "required": ("scenario", "region", "settlement", "stage", "rate"),
        "optional": (),
        "integers": (),
        "decimals": ("rate",),
    },
    "scenarios": {
        "required": ("code", "horizon"),
        "optional": ("parent",),
        "integers": ("horizon",),
        "decimals": (),
    },
}


@dataclass(frozen=True)
class RowIssue:
    table: str
    line: int
    field: str
    message: str

    def as_dict(self):
        return {"table": self.table, "line": self.line, "field": self.field, "message": self.message}


@dataclass(frozen=True)
class TableRead:
    table: str
    rows: tuple
    issues: tuple

    def require_valid(self):
        if self.issues:
            issue = self.issues[0]
            raise InputError(f"{issue.table}:{issue.line}:{issue.field}", issue.message)
        return self.rows


def parse_integer(text, field):
    if not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise InputError(field, "expected canonical nonnegative integer")
    return integer(int(text), field)


def read_table(path, table, *, maximum_rows=100000, maximum_bytes=16 * 1024 * 1024):
    if table not in TABLES:
        raise InputError("table", f"unknown table {table}")
    definition = TABLES[table]
    path = Path(path)
    with path.open("rb") as source:
        raw = source.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise InputError("table", "CSV exceeds byte limit")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError("table", "CSV must be UTF-8") from exc
    if "\x00" in text:
        raise InputError("table", "CSV contains NUL byte")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise InputError("table", "CSV is empty") from exc
    except csv.Error as exc:
        raise InputError("table", f"invalid CSV header: {exc}") from exc
    if len(header) != len(set(header)):
        raise InputError("header", "duplicate columns")
    allowed = set(definition["required"]) | set(definition["optional"])
    unknown = set(header) - allowed
    missing = set(definition["required"]) - set(header)
    if unknown:
        raise InputError("header", f"unknown columns {sorted(unknown)}")
    if missing:
        raise InputError("header", f"missing columns {sorted(missing)}")
    rows = []
    issues = []
    try:
        for number, cells in enumerate(reader, 1):
            line = reader.line_num
            if number > maximum_rows:
                raise InputError("table", "CSV exceeds row limit")
            if not cells:
                continue
            if len(cells) != len(header):
                issues.append(RowIssue(table, line, "row", f"expected {len(header)} cells, received {len(cells)}"))
                continue
            record = {}
            row_valid = True
            for field, raw_value in zip(header, cells):
                value = raw_value.strip()
                if not value:
                    if field in definition["required"]:
                        issues.append(RowIssue(table, line, field, "required value missing"))
                        row_valid = False
                    continue
                if len(value) > 4000:
                    issues.append(RowIssue(table, line, field, "value exceeds length limit"))
                    row_valid = False
                    continue
                try:
                    if field in definition["integers"]:
                        record[field] = parse_integer(value, field)
                    else:
                        record[field] = value
                except InputError as exc:
                    issues.append(RowIssue(table, line, field, str(exc)))
                    row_valid = False
            if row_valid:
                rows.append(record)
    except csv.Error as exc:
        issues.append(RowIssue(table, reader.line_num, "row", str(exc)))
    return TableRead(table, tuple(rows), tuple(issues))


def read_bundle(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise InputError("bundle", "CSV bundle directory does not exist")
    required = ("regions", "population", "stages", "capacities", "scenarios")
    tables = {}
    issues = []
    for table in TABLES:
        path = directory / (table + ".csv")
        if not path.exists():
            if table in required:
                raise InputError("bundle", f"required table {table}.csv missing")
            continue
        if path.is_symlink() or not path.is_file():
            raise InputError("bundle", f"table {table} must be a regular file")
        result = read_table(path, table)
        tables[table] = result.rows
        issues.extend(result.issues)
    if issues:
        return None, tuple(issues)
    payload = {"schema_version": 1}
    for table in ("regions", "population", "stages", "capacities"):
        payload[table] = list(tables[table])
    scenarios = {}
    for row in tables["scenarios"]:
        if row["code"] in scenarios:
            raise InputError("scenarios", "duplicate scenario code")
        scenarios[row["code"]] = dict(row)
    for table in ("migrations", "births", "adjustments", "survival", "participation"):
        for source in tables.get(table, ()):
            row = dict(source)
            scenario = row.pop("scenario")
            if scenario not in scenarios:
                raise InputError("scenario", f"table {table} refers to unknown scenario {scenario}")
            scenarios[scenario].setdefault(table, []).append(row)
    payload["scenarios"] = [scenarios[code] for code in sorted(scenarios)]
    return payload, ()


def write_table(path, table, rows):
    if table not in TABLES:
        raise InputError("table", "unknown table")
    definition = TABLES[table]
    headers = list(definition["required"]) + list(definition["optional"])
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, headers, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def export_bundle(payload, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tables = {table: [] for table in TABLES}
    for table in ("regions", "population", "stages", "capacities"):
        tables[table] = payload[table]
    for scenario in payload["scenarios"]:
        tables["scenarios"].append({key: scenario[key] for key in ("code", "horizon", "parent") if key in scenario})
        for table in ("migrations", "births", "adjustments", "survival", "participation"):
            for row in scenario.get(table, []):
                tables[table].append({"scenario": scenario["code"], **row})
    written = []
    for table, rows in tables.items():
        if rows or table in ("regions", "population", "stages", "capacities", "scenarios"):
            path = directory / (table + ".csv")
            write_table(path, table, rows)
            written.append(path.name)
    return tuple(sorted(written))
