from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json

from cohorts.records import InputError, integer
from school_pressure.capacity import ratio
from .ledger import canonical


DIMENSIONS = ("year", "region", "settlement", "stage")
MEASURES = (
    "pupils", "seats", "classrooms", "accessible_seats", "seat_shortage",
    "spare_seats", "target_class_shortage", "legal_class_shortage",
)
OUTPUT_COLUMNS = DIMENSIONS + MEASURES + ("status", "utilization")
STATUSES = {"unserved", "critical", "watch", "adequate"}


@dataclass(frozen=True)
class PressureQuery:
    years: tuple = ()
    regions: tuple = ()
    settlements: tuple = ()
    stages: tuple = ()
    statuses: tuple = ()
    minimum_shortage: int = 0
    limit: int = 100
    columns: tuple = OUTPUT_COLUMNS

    def __post_init__(self):
        integer(self.minimum_shortage, "minimum_shortage")
        integer(self.limit, "limit", 1, 1000)
        for year in self.years:
            integer(year, "year", 1900, 2200)
        if set(self.settlements) - {"urban", "rural"}:
            raise InputError("settlements", "unknown settlement")
        if set(self.statuses) - STATUSES:
            raise InputError("statuses", "unknown pressure status")
        if not self.columns or set(self.columns) - set(OUTPUT_COLUMNS):
            raise InputError("columns", "unknown or empty output columns")
        if len(set(self.columns)) != len(self.columns):
            raise InputError("columns", "duplicate output columns")
        for attribute in ("years", "regions", "settlements", "stages", "statuses"):
            values = getattr(self, attribute)
            if len(values) != len(set(values)):
                raise InputError(attribute, "duplicate filter value")

    def identity(self):
        return {
            "years": sorted(self.years),
            "regions": sorted(self.regions),
            "settlements": sorted(self.settlements),
            "stages": sorted(self.stages),
            "statuses": sorted(self.statuses),
            "minimum_shortage": self.minimum_shortage,
            "columns": self.columns,
        }

    def matches(self, row):
        filters = (
            (self.years, "year"),
            (self.regions, "region"),
            (self.settlements, "settlement"),
            (self.stages, "stage"),
            (self.statuses, "status"),
        )
        if row["seat_shortage"] < self.minimum_shortage:
            return False
        return all(not values or row[field] in values for values, field in filters)


def query_fingerprint(release_digest, query):
    return hashlib.sha256(canonical({"release": release_digest, "query": query.identity()}).encode()).hexdigest()


def encode_cursor(fingerprint, key):
    raw = canonical({"version": 1, "fingerprint": fingerprint, "after": key}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor, fingerprint):
    if not isinstance(cursor, str) or len(cursor) > 4096:
        raise InputError("cursor", "cursor must be a short string")
    try:
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        payload = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise InputError("cursor", "invalid cursor encoding") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise InputError("cursor", "unsupported cursor version")
    if payload.get("fingerprint") != fingerprint:
        raise InputError("cursor", "cursor belongs to a different query or release")
    key = payload.get("after")
    if not isinstance(key, list) or len(key) != len(DIMENSIONS):
        raise InputError("cursor", "invalid ordering key")
    integer(key[0], "cursor.year", 1900, 2200)
    if any(not isinstance(value, str) for value in key[1:]):
        raise InputError("cursor", "invalid dimension type")
    return tuple(key)


def pressure_page(release, query=None, *, cursor=None):
    query = query or PressureQuery()
    fingerprint = query_fingerprint(release["digest"], query)
    after = decode_cursor(cursor, fingerprint) if cursor is not None else None
    rows = []
    identities = set()
    for row in release["result"]["pressure"]:
        key = tuple(row[field] for field in DIMENSIONS)
        if key in identities:
            raise InputError("release", "duplicate pressure identity")
        identities.add(key)
        if not query.matches(row):
            continue
        rows.append((key, row))
    rows.sort(key=lambda pair: pair[0])
    selected = [(key, row) for key, row in rows if after is None or key > after]
    page = selected[:query.limit]
    has_more = len(selected) > query.limit
    next_cursor = encode_cursor(fingerprint, page[-1][0]) if has_more and page else None
    return {
        "release_label": release["label"],
        "release_digest": release["digest"],
        "matched_cells": len(rows),
        "returned_cells": len(page),
        "rows": [{column: row[column] for column in query.columns} for _, row in page],
        "next_cursor": next_cursor,
    }


def pressure_totals(release, query=None, *, group_by=()):
    query = query or PressureQuery()
    if set(group_by) - set(DIMENSIONS) or len(set(group_by)) != len(group_by):
        raise InputError("group_by", "unknown or duplicate dimensions")
    buckets = {}
    for row in release["result"]["pressure"]:
        if not query.matches(row):
            continue
        key = tuple(row[field] for field in group_by)
        bucket = buckets.setdefault(key, {**{measure: 0 for measure in MEASURES}, "cells": 0})
        bucket["cells"] += 1
        for measure in MEASURES:
            bucket[measure] += row[measure]
    output = []
    for key, bucket in sorted(buckets.items()):
        output.append({
            **dict(zip(group_by, key)),
            **bucket,
            "utilization": ratio(bucket["pupils"], bucket["seats"]),
            "shortage_rate": ratio(bucket["seat_shortage"], bucket["pupils"]),
            "net_seat_gap": bucket["pupils"] - bucket["seats"],
        })
    return output


def release_facets(release, query=None):
    query = query or PressureQuery()
    values = {dimension: set() for dimension in DIMENSIONS}
    statuses = set()
    for row in release["result"]["pressure"]:
        if not query.matches(row):
            continue
        for dimension in DIMENSIONS:
            values[dimension].add(row[dimension])
        statuses.add(row["status"])
    return {**{dimension: sorted(items) for dimension, items in values.items()}, "status": sorted(statuses)}
