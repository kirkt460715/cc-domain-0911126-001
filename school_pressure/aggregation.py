from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from cohorts.records import InputError, expand_ancestors
from .capacity import ratio


ADDITIVE = (
    "pupils", "seats", "classrooms", "accessible_seats", "seat_shortage",
    "spare_seats", "target_class_shortage", "legal_class_shortage",
)


def aggregate(rows, regions, *, group_settlement=True, group_stage=True):
    buckets = {}
    seen = set()
    parents = {region.parent for region in regions.values() if region.parent is not None}
    for row in rows:
        identity = tuple(row[field] for field in ("year", "region", "settlement", "stage"))
        if identity in seen:
            raise InputError("aggregation", "duplicate pressure cell")
        seen.add(identity)
        if row["region"] in parents:
            raise InputError("aggregation", "input must contain only leaf regions")
        for region in expand_ancestors(row["region"], regions):
            key = (
                row["year"], region,
                row["settlement"] if group_settlement else "all",
                row["stage"] if group_stage else "all",
            )
            bucket = buckets.setdefault(key, {
                **{metric: 0 for metric in ADDITIVE},
                "leaf_cells": 0,
                "critical_cells": 0,
                "unserved_cells": 0,
                "watch_cells": 0,
            })
            for metric in ADDITIVE:
                bucket[metric] += row[metric]
            bucket["leaf_cells"] += 1
            bucket["critical_cells"] += row["status"] == "critical"
            bucket["unserved_cells"] += row["status"] == "unserved"
            bucket["watch_cells"] += row["status"] == "watch"
    result = []
    for key, bucket in sorted(buckets.items()):
        result.append({
            **dict(zip(("year", "region", "settlement", "stage"), key)),
            **bucket,
            "utilization": ratio(bucket["pupils"], bucket["seats"]),
            "net_seat_gap": bucket["pupils"] - bucket["seats"],
            "locally_unmet_pupils": bucket["seat_shortage"],
        })
    return result


def rural_urban_gaps(rows):
    buckets = defaultdict(dict)
    for row in rows:
        key = row["year"], row["region"], row["stage"]
        settlement = row["settlement"]
        if settlement not in ("urban", "rural"):
            raise InputError("gaps", "requires separate urban and rural rows")
        if settlement in buckets[key]:
            raise InputError("gaps", "duplicate settlement")
        buckets[key][settlement] = row
    result = []
    for key, pair in sorted(buckets.items()):
        urban = pair.get("urban")
        rural = pair.get("rural")
        complete = urban is not None and rural is not None
        urban_rate = ratio(urban["seat_shortage"], urban["pupils"]) if urban else None
        rural_rate = ratio(rural["seat_shortage"], rural["pupils"]) if rural else None
        gap = None
        if complete and urban_rate is not None and rural_rate is not None:
            gap = str(Decimal(rural_rate) - Decimal(urban_rate))
        result.append({
            **dict(zip(("year", "region", "stage"), key)),
            "complete": complete,
            "urban_shortage_rate": urban_rate,
            "rural_shortage_rate": rural_rate,
            "rural_minus_urban": gap,
            "missing_settlements": [settlement for settlement in ("urban", "rural") if settlement not in pair],
        })
    return result


def turning_points(rows):
    series = defaultdict(dict)
    for row in rows:
        key = row["region"], row["settlement"], row["stage"]
        if row["year"] in series[key]:
            raise InputError("trends", "duplicate year")
        series[key][row["year"]] = row
    result = []
    for key, values in sorted(series.items()):
        years = sorted(values)
        first = years[0]
        last = years[-1]
        if years != list(range(first, last + 1)):
            raise InputError("trends", "annual series contains gaps")
        peak = max(values[year]["pupils"] for year in years)
        peaks = [year for year in years if values[year]["pupils"] == peak]
        first_shortage = next((year for year in years if values[year]["seat_shortage"] > 0), None)
        changes = []
        for before, after in zip(years, years[1:]):
            delta = values[after]["pupils"] - values[before]["pupils"]
            changes.append({"year": after, "delta": delta, "growth_rate": ratio(delta, values[before]["pupils"])})
        result.append({
            **dict(zip(("region", "settlement", "stage"), key)),
            "first_year": first,
            "last_year": last,
            "peak_pupils": peak,
            "peak_years": peaks,
            "first_shortage_year": first_shortage,
            "pupil_change": values[last]["pupils"] - values[first]["pupils"],
            "annual_changes": changes,
        })
    return result
