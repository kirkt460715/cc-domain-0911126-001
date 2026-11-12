from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from cohorts.records import InputError
from .capacity import compare_pressure, ratio


def scenario_summary(result):
    grouped = defaultdict(list)
    for row in result["pressure"]:
        grouped[row["year"]].append(row)
    output = []
    for year, rows in sorted(grouped.items()):
        population = sum(row["count"] for row in result["population"] if row["year"] == year)
        pupils = sum(row["pupils"] for row in rows)
        seats = sum(row["seats"] for row in rows)
        shortage = sum(row["seat_shortage"] for row in rows)
        spare = sum(row["spare_seats"] for row in rows)
        maximum_utilization = max((Decimal(row["utilization"]) for row in rows if row["utilization"] is not None), default=None)
        output.append({
            "scenario": result["scenario"],
            "year": year,
            "population": population,
            "pupils": pupils,
            "seats": seats,
            "locally_unmet_pupils": shortage,
            "spare_seats": spare,
            "net_seat_gap": pupils - seats,
            "shortage_rate": ratio(shortage, pupils),
            "utilization": ratio(pupils, seats),
            "maximum_local_utilization": str(maximum_utilization) if maximum_utilization is not None else None,
            "unserved_cells": sum(row["status"] == "unserved" for row in rows),
            "critical_cells": sum(row["status"] == "critical" for row in rows),
            "watch_cells": sum(row["status"] == "watch" for row in rows),
            "legal_class_shortage": sum(row["legal_class_shortage"] for row in rows),
            "target_class_shortage": sum(row["target_class_shortage"] for row in rows),
        })
    return output


def rank_hotspots(result, *, year=None, limit=20):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise InputError("limit", "expected integer in 1..1000")
    rows = [row for row in result["pressure"] if year is None or row["year"] == year]
    rows.sort(key=lambda row: (
        -row["seat_shortage"],
        -row["legal_class_shortage"],
        -row["target_class_shortage"],
        row["year"], row["region"], row["settlement"], row["stage"],
    ))
    return [{"rank": index, **row} for index, row in enumerate(rows[:limit], 1)]


def compare_scenarios(baseline, alternative):
    if baseline["batch_digest"] != alternative["batch_digest"]:
        raise InputError("comparison", "scenarios must come from the same input batch")
    if baseline["baseline_year"] != alternative["baseline_year"] or baseline["through_year"] != alternative["through_year"]:
        raise InputError("comparison", "forecast calendars differ")
    differences = compare_pressure(baseline["pressure"], alternative["pressure"])
    grouped = defaultdict(list)
    for row in differences:
        grouped[row["year"]].append(row)
    summary = []
    for year, rows in sorted(grouped.items()):
        summary.append({
            "year": year,
            "pupil_change": sum(row["pupil_change"] for row in rows),
            "seat_change": sum(row["seat_change"] for row in rows),
            "local_shortage_change": sum(row["shortage_change"] for row in rows),
            "legal_class_shortage_change": sum(row["class_shortage_change"] for row in rows),
            "cells_with_reduced_shortage": sum(row["shortage_change"] < 0 for row in rows),
            "cells_with_increased_shortage": sum(row["shortage_change"] > 0 for row in rows),
        })
    return {"baseline": baseline["scenario"], "alternative": alternative["scenario"], "summary": summary, "differences": differences}


def envelope(results):
    results = list(results)
    if not results:
        raise InputError("results", "at least one scenario required")
    if len({result["scenario"] for result in results}) != len(results):
        raise InputError("results", "duplicate scenario")
    if len({result["batch_digest"] for result in results}) != 1:
        raise InputError("results", "results belong to different batches")
    series = defaultdict(list)
    reference_cells = None
    for result in results:
        cells = set()
        for row in result["pressure"]:
            key = tuple(row[field] for field in ("year", "region", "settlement", "stage"))
            if key in cells:
                raise InputError("results", "duplicate pressure cell")
            cells.add(key)
            series[key].append((result["scenario"], row))
        if reference_cells is None:
            reference_cells = cells
        elif cells != reference_cells:
            raise InputError("results", "scenario cells do not align")
    output = []
    for key, rows in sorted(series.items()):
        pupil_minimum = min(row["pupils"] for _, row in rows)
        pupil_maximum = max(row["pupils"] for _, row in rows)
        shortage_minimum = min(row["seat_shortage"] for _, row in rows)
        shortage_maximum = max(row["seat_shortage"] for _, row in rows)
        output.append({
            **dict(zip(("year", "region", "settlement", "stage"), key)),
            "scenario_count": len(rows),
            "pupil_minimum": pupil_minimum,
            "pupil_maximum": pupil_maximum,
            "pupil_range": pupil_maximum - pupil_minimum,
            "shortage_minimum": shortage_minimum,
            "shortage_maximum": shortage_maximum,
            "shortage_in_every_scenario": shortage_minimum > 0,
            "shortage_in_some_scenario": shortage_maximum > 0,
            "highest_demand_scenarios": sorted(code for code, row in rows if row["pupils"] == pupil_maximum),
            "lowest_demand_scenarios": sorted(code for code, row in rows if row["pupils"] == pupil_minimum),
        })
    return output
