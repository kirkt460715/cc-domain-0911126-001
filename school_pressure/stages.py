from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from cohorts.records import InputError, decimal, fields, integer, name, required
from cohorts.transitions import apportion


@dataclass(frozen=True)
class Stage:
    code: str
    min_age: int
    max_age: int
    participation: Decimal
    class_size: int
    max_class_size: int

    @classmethod
    def parse(cls, record):
        fields(record, {"code", "min_age", "max_age", "participation", "class_size", "max_class_size"}, "stage")
        minimum = integer(required(record, "min_age"), "min_age", 0, 99)
        maximum = integer(required(record, "max_age"), "max_age", minimum, 99)
        size = integer(required(record, "class_size"), "class_size", 1, 100)
        return cls(
            name(required(record, "code"), "code"),
            minimum,
            maximum,
            decimal(record.get("participation", 1), "participation"),
            size,
            integer(record.get("max_class_size", size), "max_class_size", size, 100),
        )

    def contains(self, age):
        return self.min_age <= age <= self.max_age


def validate_stages(stages):
    result = {}
    ages = {}
    for stage in stages:
        if stage.code in result:
            raise InputError("stages", f"duplicate stage {stage.code}")
        result[stage.code] = stage
        for age in range(stage.min_age, stage.max_age + 1):
            if age in ages:
                raise InputError("stages", f"age {age} assigned to multiple stages")
            ages[age] = stage.code
    if not result:
        raise InputError("stages", "at least one stage required")
    return result


def demand_by_stage(population, stages, overrides=None):
    stages = validate_stages(stages)
    overrides = overrides or {}
    raw = {}
    unmatched = 0
    for cell, count in sorted(population.items()):
        matching = [stage for stage in stages.values() if stage.contains(cell.age)]
        if not matching:
            unmatched += count
            continue
        stage = matching[0]
        key = (cell.year, cell.region, cell.settlement, stage.code)
        rate_key = (cell.region, cell.settlement, stage.code)
        rate = decimal(overrides.get(rate_key, stage.participation), "participation_override")
        if key not in raw:
            raw[key] = {"population": 0, "expected": Decimal(0), "ages": {}}
        raw[key]["population"] += count
        raw[key]["expected"] += Decimal(count) * rate
        raw[key]["ages"][cell.age] = raw[key]["ages"].get(cell.age, 0) + count
    result = []
    for key, item in sorted(raw.items()):
        year, region, settlement, code = key
        stage = stages[code]
        population_total = item["population"]
        expected = item["expected"]
        if population_total:
            pupils = apportion(population_total, {"enrolled": expected, "outside": population_total - expected})["enrolled"]
        else:
            pupils = 0
        result.append({
            "year": year,
            "region": region,
            "settlement": settlement,
            "stage": code,
            "population": population_total,
            "pupils": pupils,
            "expected_pupils": str(expected),
            "classes_target": (pupils + stage.class_size - 1) // stage.class_size,
            "classes_legal_minimum": (pupils + stage.max_class_size - 1) // stage.max_class_size,
            "age_counts": {str(age): count for age, count in sorted(item["ages"].items())},
        })
    return result, unmatched


def attendance_matrix(demand, transfers):
    transfers = tuple(transfers)
    by_key = {}
    for row in demand:
        key = (row["year"], row["region"], row["settlement"], row["stage"])
        if key in by_key:
            raise InputError("demand", "duplicate demand cell")
        by_key[key] = dict(row)
    totals = {}
    for transfer in transfers:
        fields(transfer, {"year", "source", "destination", "stage", "source_settlement", "destination_settlement", "count"}, "attendance_transfer")
        year = integer(required(transfer, "year"), "year", 1900, 2200)
        count = integer(required(transfer, "count"), "count")
        stage = name(required(transfer, "stage"), "stage")
        source = (year, name(required(transfer, "source"), "source"), transfer["source_settlement"], stage)
        destination = (year, name(required(transfer, "destination"), "destination"), transfer["destination_settlement"], stage)
        if source == destination:
            raise InputError("attendance_transfer", "identical source and destination")
        if source not in by_key or destination not in by_key:
            raise InputError("attendance_transfer", "unknown demand cell")
        totals[source] = totals.get(source, 0) + count
    for key, count in totals.items():
        if count > by_key[key]["pupils"]:
            raise InputError("attendance_transfer", "outflow exceeds resident pupils")
    for row in by_key.values():
        row["resident_pupils"] = row["pupils"]
        row["incoming_pupils"] = 0
        row["outgoing_pupils"] = 0
    for transfer in transfers:
        source = (transfer["year"], transfer["source"], transfer["source_settlement"], transfer["stage"])
        destination = (transfer["year"], transfer["destination"], transfer["destination_settlement"], transfer["stage"])
        count = transfer["count"]
        by_key[source]["outgoing_pupils"] += count
        by_key[destination]["incoming_pupils"] += count
    for row in by_key.values():
        row["pupils"] = row["resident_pupils"] + row["incoming_pupils"] - row["outgoing_pupils"]
    if sum(row["pupils"] for row in by_key.values()) != sum(row["pupils"] for row in demand):
        raise ArithmeticError("attendance transfer changed pupil total")
    return [by_key[key] for key in sorted(by_key)]
