from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from cohorts.records import InputError, decimal, fields, integer, name, required


@dataclass(frozen=True)
class Capacity:
    facility: str
    region: str
    settlement: str
    stage: str
    from_year: int
    through_year: int
    seats: int
    classrooms: int
    accessible_seats: int

    @classmethod
    def parse(cls, record):
        allowed = {
            "facility", "region", "settlement", "stage", "from_year", "through_year",
            "seats", "classrooms", "accessible_seats",
        }
        fields(record, allowed, "capacity")
        first = integer(required(record, "from_year"), "from_year", 1900, 2200)
        last = integer(required(record, "through_year"), "through_year", first, 2200)
        settlement = required(record, "settlement")
        if settlement not in ("urban", "rural"):
            raise InputError("settlement", "expected urban or rural")
        seats = integer(required(record, "seats"), "seats")
        return cls(
            name(required(record, "facility"), "facility"),
            name(required(record, "region"), "region"),
            settlement,
            name(required(record, "stage"), "stage"),
            first,
            last,
            seats,
            integer(required(record, "classrooms"), "classrooms"),
            integer(record.get("accessible_seats", 0), "accessible_seats", 0, seats),
        )

    def active(self, year):
        return self.from_year <= year <= self.through_year


def validate_capacity(capacities, regions, stages):
    capacities = tuple(capacities)
    intervals = {}
    for capacity in capacities:
        if capacity.region not in regions:
            raise InputError("capacity.region", f"unknown region {capacity.region}")
        if capacity.stage not in stages:
            raise InputError("capacity.stage", f"unknown stage {capacity.stage}")
        key = capacity.facility
        previous = intervals.setdefault(key, [])
        for other in previous:
            if max(other.from_year, capacity.from_year) <= min(other.through_year, capacity.through_year):
                raise InputError("capacity", f"overlapping facility interval {key}")
        previous.append(capacity)
    return tuple(capacities)


def capacity_index(capacities, year):
    result = {}
    for capacity in capacities:
        if not capacity.active(year):
            continue
        key = (year, capacity.region, capacity.settlement, capacity.stage)
        row = result.setdefault(key, {"seats": 0, "classrooms": 0, "accessible_seats": 0, "facilities": []})
        row["seats"] += capacity.seats
        row["classrooms"] += capacity.classrooms
        row["accessible_seats"] += capacity.accessible_seats
        row["facilities"].append(capacity.facility)
    for row in result.values():
        row["facilities"].sort()
    return result


def ratio(numerator, denominator):
    if denominator == 0:
        return None
    return str((Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.000001")))


def assess(demand, capacities, stages, warning_ratio="0.9"):
    threshold = decimal(warning_ratio, "warning_ratio")
    years = sorted({row["year"] for row in demand})
    supply = {}
    for year in years:
        supply.update(capacity_index(capacities, year))
    demand_map = {}
    for row in demand:
        key = (row["year"], row["region"], row["settlement"], row["stage"])
        if key in demand_map:
            raise InputError("demand", "duplicate row")
        demand_map[key] = row
    result = []
    for key in sorted(set(demand_map) | set(supply)):
        year, region, settlement, stage_code = key
        if stage_code not in stages:
            raise InputError("stage", "unknown stage in demand or capacity")
        stage = stages[stage_code]
        pupils = demand_map.get(key, {}).get("pupils", 0)
        capacity = supply.get(key, {"seats": 0, "classrooms": 0, "accessible_seats": 0, "facilities": []})
        seats = capacity["seats"]
        classrooms = capacity["classrooms"]
        target_classes = (pupils + stage.class_size - 1) // stage.class_size
        minimum_classes = (pupils + stage.max_class_size - 1) // stage.max_class_size
        shortage = max(0, pupils - seats)
        reasons = []
        if shortage:
            reasons.append("seat_shortage")
        if minimum_classes > classrooms:
            reasons.append("classroom_legal_shortage")
        elif target_classes > classrooms:
            reasons.append("small_class_target_unmet")
        if pupils and not seats:
            status = "unserved"
        elif shortage or minimum_classes > classrooms:
            status = "critical"
        elif target_classes > classrooms or seats and Decimal(pupils) / seats >= threshold:
            status = "watch"
        else:
            status = "adequate"
        result.append({
            "year": year,
            "region": region,
            "settlement": settlement,
            "stage": stage_code,
            "pupils": pupils,
            "seats": seats,
            "classrooms": classrooms,
            "accessible_seats": capacity["accessible_seats"],
            "utilization": ratio(pupils, seats),
            "seat_shortage": shortage,
            "spare_seats": max(0, seats - pupils),
            "target_class_shortage": max(0, target_classes - classrooms),
            "legal_class_shortage": max(0, minimum_classes - classrooms),
            "status": status,
            "reasons": reasons,
            "facilities": capacity["facilities"],
        })
    return result


def compare_pressure(baseline, alternative):
    def index(rows):
        result = {}
        for row in rows:
            key = tuple(row[field] for field in ("year", "region", "settlement", "stage"))
            if key in result:
                raise InputError("comparison", "duplicate cell")
            result[key] = row
        return result
    left = index(baseline)
    right = index(alternative)
    if set(left) != set(right):
        raise InputError("comparison", "scenarios do not cover identical cells")
    result = []
    for key in sorted(left):
        before = left[key]
        after = right[key]
        result.append({
            **dict(zip(("year", "region", "settlement", "stage"), key)),
            "pupil_change": after["pupils"] - before["pupils"],
            "seat_change": after["seats"] - before["seats"],
            "shortage_change": after["seat_shortage"] - before["seat_shortage"],
            "class_shortage_change": after["legal_class_shortage"] - before["legal_class_shortage"],
            "status_before": before["status"],
            "status_after": after["status"],
        })
    return result
