from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from cohorts.records import InputError, decimal, fields, integer, name, required
from cohorts.transitions import Birth, ExternalAdjustment, Migration, advance
from school_pressure.capacity import assess
from school_pressure.stages import demand_by_stage


@dataclass(frozen=True)
class Scenario:
    code: str
    parent: str | None
    horizon: int
    migrations: tuple = ()
    adjustments: tuple = ()
    births: tuple = ()
    survival: dict = field(default_factory=dict)
    participation: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, record):
        allowed = {
            "code", "parent", "horizon", "migrations", "adjustments", "births",
            "survival", "participation",
        }
        fields(record, allowed, "scenario")
        parent = record.get("parent")
        if parent is not None:
            parent = name(parent, "parent")
        survival = {}
        for row in record.get("survival", []):
            fields(row, {"age", "rate"}, "survival")
            age = integer(required(row, "age"), "age", 0, 100)
            if age in survival:
                raise InputError("survival", f"duplicate age {age}")
            survival[age] = decimal(required(row, "rate"), "rate")
        participation = {}
        for row in record.get("participation", []):
            fields(row, {"region", "settlement", "stage", "rate"}, "participation")
            key = (name(required(row, "region"), "region"), required(row, "settlement"), name(required(row, "stage"), "stage"))
            if key[1] not in ("urban", "rural"):
                raise InputError("settlement", "expected urban or rural")
            if key in participation:
                raise InputError("participation", "duplicate override")
            participation[key] = decimal(required(row, "rate"), "rate")
        return cls(
            name(required(record, "code"), "code"),
            parent,
            integer(required(record, "horizon"), "horizon", 1, 30),
            tuple(Migration.parse(row) for row in record.get("migrations", [])),
            tuple(ExternalAdjustment.parse(row) for row in record.get("adjustments", [])),
            tuple(Birth.parse(row) for row in record.get("births", [])),
            survival,
            participation,
        )


def validate_scenarios(scenarios):
    by_code = {}
    for scenario in scenarios:
        if scenario.code in by_code:
            raise InputError("scenarios", f"duplicate code {scenario.code}")
        by_code[scenario.code] = scenario
    order = []
    visited = set()
    visiting = set()

    def visit(code):
        if code in visiting:
            raise InputError("scenarios", "inheritance contains a cycle")
        if code in visited:
            return
        if code not in by_code:
            raise InputError("scenarios", f"unknown parent {code}")
        visiting.add(code)
        parent = by_code[code].parent
        if parent is not None:
            visit(parent)
        visiting.remove(code)
        visited.add(code)
        order.append(code)

    for code in sorted(by_code):
        visit(code)
    if not order:
        raise InputError("scenarios", "at least one scenario required")
    return by_code, tuple(order)


def merge_sequence(parent, child, key):
    result = {key(item): item for item in parent}
    seen = set()
    for item in child:
        identity = key(item)
        if identity in seen:
            raise InputError("scenario", f"duplicate override {identity}")
        seen.add(identity)
        result[identity] = item
    return tuple(result[identity] for identity in sorted(result, key=str))


def resolve_scenarios(scenarios):
    by_code, order = validate_scenarios(scenarios)
    resolved = {}
    for code in order:
        child = by_code[code]
        if child.parent is None:
            resolved[code] = child
            continue
        parent = resolved[child.parent]
        migrations = merge_sequence(parent.migrations, child.migrations, lambda row: (row.source_cell(), row.destination_cell()))
        adjustments = merge_sequence(parent.adjustments, child.adjustments, lambda row: row.cell)
        births = merge_sequence(parent.births, child.births, lambda row: row.cell)
        resolved[code] = Scenario(
            child.code,
            child.parent,
            child.horizon,
            migrations,
            adjustments,
            births,
            {**parent.survival, **child.survival},
            {**parent.participation, **child.participation},
        )
    return resolved


def validate_scenario_domain(scenario, baseline, regions, stages):
    first = min(cell.year for cell in baseline)
    if any(cell.year != first for cell in baseline):
        raise InputError("baseline", "mixed years")
    last = first + scenario.horizon
    if last > 2200:
        raise InputError("horizon", "forecast exceeds supported calendar")
    for movement in scenario.migrations:
        if not first <= movement.year < last:
            raise InputError("migration.year", "outside scenario horizon")
        if movement.source not in regions or movement.destination not in regions:
            raise InputError("migration", "unknown source or destination region")
    for adjustment in scenario.adjustments:
        if not first <= adjustment.cell.year < last:
            raise InputError("adjustment.year", "outside scenario horizon")
        if adjustment.cell.region not in regions:
            raise InputError("adjustment.region", "unknown region")
    for birth in scenario.births:
        if not first < birth.year <= last:
            raise InputError("birth.year", "outside scenario horizon")
        if birth.region not in regions:
            raise InputError("birth.region", "unknown region")
    for region, settlement, stage in scenario.participation:
        if region not in regions or stage not in stages:
            raise InputError("participation", "unknown region or stage")
    return first, last


def run_scenario(scenario, baseline, regions, stages, capacities, *, progress=None):
    first, last = validate_scenario_domain(scenario, baseline, regions, stages)
    current = dict(baseline)
    populations = []
    accountings = []
    demands = []
    unassigned = []
    for year in range(first, last + 1):
        if progress is not None:
            progress(year, first, last)
        populations.extend({**cell.as_dict(), "count": count} for cell, count in sorted(current.items()))
        demand, unmatched = demand_by_stage(current, list(stages.values()), scenario.participation)
        demands.extend(demand)
        unassigned.append({"year": year, "population": unmatched})
        if year == last:
            break
        transition = advance(
            current,
            year=year,
            migrations=tuple(row for row in scenario.migrations if row.year == year),
            adjustments=tuple(row for row in scenario.adjustments if row.cell.year == year),
            births=tuple(row for row in scenario.births if row.year == year + 1),
            survival=scenario.survival,
        )
        current = transition.population
        accountings.append(transition.accounting)
    return {
        "scenario": scenario.code,
        "parent": scenario.parent,
        "baseline_year": first,
        "through_year": last,
        "population": populations,
        "accounting": accountings,
        "demand": demands,
        "pressure": assess(demands, capacities, stages),
        "unassigned_age_population": unassigned,
    }
