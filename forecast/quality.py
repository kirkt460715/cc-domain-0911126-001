from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal

from cohorts.records import Cell, InputError, decimal, integer


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    dimensions: dict
    message: str
    evidence: dict

    def as_dict(self):
        return {
            "code": self.code,
            "severity": self.severity,
            "dimensions": self.dimensions,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class QualityPolicy:
    require_full_age_coverage: bool = True
    minimum_population: int = 20
    migration_fraction_warning: Decimal = Decimal("0.25")
    capacity_utilization_warning: Decimal = Decimal("1.2")
    maximum_zero_capacity_cells: int = 0

    def __post_init__(self):
        if not isinstance(self.require_full_age_coverage, bool):
            raise InputError("require_full_age_coverage", "expected boolean")
        integer(self.minimum_population, "minimum_population")
        object.__setattr__(self, "migration_fraction_warning", decimal(self.migration_fraction_warning, "migration_fraction_warning"))
        object.__setattr__(self, "capacity_utilization_warning", decimal(self.capacity_utilization_warning, "capacity_utilization_warning", high=Decimal(100)))
        integer(self.maximum_zero_capacity_cells, "maximum_zero_capacity_cells")


def input_findings(batch, policy=None):
    policy = policy or QualityPolicy()
    findings = []
    baseline = min(cell.year for cell in batch.population)
    groups = defaultdict(dict)
    for cell, count in batch.population.items():
        groups[cell.region, cell.settlement][cell.age] = count
    for (region, settlement), counts in sorted(groups.items()):
        dimensions = {"region": region, "settlement": settlement, "year": baseline}
        if policy.require_full_age_coverage:
            for stage in batch.stages.values():
                expected = set(range(stage.min_age, stage.max_age + 1))
                missing = sorted(expected - set(counts))
                if missing:
                    findings.append(Finding(
                        "missing_age_cells", "warning", {**dimensions, "stage": stage.code},
                        "Missing ages differ from explicit zero population.", {"missing_ages": missing},
                    ))
        total = sum(counts.values())
        if total < policy.minimum_population:
            findings.append(Finding(
                "small_population", "warning", dimensions,
                "Population is too small for stable relative-rate interpretation.",
                {"population": total, "minimum": policy.minimum_population},
            ))
    leaves = set(batch.regions) - {region.parent for region in batch.regions.values() if region.parent is not None}
    present_regions = {cell.region for cell in batch.population}
    for region in sorted(leaves - present_regions):
        findings.append(Finding(
            "region_without_population", "warning", {"region": region, "year": baseline},
            "Registered leaf region has no population observations.", {},
        ))
    capacity_regions = {capacity.region for capacity in batch.capacities}
    for region in sorted(present_regions - capacity_regions):
        findings.append(Finding(
            "region_without_capacity", "warning", {"region": region},
            "No capacity records exist for this populated region.", {},
        ))
    for code, scenario in sorted(batch.scenarios.items()):
        outflows = Counter()
        for migration in scenario.migrations:
            if migration.year == baseline:
                outflows[migration.source_cell()] += migration.count
        for cell, count in sorted(outflows.items()):
            stock = batch.population.get(cell, 0)
            if stock == 0 or count > stock:
                findings.append(Finding(
                    "migration_exceeds_stock", "error", {"scenario": code, **cell.as_dict()},
                    "Migration outflow exceeds observed opening stock.", {"outflow": count, "stock": stock},
                ))
            elif Decimal(count) / stock > policy.migration_fraction_warning:
                findings.append(Finding(
                    "large_migration_assumption", "warning", {"scenario": code, **cell.as_dict()},
                    "Migration assumption moves a large fraction of one cohort.",
                    {"outflow": count, "stock": stock, "fraction": str(Decimal(count) / stock)},
                ))
        for capacity in batch.capacities:
            if capacity.through_year < baseline + scenario.horizon:
                findings.append(Finding(
                    "capacity_expires_in_horizon", "information", {"scenario": code, "facility": capacity.facility},
                    "Capacity becomes unavailable before the forecast ends.",
                    {"through_year": capacity.through_year, "forecast_end": baseline + scenario.horizon},
                ))
    return tuple(findings)


def output_findings(batch, result, policy=None):
    policy = policy or QualityPolicy()
    findings = []
    code = result["scenario"]
    populations = defaultdict(int)
    cells = set()
    for row in result["population"]:
        cell = Cell.parse(row)
        identity = cell.year, cell.region, cell.settlement, cell.age
        if identity in cells:
            findings.append(Finding("duplicate_output_cell", "error", {"scenario": code, **cell.as_dict()}, "Output contains duplicate population cells.", {}))
        cells.add(identity)
        populations[cell.year] += integer(row["count"], "count")
    for account in result["accounting"]:
        year = account["from_year"]
        expected = account["opening"] + account["external_arrivals"] - account["external_departures"] - account["deaths"] - account["aged_out"] + account["births"]
        if expected != account["closing"] or account["balance"] != 0:
            findings.append(Finding(
                "unbalanced_accounting", "error", {"scenario": code, "year": year},
                "Transition accounting fails conservation.", {"expected": expected, "closing": account["closing"], "balance": account["balance"]},
            ))
        if populations[year] != account["opening"] or populations[year + 1] != account["closing"]:
            findings.append(Finding(
                "population_account_disagreement", "error", {"scenario": code, "year": year},
                "Population output totals disagree with the transition account.",
                {"opening_output": populations[year], "closing_output": populations[year + 1]},
            ))
    zero_capacity = []
    for row in result["pressure"]:
        dimensions = {field: row[field] for field in ("year", "region", "settlement", "stage")}
        dimensions["scenario"] = code
        expected_gap = max(0, row["pupils"] - row["seats"])
        expected_spare = max(0, row["seats"] - row["pupils"])
        if row["seat_shortage"] != expected_gap or row["spare_seats"] != expected_spare:
            findings.append(Finding(
                "pressure_arithmetic", "error", dimensions,
                "Reported pressure does not reconcile with seats and pupils.",
                {"pupils": row["pupils"], "seats": row["seats"], "shortage": row["seat_shortage"], "spare": row["spare_seats"]},
            ))
        if row["pupils"] and not row["seats"]:
            zero_capacity.append(dimensions)
        if row["utilization"] is not None and Decimal(row["utilization"]) > policy.capacity_utilization_warning:
            findings.append(Finding(
                "high_utilization", "warning", dimensions,
                "Projected demand substantially exceeds available seats.",
                {"utilization": row["utilization"], "threshold": str(policy.capacity_utilization_warning)},
            ))
    if len(zero_capacity) > policy.maximum_zero_capacity_cells:
        findings.append(Finding(
            "zero_capacity_cells", "warning", {"scenario": code},
            "Positive projected demand has no recorded capacity.",
            {"cells": zero_capacity, "allowed": policy.maximum_zero_capacity_cells},
        ))
    return tuple(findings)


def audit(batch, results, policy=None):
    policy = policy or QualityPolicy()
    findings = list(input_findings(batch, policy))
    seen = set()
    for result in results:
        code = result["scenario"]
        if code in seen:
            raise InputError("results", "duplicate scenario result")
        seen.add(code)
        if code not in batch.scenarios:
            raise InputError("results", "unknown scenario result")
        if result.get("batch_digest") != batch.fingerprint:
            raise InputError("results", "result belongs to a different batch")
        findings.extend(output_findings(batch, result, policy))
    counts = Counter(finding.severity for finding in findings)
    return {
        "batch_digest": batch.fingerprint,
        "scenarios": sorted(seen),
        "passed": counts["error"] == 0,
        "counts": {level: counts[level] for level in ("error", "warning", "information")},
        "findings": [finding.as_dict() for finding in findings],
    }
