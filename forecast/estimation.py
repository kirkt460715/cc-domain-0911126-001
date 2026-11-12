from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from cohorts.records import InputError, decimal, integer


def mean(values):
    if not values:
        return None
    return sum(values, Decimal(0)) / len(values)


def median(values):
    ordered = sorted(values)
    size = len(ordered)
    if not size:
        return None
    middle = size // 2
    if size % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def quantile(values, probability):
    probability = decimal(probability, "probability")
    ordered = sorted(Decimal(value) for value in values)
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    remainder = position - lower
    return ordered[lower] * (1 - remainder) + ordered[upper] * remainder


def mad(values):
    center = median(values)
    if center is None:
        return None
    return median([abs(value - center) for value in values])


@dataclass(frozen=True)
class Estimate:
    region: str
    settlement: str
    age: int
    rate: Decimal
    observations: int
    level: str
    excluded: int

    def as_dict(self):
        return {
            "region": self.region,
            "settlement": self.settlement,
            "age": self.age,
            "rate": str(self.rate),
            "observations": self.observations,
            "level": self.level,
            "excluded": self.excluded,
        }


def cohort_ratios(population, *, through_year, minimum_denominator=10, maximum_ratio="3"):
    integer(through_year, "through_year", 1901, 2200)
    integer(minimum_denominator, "minimum_denominator", 1)
    maximum_ratio = decimal(maximum_ratio, "maximum_ratio", high=Decimal(100))
    if maximum_ratio == 0:
        raise InputError("maximum_ratio", "must be positive")
    result = []
    for cell, count in sorted(population.items()):
        if cell.year >= through_year or cell.age >= 100:
            continue
        next_cell = cell.next()
        if next_cell not in population:
            continue
        following = population[next_cell]
        if count < minimum_denominator:
            result.append({"region": cell.region, "settlement": cell.settlement, "age": cell.age, "year": cell.year, "ratio": None, "reason": "small_denominator"})
            continue
        value = Decimal(following) / count
        if value > maximum_ratio:
            result.append({"region": cell.region, "settlement": cell.settlement, "age": cell.age, "year": cell.year, "ratio": None, "reason": "ratio_outside_limit"})
            continue
        result.append({"region": cell.region, "settlement": cell.settlement, "age": cell.age, "year": cell.year, "ratio": value, "reason": None})
    return result


def estimate_progression(population, targets, *, through_year, minimum_observations=2, minimum_denominator=10):
    integer(minimum_observations, "minimum_observations", 1, 100)
    ratios = cohort_ratios(population, through_year=through_year, minimum_denominator=minimum_denominator)
    exact = defaultdict(list)
    settlement_age = defaultdict(list)
    age_pool = defaultdict(list)
    exclusions = defaultdict(int)
    for row in ratios:
        exact_key = row["region"], row["settlement"], row["age"]
        if row["ratio"] is None:
            exclusions[exact_key] += 1
            continue
        exact[exact_key].append(row["ratio"])
        settlement_age[row["settlement"], row["age"]].append(row["ratio"])
        age_pool[row["age"]].append(row["ratio"])
    result = []
    seen = set()
    for region, settlement, age in targets:
        key = region, settlement, age
        if key in seen:
            raise InputError("targets", "duplicate estimation target")
        seen.add(key)
        candidates = (
            ("local", exact[key]),
            ("settlement_age", settlement_age[settlement, age]),
            ("age_pool", age_pool[age]),
        )
        selected = None
        for level, values in candidates:
            if len(values) >= minimum_observations:
                selected = Estimate(region, settlement, age, median(values), len(values), level, exclusions[key])
                break
        if selected is None:
            selected = Estimate(region, settlement, age, Decimal(1), 0, "identity_fallback", exclusions[key])
        result.append(selected)
    return tuple(result)


def estimate_births(history, *, target_year, through_year, window=3):
    integer(target_year, "target_year", through_year + 1, 2200)
    integer(window, "window", 1, 20)
    grouped = defaultdict(dict)
    for record in history:
        if record.cell.age != 0 or record.cell.year > through_year:
            continue
        key = record.cell.region, record.cell.settlement
        if record.cell.year in grouped[key]:
            raise InputError("birth_history", "duplicate year in birth series")
        grouped[key][record.cell.year] = record.count
    result = []
    for (region, settlement), values in sorted(grouped.items()):
        selected_years = [year for year in sorted(values) if through_year - window < year <= through_year]
        selected = [Decimal(values[year]) for year in selected_years]
        if not selected:
            continue
        center = median(selected)
        count = int(center.to_integral_value(rounding=ROUND_HALF_UP))
        result.append({
            "year": target_year,
            "region": region,
            "settlement": settlement,
            "count": count,
            "method": "trailing_median",
            "training_years": selected_years,
            "through_year": through_year,
            "dispersion_mad": str(mad(selected)),
        })
    return result


def residual_diagnostics(backtest_result):
    groups = defaultdict(list)
    for row in backtest_result["details"]:
        groups[row["model"], row["horizon"]].append(row)
    result = []
    for (model, horizon), rows in sorted(groups.items()):
        errors = [Decimal(row["error"]) for row in rows]
        absolute = [abs(value) for value in errors]
        ordered = sorted(rows, key=lambda row: (row["issued_year"], row["region"], row["settlement"], row["stage"]))
        signed_runs = 0
        last_sign = None
        for row in ordered:
            sign = 1 if row["error"] > 0 else -1 if row["error"] < 0 else 0
            if sign == 0:
                continue
            if last_sign != sign:
                signed_runs += 1
            last_sign = sign
        center = median(errors)
        dispersion = mad(errors)
        outliers = []
        if dispersion:
            for row in rows:
                if abs(Decimal(row["error"]) - center) > dispersion * 3:
                    outliers.append({key: row[key] for key in ("issued_year", "target_year", "region", "settlement", "stage", "error")})
        result.append({
            "model": model,
            "horizon": horizon,
            "cells": len(rows),
            "median_error": str(center),
            "median_absolute_deviation": str(dispersion),
            "p90_absolute_error": str(quantile(absolute, "0.9")),
            "p95_absolute_error": str(quantile(absolute, "0.95")),
            "positive_errors": sum(value > 0 for value in errors),
            "negative_errors": sum(value < 0 for value in errors),
            "zero_errors": sum(value == 0 for value in errors),
            "signed_runs": signed_runs,
            "outliers": outliers,
        })
    return result
