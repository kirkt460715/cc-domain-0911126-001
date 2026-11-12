from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from cohorts.records import InputError, integer
from school_pressure.capacity import ratio


@dataclass(frozen=True)
class ForecastPoint:
    issued_year: int
    target_year: int
    region: str
    settlement: str
    stage: str
    predicted: int
    model: str

    def __post_init__(self):
        integer(self.issued_year, "issued_year", 1900, 2199)
        integer(self.target_year, "target_year", self.issued_year + 1, 2200)
        integer(self.predicted, "predicted")

    @property
    def cell(self):
        return self.target_year, self.region, self.settlement, self.stage


def evaluate(points, observations, *, observed_through, minimum_cells=1):
    integer(observed_through, "observed_through", 1900, 2200)
    integer(minimum_cells, "minimum_cells", 1, 10**9)
    actuals = {}
    for row in observations:
        year = integer(row["year"], "year", 1900, observed_through)
        key = (year, row["region"], row["settlement"], row["stage"])
        if key in actuals:
            raise InputError("observations", "duplicate observation")
        actuals[key] = integer(row["pupils"], "pupils")
    seen = set()
    groups = {}
    details = []
    for point in points:
        key = (point.model, point.issued_year, point.cell)
        if key in seen:
            raise InputError("forecasts", "duplicate forecast point")
        seen.add(key)
        horizon = point.target_year - point.issued_year
        group_key = point.model, horizon
        group = groups.setdefault(group_key, {"matched": [], "missing": 0, "future": 0})
        if point.target_year > observed_through:
            group["future"] += 1
            continue
        if point.cell not in actuals:
            group["missing"] += 1
            continue
        actual = actuals[point.cell]
        error = point.predicted - actual
        detail = {
            "model": point.model,
            "issued_year": point.issued_year,
            "target_year": point.target_year,
            "region": point.region,
            "settlement": point.settlement,
            "stage": point.stage,
            "horizon": horizon,
            "predicted": point.predicted,
            "actual": actual,
            "error": error,
            "absolute_error": abs(error),
            "absolute_percentage_error": ratio(abs(error), actual),
        }
        group["matched"].append(detail)
        details.append(detail)
    summaries = []
    for (model, horizon), group in sorted(groups.items()):
        rows = group["matched"]
        enough = len(rows) >= minimum_cells
        absolute_error = sum(row["absolute_error"] for row in rows)
        actual_total = sum(row["actual"] for row in rows)
        signed_error = sum(row["error"] for row in rows)
        squared_error = sum(row["error"] ** 2 for row in rows)
        if enough:
            mae = ratio(absolute_error, len(rows))
            bias = ratio(signed_error, len(rows))
            rmse = str((Decimal(squared_error) / len(rows)).sqrt().quantize(Decimal("0.000001")))
            wape = ratio(absolute_error, actual_total)
        else:
            mae = None
            bias = None
            rmse = None
            wape = None
        summaries.append({
            "model": model,
            "horizon": horizon,
            "matched_cells": len(rows),
            "missing_cells": group["missing"],
            "not_yet_observed_cells": group["future"],
            "zero_actual_cells": sum(row["actual"] == 0 for row in rows),
            "sufficient": enough,
            "mae": mae,
            "bias": bias,
            "rmse": rmse,
            "wape": wape,
            "absolute_error_total": absolute_error,
            "actual_total": actual_total,
        })
    details.sort(key=lambda row: (row["model"], row["issued_year"], row["target_year"], row["region"], row["settlement"], row["stage"]))
    return {"observed_through": observed_through, "summary": summaries, "details": details}


def paired_comparison(backtest_result, left_model, right_model):
    if left_model == right_model:
        raise InputError("models", "paired comparison requires distinct models")
    indexed = {}
    for row in backtest_result["details"]:
        if row["model"] not in (left_model, right_model):
            continue
        key = tuple(row[field] for field in ("issued_year", "target_year", "region", "settlement", "stage"))
        indexed.setdefault(key, {})[row["model"]] = row
    deltas = []
    missing_left = 0
    missing_right = 0
    for key, pair in sorted(indexed.items()):
        if left_model not in pair:
            missing_left += 1
            continue
        if right_model not in pair:
            missing_right += 1
            continue
        left = pair[left_model]
        right = pair[right_model]
        if left["actual"] != right["actual"]:
            raise InputError("comparison", "paired actual values differ")
        delta = right["absolute_error"] - left["absolute_error"]
        deltas.append({
            **dict(zip(("issued_year", "target_year", "region", "settlement", "stage"), key)),
            "left_absolute_error": left["absolute_error"],
            "right_absolute_error": right["absolute_error"],
            "difference": delta,
        })
    count = len(deltas)
    return {
        "left": left_model,
        "right": right_model,
        "paired_cells": count,
        "unpaired_left_missing": missing_left,
        "unpaired_right_missing": missing_right,
        "right_better": sum(row["difference"] < 0 for row in deltas),
        "left_better": sum(row["difference"] > 0 for row in deltas),
        "ties": sum(row["difference"] == 0 for row in deltas),
        "mean_absolute_error_difference": ratio(sum(row["difference"] for row in deltas), count),
        "details": deltas,
    }


def rolling_origins(population_records, *, first_origin, last_origin, horizon):
    integer(first_origin, "first_origin", 1900, 2199)
    integer(last_origin, "last_origin", first_origin, 2199)
    integer(horizon, "horizon", 1, 30)
    by_year = {}
    for record in population_records:
        by_year.setdefault(record.cell.year, []).append(record)
    result = []
    for origin in range(first_origin, last_origin + 1):
        if origin not in by_year:
            raise InputError("rolling_origins", f"missing origin year {origin}")
        targets = list(range(origin + 1, min(2200, origin + horizon) + 1))
        training = [record for year in sorted(by_year) if year <= origin for record in by_year[year]]
        if any(record.cell.year > origin for record in training):
            raise ArithmeticError("future records leaked into training")
        result.append({"origin": origin, "baseline": tuple(by_year[origin]), "training": tuple(training), "target_years": tuple(targets)})
    return tuple(result)
