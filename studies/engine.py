"""联合预测研究引擎。

方法约束（与规划组的研究约定一一对应）：

* 历史点预测与实际人数按（发布年份、目标年份、叶子地区、城乡、学段）登记，
  实际值带首次可知年份；某个预测起点只能使用当时已经可知且目标已到期的误差。
* 按预测跨度构造覆盖整个叶子单元空间的完整联合残差向量；缺地区的历史向量
  逐条列出原因，绝不补零后混入重采样。
* 重采样以发布年份为整体抽取同一向量，保留同一冲击在各地区间的关联；
  训练、区间校准、最终回测所用发布年份彼此分开。
* 每次模拟先形成非负整数的叶子需求，再由同一次模拟聚合到祖先地区，
  并调用 school_pressure.capacity.assess 的既有容量口径计算超载。
* 分位数不保证可加；联合样本与每级人数必须相符（逐次模拟核对）。
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
import hashlib
import random

from cohorts.records import InputError, Region, fields, integer, name, region_tree, required
from forecast.estimation import median as decimal_median
from forecast.estimation import quantile as decimal_quantile
from releases.ledger import canonical
from school_pressure.aggregation import aggregate
from school_pressure.capacity import Capacity, assess, ratio, validate_capacity
from school_pressure.stages import Stage, validate_stages

DEPENDENCE_MODES = ("as_observed", "aligned", "opposed")
INSUFFICIENT_POLICIES = ("stop", "merge_horizons")
MAX_DRAWS = 100000


def _probability(value, field):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise InputError(field, "expected decimal probability", value)
    try:
        result = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive
        raise InputError(field, "invalid probability", value) from exc
    if not result.is_finite() or not Decimal(0) < result < Decimal(1):
        raise InputError(field, "probability must lie strictly inside 0..1", value)
    return result


def _year_list(value, field):
    if not isinstance(value, list) or not value or any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise InputError(field, "expected a non-empty array of integer years")
    years = [integer(v, field, 1900, 2200) for v in value]
    if len(set(years)) != len(years):
        raise InputError(field, "duplicate year in explicit window")
    return years


def _horizon_list(value, field):
    if not isinstance(value, list) or not value or any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise InputError(field, "expected a non-empty array of integer horizons")
    horizons = [integer(v, field, 1, 30) for v in value]
    if len(set(horizons)) != len(horizons):
        raise InputError(field, "duplicate horizon in merge group")
    return horizons


class StudyInputs:
    """校验并索引一次研究的全部输入。"""

    def __init__(self, payload):
        fields(payload, {
            "study", "regions", "stages", "capacities", "forecasts", "observations",
        }, "study_input")
        self.study = self._parse_study(required(payload, "study"))
        self.regions = region_tree(Region.parse(row) for row in self._collection(payload, "regions", 10000))
        self.leaf_codes = frozenset(
            code for code in self.regions
            if code not in {region.parent for region in self.regions.values() if region.parent is not None}
        )
        self.root_codes = frozenset(
            code for code in self.regions if self.regions[code].parent is None
        )
        if len(self.root_codes) != 1:
            raise InputError("regions", "joint study expects exactly one county root")
        self.root = next(iter(self.root_codes))
        self.stages = validate_stages(Stage.parse(row) for row in self._collection(payload, "stages", 30))
        self.capacities = tuple(Capacity.parse(row) for row in self._collection(payload, "capacities", 100000, allow_empty=True))
        validate_capacity(self.capacities, self.regions, self.stages)
        if any(capacity.region not in self.leaf_codes for capacity in self.capacities):
            raise InputError("capacities", "capacity may be supplied only for leaf regions")
        self.forecasts = self._parse_forecasts(self._collection(payload, "forecasts", 1000000))
        self.observations = self._parse_observations(self._collection(payload, "observations", 1000000))
        self.cell_universe = tuple(sorted({
            (row["region"], row["settlement"], row["stage"]) for row in self.forecasts
        }))
        if not self.cell_universe:
            raise InputError("forecasts", "at least one forecast cell required")
        for region, settlement, stage in self.cell_universe:
            if region not in self.leaf_codes:
                raise InputError("forecasts", f"region {region} is not a leaf region")
            if settlement not in ("urban", "rural"):
                raise InputError("forecasts", "settlement must be urban or rural")
            if stage not in self.stages:
                raise InputError("forecasts", f"unknown stage {stage}")
        for key in self.observations:
            if (key[1], key[2], key[3]) not in set(self.cell_universe):
                raise InputError("observations", "observation cell has no forecast history")
        self.horizons = tuple(sorted({row["horizon"] for row in self.forecasts}))
        self._index_forecasts()
        self._validate_centerline()

    @staticmethod
    def _collection(payload, key, maximum, *, allow_empty=False):
        value = required(payload, key)
        if not isinstance(value, list):
            raise InputError(key, "expected array")
        if not allow_empty and not value:
            raise InputError(key, "at least one record required")
        if len(value) > maximum:
            raise InputError(key, f"too many records; maximum {maximum}")
        return value

    def _parse_study(self, record):
        fields(record, {
            "study_id", "as_of_year", "model", "draws", "seed", "coverage_levels",
            "dependence_modes", "minimum_complete_vectors", "insufficient_history",
            "merge_horizons", "train_years", "calibration_years", "backtest_years",
            "target_years",
        }, "study")
        modes = record.get("dependence_modes", list(DEPENDENCE_MODES))
        if not isinstance(modes, list) or not modes or any(mode not in DEPENDENCE_MODES for mode in modes):
            raise InputError("dependence_modes", f"expected subset of {DEPENDENCE_MODES}")
        if len(set(modes)) != len(modes):
            raise InputError("dependence_modes", "duplicate mode")
        policy = record.get("insufficient_history", "stop")
        if policy not in INSUFFICIENT_POLICIES:
            raise InputError("insufficient_history", f"expected one of {INSUFFICIENT_POLICIES}")
        levels = [_probability(value, "coverage_levels") for value in record.get("coverage_levels", ("0.5", "0.8", "0.9"))]
        if not levels or len(set(levels)) != len(levels):
            raise InputError("coverage_levels", "expected distinct probabilities in 0..1")
        merge_groups = []
        for group in record.get("merge_horizons", ()):
            merge_groups.append(tuple(_horizon_list(group, "merge_horizons")))
        explicit = {key: _year_list(record[key], key) for key in ("train_years", "calibration_years", "backtest_years") if key in record}
        target_years = None
        if "target_years" in record:
            target_years = _year_list(record["target_years"], "target_years")
        config = {
            "study_id": name(required(record, "study_id"), "study_id"),
            "as_of_year": integer(required(record, "as_of_year"), "as_of_year", 1900, 2199),
            "model": name(required(record, "model"), "model"),
            "draws": integer(record.get("draws", 1000), "draws", 1, MAX_DRAWS),
            "seed": integer(record.get("seed", 0), "seed", 0, 10**12),
            "coverage_levels": tuple(sorted(levels)),
            "dependence_modes": tuple(modes),
            "minimum_complete_vectors": integer(record.get("minimum_complete_vectors", 3), "minimum_complete_vectors", 1, 10000),
            "insufficient_history": policy,
            "merge_horizons": tuple(merge_groups),
            "target_years": target_years,
            **explicit,
        }
        return config

    def _parse_forecasts(self, value):
        rows = []
        seen = set()
        models = set()
        for row in value:
            fields(row, {"issued_year", "target_year", "region", "settlement", "stage", "predicted", "model"}, "forecasts")
            issued = integer(required(row, "issued_year"), "issued_year", 1900, 2200)
            target = integer(required(row, "target_year"), "target_year", issued + 1, 2200)
            region = name(required(row, "region"), "region")
            settlement = required(row, "settlement")
            if settlement not in ("urban", "rural"):
                raise InputError("settlement", "expected urban or rural")
            stage = name(required(row, "stage"), "stage")
            predicted = integer(required(row, "predicted"), "predicted")
            model = name(required(row, "model"), "model")
            key = (model, issued, target, region, settlement, stage)
            if key in seen:
                raise InputError("forecasts", f"duplicate forecast {key}")
            seen.add(key)
            models.add(model)
            rows.append({
                "issued_year": issued, "target_year": target, "region": region,
                "settlement": settlement, "stage": stage, "predicted": predicted,
                "model": model, "horizon": target - issued,
            })
        if self.study["model"] not in models:
            raise InputError("model", f"model {self.study['model']} has no forecast records")
        return tuple(rows)

    def _parse_observations(self, value):
        result = {}
        for row in value:
            fields(row, {"year", "region", "settlement", "stage", "pupils", "first_known_year"}, "observations")
            year = integer(required(row, "year"), "year", 1900, 2200)
            region = name(required(row, "region"), "region")
            settlement = required(row, "settlement")
            if settlement not in ("urban", "rural"):
                raise InputError("settlement", "expected urban or rural")
            stage = name(required(row, "stage"), "stage")
            pupils = integer(required(row, "pupils"), "pupils")
            first_known = integer(required(row, "first_known_year"), "first_known_year", year, 2200)
            key = (year, region, settlement, stage)
            if key in result:
                raise InputError("observations", f"duplicate observation {key}")
            result[key] = {"pupils": pupils, "first_known_year": first_known}
        return result

    def _index_forecasts(self):
        self.by_origin_horizon = defaultdict(dict)
        for row in self.forecasts:
            if row["model"] != self.study["model"]:
                continue
            self.by_origin_horizon[(row["issued_year"], row["horizon"])][
                (row["region"], row["settlement"], row["stage"])
            ] = row["predicted"]

    def _validate_centerline(self):
        """未来中心线：as_of 起点对每个跨度、每个叶子单元都必须有点预测。"""
        as_of = self.study["as_of_year"]
        center = {}
        for horizon in self.horizons:
            preds = self.by_origin_horizon.get((as_of, horizon))
            if not preds:
                raise InputError("forecasts", f"no centerline forecast issued in {as_of} for horizon {horizon}")
            missing = sorted(set(self.cell_universe) - set(preds))
            if missing:
                raise InputError("forecasts", f"centerline for horizon {horizon} misses leaf cells {missing}")
            center[horizon] = dict(preds)
        self.centerline = center

    # ---- 残差向量 -------------------------------------------------------

    def build_vectors(self, horizon):
        """返回某跨度的完整向量列表与被排除起点的逐条说明。"""
        complete = []
        excluded = []
        as_of = self.study["as_of_year"]
        origins = sorted({row["issued_year"] for row in self.forecasts if row["horizon"] == horizon and row["model"] == self.study["model"]})
        for origin in origins:
            target = origin + horizon
            preds = self.by_origin_horizon.get((origin, horizon), {})
            reasons = []
            if target > as_of:
                reasons.append("target_not_matured")
            missing_forecast = []
            missing_observation = []
            not_knowable = []
            vector = {}
            knowable_at_origin = True
            max_first_known = None
            for cell in self.cell_universe:
                region, settlement, stage = cell
                if cell not in preds:
                    missing_forecast.append(list(cell))
                    continue
                observation = self.observations.get((target, region, settlement, stage))
                if observation is None:
                    missing_observation.append(list(cell))
                    continue
                if observation["first_known_year"] > as_of:
                    not_knowable.append({"cell": list(cell), "first_known_year": observation["first_known_year"]})
                    continue
                shock = observation["pupils"] - preds[cell]
                vector[cell] = {
                    "predicted": preds[cell],
                    "actual": observation["pupils"],
                    "shock": shock,
                    "first_known_year": observation["first_known_year"],
                }
                if observation["first_known_year"] > target:
                    knowable_at_origin = False
                max_first_known = observation["first_known_year"] if max_first_known is None else max(max_first_known, observation["first_known_year"])
            if missing_forecast:
                reasons.append("missing_forecast_cells")
            if missing_observation:
                reasons.append("missing_observation_cells")
            if not_knowable:
                reasons.append("observation_not_yet_knowable")
            entry = {
                "horizon": horizon,
                "issued_year": origin,
                "target_year": target,
                "reasons": reasons,
                "missing_forecast_cells": missing_forecast,
                "missing_observation_cells": missing_observation,
                "not_yet_knowable_cells": not_knowable,
                "all_cells_knowable_at_origin": knowable_at_origin and not reasons,
                "first_known_year": max_first_known,
            }
            if reasons:
                excluded.append(entry)
                continue
            if len(vector) != len(self.cell_universe):  # defensive, completeness already implied
                raise ArithmeticError("incomplete vector passed completeness check")
            complete.append({**entry, "vector": vector})
        return tuple(complete), tuple(excluded)


def _default_split(years):
    """训练/校准/回测三段分离；年份不足时显式说明无法三分。"""
    ordered = sorted(years)
    size = len(ordered)
    if size < 3:
        return None, "insufficient_years_for_three_way_split"
    holdout = max(1, size // 5)
    backtest = ordered[-holdout:]
    calibration = ordered[-2 * holdout:-holdout]
    training = ordered[:-2 * holdout]
    if not training or not calibration or not backtest:
        return None, "insufficient_years_for_three_way_split"
    return {"training": training, "calibration": calibration, "backtest": backtest}, None


def _derive_seed(study_id, effective_key, base_seed):
    material = canonical({"study_id": study_id, "horizon_group": effective_key, "seed": base_seed}).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _opposed_factor(region, ordered_leaves):
    return 1 if ordered_leaves.index(region) % 2 == 0 else -1


def _transform_shocks(shocks, cells, mode, ordered_leaves):
    if mode == "as_observed":
        return list(shocks)
    if mode == "aligned":
        direction = 1 if sum(shocks) >= 0 else -1
        return [direction * abs(value) for value in shocks]
    if mode == "opposed":
        return [value * _opposed_factor(cells[index][0], ordered_leaves) for index, value in enumerate(shocks)]
    raise InputError("dependence_modes", f"unknown mode {mode}")


def _sample_quantile(values, probability):
    if not values:
        return None
    return decimal_quantile([Decimal(v) for v in values], str(probability))


def _stats_from_samples(pupils, shortages, seats, levels):
    ordered_levels = sorted(levels)
    intervals = {}
    for level in ordered_levels:
        tail = (1 - level) / 2
        lower = _sample_quantile(pupils, tail)
        upper = _sample_quantile(pupils, 1 - tail)
        intervals[str(level)] = {
            "lower": str(lower),
            "upper": str(upper),
            "width": str(upper - lower),
        }
    exceeding = [value for value in shortages if value > 0]
    conditional = None
    if exceeding:
        conditional = {
            "draws": len(exceeding),
            "mean_shortage": str((sum(Decimal(v) for v in exceeding) / len(exceeding)).quantize(Decimal("0.000001"))),
            "median_shortage": str(decimal_median(exceeding)),
            "shortage_quantiles": {
                str(level): str(_sample_quantile(exceeding, level)) for level in ordered_levels
            },
        }
    return {
        "seats": seats,
        "draws": len(pupils),
        "mean_pupils": str((sum(Decimal(v) for v in pupils) / len(pupils)).quantize(Decimal("0.000001"))),
        "median_pupils": str(decimal_median(pupils)),
        "intervals": intervals,
        "exceeding_draws": len(exceeding),
        "exceedance_frequency": ratio(len(exceeding), len(pupils)),
        "mean_shortage": str((sum(Decimal(v) for v in shortages) / len(shortages)).quantize(Decimal("0.000001"))),
        "conditional_overload": conditional,
    }


def _interval_score(lower, upper, actual, level):
    width = upper - lower
    alpha = Decimal(1) - level
    penalty_low = Decimal(2) * max(Decimal(0), lower - actual) / alpha
    penalty_high = Decimal(2) * max(Decimal(0), actual - upper) / alpha
    return width + penalty_low + penalty_high


def _window_diagnostics(inputs, window_vectors, training_vectors, levels, *, delta_pool=None):
    """在未参与拟合的年份上评估原始区间与校准区间。

    delta_pool 为校准窗口向量；为 None 时只评估原始经验区间。
    训练池随回测起点滚动，只取更早发布、且其目标实际在被评估目标年
    之前已经可知的向量，与研究的可知性规则保持一致；全县按学段汇总点
    仅在该学段所有叶子单元在该目标年均可知时才计分。
    """
    cells = inputs.cell_universe
    stages = sorted(inputs.stages)
    levels = sorted(levels)
    delta_by_cell = {}
    delta_county = {}
    if delta_pool is not None:
        # 校准窗口内的向量在研究执行时（as_of）都已经可知；
        # 分位差在此窗口拟合，并在从未参与拟合的回测窗口上检验。
        for cell in cells:
            absolutes = [abs(item["vector"][cell]["shock"]) for item in delta_pool]
            if absolutes:
                delta_by_cell[cell] = {
                    str(level): str(_conformal_quantile(absolutes, level)) for level in levels
                }
        county_absolutes = [abs(sum(item["vector"][cell]["shock"] for cell in cells)) for item in delta_pool]
        if county_absolutes:
            delta_county = {
                str(level): str(_conformal_quantile(county_absolutes, level)) for level in levels
            }

    def fresh_buckets():
        return {
            str(level): {
                "raw": {"points": 0, "inside": 0, "width_total": 0, "score_total": Decimal(0), "excluded_knowability": 0},
                "calibrated": {"points": 0, "inside": 0, "width_total": 0, "score_total": Decimal(0), "excluded_knowability": 0},
            }
            for level in levels
        }

    per_level = fresh_buckets()
    county_levels = fresh_buckets()
    origins = sorted({item["issued_year"] for item in window_vectors})
    for origin in origins:
        item = next(entry for entry in window_vectors if entry["issued_year"] == origin)
        target_year = item["target_year"]
        # 回测该起点时，只能用更早发布、且其目标实际已在本目标年之前可知的历史误差
        rolling_pool = [
            entry for entry in training_vectors
            if entry["issued_year"] < origin and entry["first_known_year"] <= target_year
        ]
        stage_state = {
            stage: {"actual": 0, "center": 0, "cells": 0, "knowable": 0, "raw_shocks": defaultdict(list)}
            for stage in stages
        }
        for cell in cells:
            region, settlement, stage = cell
            detail = item["vector"][cell]
            state = stage_state[stage]
            state["cells"] += 1
            state["actual"] += detail["actual"]
            state["center"] += detail["predicted"]
            for pool_item in rolling_pool:
                state["raw_shocks"][cell].append(pool_item["vector"][cell]["shock"])
            if detail["first_known_year"] > target_year:
                for bucket in per_level.values():
                    bucket["raw"]["excluded_knowability"] += 1
                    bucket["calibrated"]["excluded_knowability"] += 1
                continue
            state["knowable"] += 1
            actual = detail["actual"]
            center = detail["predicted"]
            raw_shocks = [entry["vector"][cell]["shock"] for entry in rolling_pool]
            for level in levels:
                bucket = per_level[str(level)]
                if raw_shocks:
                    lower = max(0, int(center + _sample_quantile(raw_shocks, (1 - level) / 2)))
                    upper = max(0, int(center + _sample_quantile(raw_shocks, 1 - (1 - level) / 2)))
                    _record_bucket(bucket["raw"], lower, upper, actual, level)
                if cell in delta_by_cell:
                    delta = Decimal(delta_by_cell[cell][str(level)])
                    lower = max(0, int((Decimal(center) - delta).to_integral_value(rounding=ROUND_HALF_UP)))
                    upper = max(0, int((Decimal(center) + delta).to_integral_value(rounding=ROUND_HALF_UP)))
                    _record_bucket(bucket["calibrated"], lower, upper, actual, level)
        for stage in stages:
            state = stage_state[stage]
            if state["knowable"] != state["cells"]:
                for bucket in county_levels.values():
                    bucket["raw"]["excluded_knowability"] += 1
                    bucket["calibrated"]["excluded_knowability"] += 1
                continue
            county_shocks = [
                sum(pool_item["vector"][cell]["shock"] for cell in cells if cell[2] == stage)
                for pool_item in rolling_pool
            ]
            for level in levels:
                bucket = county_levels[str(level)]
                if county_shocks:
                    lower = max(0, int(state["center"] + _sample_quantile(county_shocks, (1 - level) / 2)))
                    upper = max(0, int(state["center"] + _sample_quantile(county_shocks, 1 - (1 - level) / 2)))
                    _record_bucket(bucket["raw"], lower, upper, state["actual"], level)
                if delta_county:
                    delta = Decimal(delta_county[str(level)])
                    lower = max(0, int((Decimal(state["center"]) - delta).to_integral_value(rounding=ROUND_HALF_UP)))
                    upper = max(0, int((Decimal(state["center"]) + delta).to_integral_value(rounding=ROUND_HALF_UP)))
                    _record_bucket(bucket["calibrated"], lower, upper, state["actual"], level)
    return {
        "origins": origins,
        "leaf_cells": _summarize_buckets(per_level),
        "county_by_stage": _summarize_buckets(county_levels),
    }


def _record_bucket(bucket, lower, upper, actual, level):
    bucket["points"] += 1
    bucket["inside"] += int(lower <= actual <= upper)
    bucket["width_total"] += upper - lower
    bucket["score_total"] += _interval_score(Decimal(lower), Decimal(upper), Decimal(actual), level)


def _summarize_buckets(per_level):
    result = {}
    for level, kinds in sorted(per_level.items()):
        result[level] = {}
        for kind, bucket in kinds.items():
            points = bucket["points"]
            result[level][kind] = None
            if points:
                result[level][kind] = {
                    "points": points,
                    "coverage": ratio(bucket["inside"], points),
                    "mean_width": str((Decimal(bucket["width_total"]) / points).quantize(Decimal("0.000001"))),
                    "mean_interval_score": str((bucket["score_total"] / points).quantize(Decimal("0.000001"))),
                    "excluded_knowability": bucket["excluded_knowability"],
                }
    return result


def _conformal_quantile(absolutes, level):
    """有限样本分位校正：ceil((n+1)*p)/n，截断在 1。"""
    size = len(absolutes)
    rank = min(size, int((Decimal(size + 1) * level).to_integral_value(rounding=ROUND_CEILING)))
    probability = Decimal(rank) / size
    return decimal_quantile([Decimal(v) for v in absolutes], probability)


def _simulate_horizon(inputs, effective, levels):
    config = inputs.study
    cells = inputs.cell_universe
    ordered_leaves = sorted(inputs.leaf_codes)
    target_year = effective["target_year"]
    training_vectors = effective["vectors_by_window"]["training"]
    if not training_vectors:
        effective["available"] = False
        effective["reason"] = "no_training_vectors"
        return
    rng = random.Random(_derive_seed(config["study_id"], effective["key"], config["seed"]))
    pool_size = len(training_vectors)
    indices = [rng.randrange(pool_size) for _ in range(config["draws"])]
    effective["draw_indices"] = indices
    effective["pool_size"] = pool_size
    modes = {mode: {"reconciliation_mismatches": 0, "clamp_events": 0, "clamped_cells": defaultdict(int)} for mode in config["dependence_modes"]}
    accum = {
        mode: defaultdict(lambda: {"pupils": [], "shortages": [], "seats": None})
        for mode in config["dependence_modes"]
    }
    leaf_demand_matrices = {mode: [] for mode in config["dependence_modes"]}
    for draw, index in enumerate(indices):
        picked = training_vectors[index]
        shocks_raw = [picked["vector"][cell]["shock"] for cell in cells]
        for mode in config["dependence_modes"]:
            shocks = _transform_shocks(shocks_raw, cells, mode, ordered_leaves)
            demand_rows = []
            leaf_demands = []
            for cell, shock in zip(cells, shocks):
                region, settlement, stage = cell
                center = effective["centerline"][cell]
                pupils = center + shock
                if pupils < 0:
                    modes[mode]["clamp_events"] += 1
                    modes[mode]["clamped_cells"][cell] += 1
                    pupils = 0
                leaf_demands.append(pupils)
                demand_rows.append({
                    "year": target_year, "region": region, "settlement": settlement,
                    "stage": stage, "pupils": pupils,
                })
            leaf_demand_matrices[mode].append(leaf_demands)
            pressure = assess(demand_rows, inputs.capacities, inputs.stages)
            universe_keys = {(target_year, *cell) for cell in cells}
            leaf_pressure = [row for row in pressure if (row["year"], row["region"], row["settlement"], row["stage"]) in universe_keys]
            rolled = aggregate(leaf_pressure, inputs.regions)
            rolled_all = aggregate(leaf_pressure, inputs.regions, group_settlement=False)
            # rolled 按城乡分组且会重发叶子明细：只保留祖先地区，避免叶子行重复；
            # rolled_all 不区分城乡（settlement="all"），含叶子与祖先的学段汇总，
            # 与叶子明细的键不同，不会重复。
            ancestor_rows = [row for row in rolled if row["region"] not in inputs.leaf_codes]
            settlement_all_rows = rolled_all
            # 联合样本逐次核对：祖先人数必须等于叶子之和
            leaf_totals = defaultdict(int)
            for row in leaf_pressure:
                leaf_totals[row["stage"]] += row["pupils"]
            for row in settlement_all_rows:
                if row["region"] == inputs.root and row["settlement"] == "all":
                    if row["pupils"] != leaf_totals[row["stage"]]:
                        modes[mode]["reconciliation_mismatches"] += 1
            for row in ancestor_rows + settlement_all_rows:
                key = (row["region"], row["settlement"], row["stage"])
                bucket = accum[mode][key]
                bucket["pupils"].append(row["pupils"])
                bucket["shortages"].append(row["seat_shortage"])
                bucket["seats"] = row["seats"]
            for row in leaf_pressure:
                key = (row["region"], row["settlement"], row["stage"])
                bucket = accum[mode][key]
                bucket["pupils"].append(row["pupils"])
                bucket["shortages"].append(row["seat_shortage"])
                bucket["seats"] = row["seats"]
    effective["modes"] = {}
    for mode in config["dependence_modes"]:
        rows = []
        for key in sorted(accum[mode]):
            region, settlement, stage = key
            bucket = accum[mode][key]
            stats = _stats_from_samples(bucket["pupils"], bucket["shortages"], bucket["seats"], levels)
            rows.append({
                "year": target_year, "region": region, "settlement": settlement,
                "stage": stage, **stats,
            })
        # 全县条件超载时的叶子集中度
        concentration = _leaf_concentration(inputs, target_year, accum[mode], rows, levels)
        for row in rows:
            if row["region"] == inputs.root and row["settlement"] == "all":
                row["leaf_concentration"] = concentration.get(row["stage"], [])
        effective["modes"][mode] = {
            "rows": rows,
            "clamp_events": modes[mode]["clamp_events"],
            "clamped_cells": {",".join(cell): count for cell, count in sorted(modes[mode]["clamped_cells"].items())},
            "reconciliation": {
                "draws_checked": config["draws"],
                "mismatches": modes[mode]["reconciliation_mismatches"],
            },
            "sample_digest": hashlib.sha256(
                canonical(leaf_demand_matrices[mode]).encode("utf-8")
            ).hexdigest(),
        }


def _leaf_concentration(inputs, target_year, buckets, rows, levels):
    by_stage = defaultdict(list)
    for region in sorted(inputs.leaf_codes):
        for stage in sorted(inputs.stages):
            key = (region, "all", stage)
            if key not in buckets:
                continue
            bucket = buckets[key]
            shortages = bucket["shortages"]
            exceeding = [value for value in shortages if value > 0]
            by_stage[stage].append({
                "region": region,
                "stage": stage,
                "exceedance_frequency": ratio(len(exceeding), len(shortages)),
                "mean_shortage": str((sum(Decimal(v) for v in shortages) / len(shortages)).quantize(Decimal("0.000001"))),
                "conditional_mean_shortage": str((sum(Decimal(v) for v in exceeding) / len(exceeding)).quantize(Decimal("0.000001"))) if exceeding else None,
            })
    result = {}
    county_rows = {(row["region"], row["settlement"], row["stage"]): row for row in rows}
    for stage, entries in by_stage.items():
        county = county_rows.get((inputs.root, "all", stage))
        if county is None or county["exceeding_draws"] == 0:
            for entry in entries:
                entry["mean_share_of_county_shortage"] = None
            result[stage] = entries
            continue
        county_conditional_mean = Decimal(county["conditional_overload"]["mean_shortage"])
        for entry in entries:
            share = None
            if entry["conditional_mean_shortage"] is not None and county_conditional_mean > 0:
                share = str((Decimal(entry["conditional_mean_shortage"]) / county_conditional_mean).quantize(Decimal("0.000001")))
            entry["mean_share_of_county_shortage"] = share
        result[stage] = sorted(entries, key=lambda entry: (entry["mean_share_of_county_shortage"] is None, -(Decimal(entry["mean_share_of_county_shortage"]) if entry["mean_share_of_county_shortage"] else 0), entry["region"]))
    return result


def _window_partition(inputs, horizon_vectors, explicit_windows):
    years = [item["issued_year"] for item in horizon_vectors]
    if explicit_windows:
        for key in ("train_years", "calibration_years", "backtest_years"):
            if key in explicit_windows:
                unknown = sorted(set(explicit_windows[key]) - set(years))
                if unknown:
                    raise InputError(key, f"years without complete vectors for this horizon group: {unknown}")
        windows = {
            "training": list(explicit_windows.get("train_years", [])),
            "calibration": list(explicit_windows.get("calibration_years", [])),
            "backtest": list(explicit_windows.get("backtest_years", [])),
        }
        assigned = [year for group in windows.values() for year in group]
        if len(set(assigned)) != len(assigned):
            raise InputError("study", "training, calibration and backtest windows must be disjoint")
        unassigned = sorted(set(years) - set(assigned))
        if unassigned:
            raise InputError("study", f"explicit windows leave complete-vector years unassigned: {unassigned}")
        if not windows["training"] or not windows["calibration"] or not windows["backtest"]:
            raise InputError("study", "each explicit window must contain at least one year")
        return windows, None
    split, reason = _default_split(years)
    if split is None:
        return None, reason
    return split, None


def _build_effective_horizons(inputs, history):
    config = inputs.study
    effective = []
    explicit_groups = [set(group) for group in config["merge_horizons"]]
    handled = set()
    for group in explicit_groups:
        unknown = sorted(group - set(inputs.horizons))
        if unknown:
            raise InputError("merge_horizons", f"unknown horizons {unknown}")
        vectors = []
        provenance = []
        excluded = []
        for horizon in sorted(group):
            record = history[horizon]
            vectors.extend(record["complete"])
            provenance.extend(
                {"horizon": horizon, "issued_year": item["issued_year"], "target_year": item["target_year"], "first_known_year": item["first_known_year"]}
                for item in record["complete"]
            )
            excluded.extend(record["excluded"])
        handled.update(group)
        effective.append(_effective_entry(inputs, sorted(group), vectors, provenance, excluded, merged=True,
                                          merge_note="explicit merge_horizons group"))
    insufficient = [
        horizon for horizon in inputs.horizons
        if horizon not in handled and len(history[horizon]["complete"]) < config["minimum_complete_vectors"]
    ]
    if insufficient and config["insufficient_history"] == "merge_horizons":
        # 仅把历史不足的跨度并入同一池；所有跨度共享同一叶子单元空间，
        # 因此按发布年份整体抽取的向量在合并后仍保持联合完整。
        group = sorted(insufficient)
        vectors = []
        provenance = []
        excluded = []
        for horizon in group:
            record = history[horizon]
            vectors.extend(record["complete"])
            provenance.extend(
                {"horizon": horizon, "issued_year": item["issued_year"], "target_year": item["target_year"], "first_known_year": item["first_known_year"]}
                for item in record["complete"]
            )
            excluded.extend(record["excluded"])
        note = (
            "auto-pooled because each listed horizon had fewer than minimum_complete_vectors complete origins; "
            "the horizons share the identical leaf cell universe, so whole-origin vectors stay jointly complete"
        )
        effective.append(_effective_entry(inputs, group, vectors, provenance, excluded, merged=True, merge_note=note))
        handled.update(group)
    for horizon in inputs.horizons:
        if horizon in handled:
            continue
        record = history[horizon]
        if len(record["complete"]) < config["minimum_complete_vectors"]:
            effective.append(_stopped_entry(inputs, horizon, record, "below_minimum_complete_vectors"))
            handled.add(horizon)
            continue
        effective.append(_effective_entry(
            inputs, [horizon], record["complete"],
            [{"horizon": horizon, "issued_year": item["issued_year"], "target_year": item["target_year"], "first_known_year": item["first_known_year"]} for item in record["complete"]],
            record["excluded"], merged=False, merge_note=None,
        ))
    # 合并后的去重保护：同一 (horizon, issued_year) 向量只能出现一次
    seen_identities = set()
    for entry in effective:
        for provenance in entry["vector_provenance"]:
            identity = (provenance["horizon"], provenance["issued_year"])
            if identity in seen_identities:
                raise InputError("merge_horizons", f"vector {identity} assigned to multiple effective groups")
            seen_identities.add(identity)
    effective.sort(key=lambda entry: entry["key"])
    return effective


def _effective_entry(inputs, horizons, vectors, provenance, excluded, *, merged, merge_note):
    target_map = {horizon: inputs.study["as_of_year"] + horizon for horizon in inputs.horizons}
    if inputs.study["target_years"] is not None:
        if len(inputs.study["target_years"]) != len(inputs.horizons):
            raise InputError("target_years", "one target year per historical horizon is required")
        target_map = dict(zip(sorted(inputs.horizons), inputs.study["target_years"]))
    primary = horizons[0]
    return {
        "key": "+".join(str(h) for h in horizons),
        "horizons": list(horizons),
        "merged": merged,
        "merge_note": merge_note,
        "target_year": target_map[primary],
        "available": True,
        "reason": None,
        "complete_vectors": len(vectors),
        "vector_provenance": provenance,
        "excluded_origins": [_public_excluded(item) for item in excluded],
        "vectors": vectors,
        "centerline": inputs.centerline[primary],
    }


def _stopped_entry(inputs, horizon, record, reason):
    return {
        "key": str(horizon),
        "horizons": [horizon],
        "merged": False,
        "merge_note": None,
        "target_year": inputs.study["as_of_year"] + horizon,
        "available": False,
        "reason": reason,
        "complete_vectors": len(record["complete"]),
        "vector_provenance": [
            {"horizon": horizon, "issued_year": item["issued_year"], "target_year": item["target_year"], "first_known_year": item["first_known_year"]}
            for item in record["complete"]
        ],
        "excluded_origins": [_public_excluded(item) for item in record["excluded"]],
        "vectors": (),
    }


def _public_excluded(item):
    return {key: value for key, value in item.items() if key != "vector"}


def run_study(payload):
    """执行完整研究并返回可归档的字典。"""
    inputs = StudyInputs(payload)
    config = inputs.study
    history = {}
    for horizon in inputs.horizons:
        complete, excluded = inputs.build_vectors(horizon)
        history[horizon] = {"complete": complete, "excluded": excluded}
    effective_groups = _build_effective_horizons(inputs, history)
    explicit_windows = {key: config[key] for key in ("train_years", "calibration_years", "backtest_years") if key in config}
    horizon_outputs = []
    for entry in effective_groups:
        if not entry["available"]:
            horizon_outputs.append(_public_horizon(entry, windows=None, split_reason=entry["reason"]))
            continue
        windows, split_reason = _window_partition(inputs, entry["vectors"], explicit_windows)
        if windows is None:
            entry["available"] = False
            entry["reason"] = split_reason
            horizon_outputs.append(_public_horizon(entry, windows=None, split_reason=split_reason))
            continue
        by_window = {
            name: [item for item in entry["vectors"] if item["issued_year"] in set(years)]
            for name, years in windows.items()
        }
        entry["vectors_by_window"] = by_window
        _simulate_horizon(inputs, entry, config["coverage_levels"])
        diagnostics = None
        if entry["available"]:
            calibration_vectors = by_window["calibration"]
            diagnostics = {
                "calibration_window": _window_diagnostics(
                    inputs, calibration_vectors, by_window["training"], config["coverage_levels"],
                    delta_pool=calibration_vectors,
                ),
                "backtest_window": _window_diagnostics(
                    inputs, by_window["backtest"], by_window["training"], config["coverage_levels"],
                    delta_pool=calibration_vectors,
                ),
                "note": "calibration_window fitted the conformal delta; backtest_window never participated in any fit",
            }
        horizon_outputs.append(_public_horizon(entry, windows=windows, split_reason=None, diagnostics=diagnostics))
    archive = {
        "schema_version": 1,
        "study": _public_config(config),
        "input_summary": _input_summary(inputs, history),
        "input_digest": hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest(),
        "cell_universe": [list(cell) for cell in inputs.cell_universe],
        "root_region": inputs.root,
        "horizons": horizon_outputs,
        "method": {
            "residual": "shock = actual pupils minus predicted pupils, one complete joint leaf vector per issued year and horizon",
            "resampling": "whole issued-year vectors are resampled with replacement; cross-region co-movement is preserved",
            "windows": "training, calibration and final backtest issued years are disjoint; only knowable, matured errors enter each origin",
            "missing_vectors": "incomplete historical vectors are listed with reasons and never zero-filled into the pool",
            "calibration": "finite-sample split-conformal absolute-shock quantiles pooled across leaf cells and separately for county stage totals",
            "aggregation": "non-negative integer leaf demand is aggregated within the same simulation before capacity assessment; quantiles need not be additive",
        },
    }
    archive["results_digest"] = hashlib.sha256(canonical(_digest_material(archive)).encode("utf-8")).hexdigest()
    return archive


def _digest_material(archive):
    return {
        "study": archive["study"],
        "input_digest": archive["input_digest"],
        "cell_universe": archive["cell_universe"],
        "horizons": [
            {
                "horizon_group": item["horizon_group"],
                "draw_indices": item.get("draw_indices"),
                "modes": {mode: content["sample_digest"] for mode, content in item.get("modes", {}).items()} if "modes" in item else {},
            }
            for item in archive["horizons"]
        ],
    }


def _public_config(config):
    return {
        "study_id": config["study_id"],
        "as_of_year": config["as_of_year"],
        "model": config["model"],
        "draws": config["draws"],
        "seed": config["seed"],
        "coverage_levels": [str(level) for level in config["coverage_levels"]],
        "dependence_modes": list(config["dependence_modes"]),
        "minimum_complete_vectors": config["minimum_complete_vectors"],
        "insufficient_history": config["insufficient_history"],
        "merge_horizons": [list(group) for group in config["merge_horizons"]],
        "train_years": config.get("train_years"),
        "calibration_years": config.get("calibration_years"),
        "backtest_years": config.get("backtest_years"),
        "target_years": config.get("target_years"),
    }


def _public_horizon(entry, *, windows, split_reason, diagnostics=None):
    result = {
        "horizon_group": entry["horizons"],
        "merged": entry["merged"],
        "merge_note": entry["merge_note"],
        "target_year": entry["target_year"],
        "probability_output_available": entry["available"],
        "reason": split_reason,
        "complete_vector_origins": entry["complete_vectors"],
        "vector_provenance": entry["vector_provenance"],
        "excluded_origins": entry["excluded_origins"],
        "windows": windows,
    }
    if entry.get("draw_indices") is not None:
        result["draw_indices"] = entry["draw_indices"]
        result["resampling_pool_size"] = entry["pool_size"]
        result["modes"] = entry["modes"]
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    return result


def _input_summary(inputs, history):
    knowable = 0
    future_revisions = 0
    for observation in inputs.observations.values():
        if observation["first_known_year"] <= inputs.study["as_of_year"]:
            knowable += 1
        else:
            future_revisions += 1
    per_horizon = {}
    for horizon, record in history.items():
        per_horizon[str(horizon)] = {
            "complete_vectors": len(record["complete"]),
            "excluded_origins": len(record["excluded"]),
        }
    active_capacity_cells = set()
    for horizon in inputs.horizons:
        target = inputs.study["as_of_year"] + horizon
        for capacity in inputs.capacities:
            if capacity.active(target):
                active_capacity_cells.add((target, capacity.region, capacity.settlement, capacity.stage))
    zero_capacity_cells = sorted(
        list(cell) for cell in
        {(inputs.study["as_of_year"] + horizon, *universe_cell)
         for horizon in inputs.horizons for universe_cell in inputs.cell_universe}
        - active_capacity_cells
    )
    return {
        "forecast_records": len(inputs.forecasts),
        "model_forecast_records": sum(1 for row in inputs.forecasts if row["model"] == inputs.study["model"]),
        "observation_records": len(inputs.observations),
        "observations_knowable_at_as_of": knowable,
        "observations_only_knowable_after_as_of": future_revisions,
        "leaf_regions": sorted(inputs.leaf_codes),
        "stages": sorted(inputs.stages),
        "cell_count": len(inputs.cell_universe),
        "capacity_records": len(inputs.capacities),
        "horizons": per_horizon,
        "zero_capacity_cells_in_forecast_years": zero_capacity_cells,
    }
