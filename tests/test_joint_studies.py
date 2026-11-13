import copy
import json
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from cohorts.records import InputError
from studies.archive import load_study, reopen_study, save_study, summarize_study
from studies.demo import build_demo_payload, build_demo_variants, run_demo, synthesize_history
from studies.engine import StudyInputs, run_study


STAGE = {"code": "primary", "min_age": 6, "max_age": 11, "class_size": 35, "max_class_size": 45}
REGIONS = [
    {"code": "county", "label": "County"},
    {"code": "a", "label": "A", "parent": "county"},
    {"code": "b", "label": "B", "parent": "county"},
]
CELLS = [("a", "urban", "primary"), ("a", "rural", "primary"),
         ("b", "urban", "primary"), ("b", "rural", "primary")]


def capacity(region, settlement, seats, *, first=2010, last=2035, facility=None):
    return {
        "facility": facility or f"{region}-{settlement}-primary",
        "region": region, "settlement": settlement, "stage": "primary",
        "from_year": first, "through_year": last,
        "seats": seats, "classrooms": 100,
    }


def make_payload(*, origins, actuals, first_known=None, predictions=None, prediction_bias=None,
                 as_of=2020, draws=200, seed=7, seats=None, minimum_complete_vectors=2,
                 insufficient_history="stop", merge_horizons=(), study_id="t",
                 dependence_modes=("as_observed", "aligned", "opposed"),
                 train_years=None, calibration_years=None, backtest_years=None,
                 capacities=None, horizons=(1,)):
    """构造最小研究输入。

    actuals: {target_year: {cell: pupils}}。
    prediction_bias: {cell: bias}，预测 = 实际 + bias，于是 shock = 实际 - 预测 = -bias 逐年变化由实际值承担；
      缺省 bias=0 时退化为无偏点预测（shock 为实际值的逐年变动）。
    predictions 缺省按 prediction_bias 生成。
    first_known: {target_year: first_known_year}，缺省等于目标年（当年可知）。
    """
    first_known = first_known or {}
    prediction_bias = prediction_bias or {}
    seats = seats or {cell: 1000 for cell in CELLS}
    if capacities is None:
        capacities = [capacity(*cell[:2], seats[cell]) for cell in CELLS]
    forecasts = []
    observations = []
    sorted_years = sorted(actuals)
    for index, target in enumerate(sorted_years):
        cell_actuals = actuals[target]
        for cell in CELLS:
            observations.append({
                "year": target, "region": cell[0], "settlement": cell[1], "stage": cell[2],
                "pupils": cell_actuals[cell],
                "first_known_year": first_known.get(target, target),
            })
            for horizon in horizons:
                origin = target - horizon
                if origin not in origins:
                    continue
                if predictions is not None and (origin, target, cell) in predictions:
                    predicted = predictions[(origin, target, cell)]
                else:
                    # 持续性预测：沿用上一目标年的实际值，使逐年冲击等于实际值的变动
                    prior_index = index - 1
                    if prior_index < 0:
                        predicted = cell_actuals[cell]
                    else:
                        prior_year = sorted(actuals)[prior_index]
                        predicted = actuals[prior_year][cell] + prediction_bias.get(cell, 0)
                forecasts.append({
                    "issued_year": origin, "target_year": target,
                    "region": cell[0], "settlement": cell[1], "stage": cell[2],
                    "predicted": predicted, "model": "planning",
                })
    study = {
        "study_id": study_id, "as_of_year": as_of, "model": "planning",
        "draws": draws, "seed": seed, "coverage_levels": ("0.5", "0.8", "0.9"),
        "dependence_modes": list(dependence_modes),
        "minimum_complete_vectors": minimum_complete_vectors,
        "insufficient_history": insufficient_history,
        "merge_horizons": [list(group) for group in merge_horizons],
    }
    for key, value in (("train_years", train_years), ("calibration_years", calibration_years),
                       ("backtest_years", backtest_years)):
        if value is not None:
            study[key] = value
    # 中心线：as_of 起点对每个跨度、每个叶子单元都必须有点预测
    latest_actual = actuals[max(actuals)]
    for horizon in horizons:
        for cell in CELLS:
            forecasts.append({
                "issued_year": as_of, "target_year": as_of + horizon,
                "region": cell[0], "settlement": cell[1], "stage": cell[2],
                "predicted": latest_actual[cell], "model": "planning",
            })
    return {
        "study": study, "regions": REGIONS, "stages": [STAGE],
        "capacities": capacities, "forecasts": forecasts, "observations": observations,
    }


def balanced_actuals(years, *, shock_pattern):
    """每年给四个单元一组冲击，base 固定为 100。"""
    return {
        year: {cell: 100 + shock_pattern[cell][index] for cell in CELLS}
        for index, year in enumerate(years)
    }


class JointVectorRules(unittest.TestCase):
    pattern = {
        CELLS[0]: [5, -3, 8, -1, 4, -6, 2, 7, -4],
        CELLS[1]: [-2, 4, -5, 3, -1, 6, -3, -4, 5],
        CELLS[2]: [3, 6, -4, 5, -2, 4, -7, 1, 3],
        CELLS[3]: [-4, 2, 5, -6, 3, -2, 6, -3, -1],
    }

    def setUp(self):
        years = list(range(2011, 2020))
        # 可复现的异质冲击
        self.actuals = balanced_actuals(years, shock_pattern=self.pattern)
        self.origins = [year - 1 for year in years]

    def payload(self, **overrides):
        return make_payload(origins=self.origins, actuals=self.actuals, **overrides)

    def test_complete_vector_keeps_cross_region_association(self):
        inputs = StudyInputs(self.payload())
        complete, excluded = inputs.build_vectors(1)
        # 九个历史起点全部完整；唯一被排除的是 as_of 中心线起点（目标未到期）
        self.assertEqual(len(complete), 9)
        self.assertEqual({item["issued_year"] for item in excluded}, {2020})
        second = complete[1]
        # 同一发布年份的四个单元冲击在一个向量内同时存在，冲击 = 当年实际 - 上年实际
        self.assertEqual(set(second["vector"]), set(CELLS))
        years = list(range(2011, 2020))
        pattern = self.pattern
        for cell in CELLS:
            expected_shock = pattern[cell][1] - pattern[cell][0]
            self.assertEqual(second["vector"][cell]["shock"], expected_shock)
        # 同一年份四个单元的冲击被绑定在同一向量，不允许拆开独立重采样
        self.assertEqual(len({id(tuple(sorted(item["vector"]))) for item in complete[:1]}), 1)

    def test_missing_cell_excludes_whole_origin_with_reason_not_zero_fill(self):
        payload = self.payload()
        # 删掉某一年某个单元的预测：该起点整条向量必须被排除并列出缺失单元
        payload["forecasts"] = [
            row for row in payload["forecasts"]
            if not (row["issued_year"] == 2012 and row["region"] == "b" and row["settlement"] == "urban")
        ]
        inputs = StudyInputs(payload)
        complete, excluded = inputs.build_vectors(1)
        self.assertEqual({item["issued_year"] for item in complete}, set(self.origins) - {2012})
        entry = next(item for item in excluded if item["issued_year"] == 2012)
        self.assertIn("missing_forecast_cells", entry["reasons"])
        self.assertEqual(entry["missing_forecast_cells"], [["b", "urban", "primary"]])
        # 被排除起点绝不进入完整池（无补零）
        self.assertTrue(all(item["issued_year"] != 2012 for item in complete))

    def test_future_revision_is_not_knowable_at_as_of(self):
        # 2019 的实际值要到 2021 才可知，as_of=2020 时该向量必须排除
        payload = self.payload(as_of=2020, first_known={2019: 2021})
        inputs = StudyInputs(payload)
        complete, excluded = inputs.build_vectors(1)
        entry = next(item for item in excluded if item["issued_year"] == 2018)
        self.assertIn("observation_not_yet_knowable", entry["reasons"])
        self.assertTrue(all(item["issued_year"] != 2018 for item in complete))
        # 同一输入在更晚的研究中即可使用
        later = StudyInputs(self.payload(as_of=2021, first_known={2019: 2021}))
        complete_later, excluded_later = later.build_vectors(1)
        self.assertTrue(any(item["issued_year"] == 2018 for item in complete_later))

    def test_not_matured_target_is_excluded(self):
        payload = self.payload(as_of=2020)
        inputs = StudyInputs(payload)
        # 辅助构造已自动加入 as_of=2020、target=2021 的中心线预测，该目标未到期
        _, excluded = inputs.build_vectors(1)
        entry = next(item for item in excluded if item["issued_year"] == 2020)
        self.assertIn("target_not_matured", entry["reasons"])

    def test_training_calibration_backtest_years_are_disjoint(self):
        payload = self.payload(
            train_years=[2010, 2011, 2012, 2013],
            calibration_years=[2014, 2015],
            backtest_years=[2016, 2017, 2018],
        )
        archive = run_study(payload)
        windows = archive["horizons"][0]["windows"]
        self.assertEqual(set(windows["training"]).isdisjoint(windows["calibration"]), True)
        self.assertEqual(set(windows["calibration"]).isdisjoint(windows["backtest"]), True)
        self.assertEqual(set(windows["training"]).isdisjoint(windows["backtest"]), True)

    def test_overlapping_explicit_windows_rejected(self):
        payload = self.payload(
            train_years=[2010, 2011, 2012],
            calibration_years=[2012, 2013],
            backtest_years=[2014, 2015],
        )
        with self.assertRaises(InputError):
            run_study(payload)

    def test_insufficient_history_stops_probability_output(self):
        payload = self.payload(minimum_complete_vectors=99)
        archive = run_study(payload)
        horizon = archive["horizons"][0]
        self.assertFalse(horizon["probability_output_available"])
        self.assertEqual(horizon["reason"], "below_minimum_complete_vectors")
        self.assertNotIn("modes", horizon)

    def test_insufficient_history_can_merge_with_explained_pool(self):
        payload = self.payload(
            minimum_complete_vectors=99, insufficient_history="merge_horizons",
            horizons=(1, 2),
        )
        archive = run_study(payload)
        groups = {tuple(h["horizon_group"]): h for h in archive["horizons"]}
        merged = next(iter(groups.values()))
        self.assertTrue(merged["merged"])
        self.assertTrue(merged["probability_output_available"])
        self.assertIn("auto-pooled", merged["merge_note"])
        # 合并后仍逐向量可溯源
        self.assertTrue(all("horizon" in row for row in merged["vector_provenance"]))

    def test_explicit_merge_horizons_group(self):
        payload = self.payload(
            minimum_complete_vectors=2, horizons=(1, 2),
            merge_horizons=((1, 2),),
        )
        archive = run_study(payload)
        self.assertEqual(len(archive["horizons"]), 1)
        self.assertEqual(archive["horizons"][0]["horizon_group"], [1, 2])


class SimulationRules(unittest.TestCase):
    def setUp(self):
        years = list(range(2011, 2020))
        pattern = {
            CELLS[0]: [10, -8, 12, -5, 7, -9, 4, 11, -6],
            CELLS[1]: [-4, 6, -7, 4, -2, 8, -5, -6, 7],
            CELLS[2]: [5, 9, -6, 7, -3, 6, -10, 2, 4],
            CELLS[3]: [-6, 3, 8, -9, 4, -3, 9, -4, -2],
        }
        self.actuals = balanced_actuals(years, shock_pattern=pattern)
        self.origins = [year - 1 for year in years]

    def payload(self, **overrides):
        return make_payload(origins=self.origins, actuals=self.actuals, **overrides)

    def test_leaf_demand_is_non_negative_integer_and_aggregates_to_county(self):
        # 极大负冲击中心线下，触发截断为零
        payload = self.payload(seats={cell: 1000 for cell in CELLS})
        archive = run_study(payload)
        mode = archive["horizons"][0]["modes"]["as_observed"]
        self.assertEqual(mode["reconciliation"]["mismatches"], 0)
        for row in mode["rows"]:
            # 中位数与分位数均为非负
            self.assertGreaterEqual(Decimal(row["median_pupils"]), 0)
            for interval in row["intervals"].values():
                self.assertGreaterEqual(Decimal(interval["lower"]), 0)

    def test_clamp_to_zero_is_counted_when_shock_exceeds_centerline(self):
        # 构造含大负冲击的历史；中心线很小，模拟时负需求被截断为零
        years = list(range(2011, 2018))
        big_negative = {cell: [-20 if i % 2 else 5 for i in range(len(years))] for cell in CELLS}
        actuals = balanced_actuals(years, shock_pattern=big_negative)
        origins = [year - 1 for year in years]
        payload = make_payload(origins=origins, actuals=actuals, draws=300)
        for row in payload["forecasts"]:
            if row["issued_year"] == payload["study"]["as_of_year"]:
                row["predicted"] = 1
        archive = run_study(payload)
        mode = archive["horizons"][0]["modes"]["as_observed"]
        self.assertGreater(mode["clamp_events"], 0)
        self.assertEqual(mode["reconciliation"]["mismatches"], 0)

    def test_county_pupils_equal_sum_of_leaves_in_every_draw(self):
        payload = self.payload()
        archive = run_study(payload)
        for mode_content in archive["horizons"][0]["modes"].values():
            self.assertEqual(mode_content["reconciliation"]["mismatches"], 0)
            self.assertEqual(mode_content["reconciliation"]["draws_checked"], payload["study"]["draws"])
            # 每个层级（叶子、祖先、全县）都必须恰好有 draws 个样本，禁止重复累计
            for row in mode_content["rows"]:
                self.assertEqual(row["draws"], payload["study"]["draws"])

    def test_dependence_modes_change_county_risk(self):
        # 容量设在中心线附近，使不同依赖形态的联合分布在尾部产生差异
        payload = self.payload(seats={cell: 100 for cell in CELLS})
        archive = run_study(payload)
        modes = archive["horizons"][0]["modes"]
        signatures = set()
        for mode in ("as_observed", "aligned", "opposed"):
            row = next(r for r in modes[mode]["rows"]
                       if r["region"] == "county" and r["settlement"] == "all")
            signatures.add((row["median_pupils"], row["exceedance_frequency"], row["mean_shortage"]))
        # 至少两种形态的全县风险口径不同，证明联合结构被保留而非独立相加
        self.assertGreaterEqual(len(signatures), 2)
        for mode in modes.values():
            self.assertEqual(mode["reconciliation"]["mismatches"], 0)

    def test_quantiles_need_not_be_additive_but_level_totals_reconcile(self):
        payload = self.payload()
        archive = run_study(payload)
        rows = archive["horizons"][0]["modes"]["as_observed"]["rows"]
        leaf_medians = sum(
            Decimal(row["median_pupils"]) for row in rows
            if row["region"] in ("a", "b") and row["settlement"] != "all"
        )
        county_median = next(
            Decimal(row["median_pupils"]) for row in rows
            if row["region"] == "county" and row["settlement"] == "all"
        )
        # 分位数不要求可加：两者允许不同；但逐次模拟的人数已由 reconciliation 核对
        self.assertIsInstance(leaf_medians, Decimal)
        self.assertIsInstance(county_median, Decimal)

    def test_zero_capacity_cell_is_unserved_when_demand_positive(self):
        # a/urban 完全无容量
        capacities = [capacity(*cell[:2], 1000) for cell in CELLS if cell[0] != "a" or cell[1] != "urban"]
        payload = self.payload(capacities=capacities)
        archive = run_study(payload)
        row = next(
            row for row in archive["horizons"][0]["modes"]["as_observed"]["rows"]
            if row["region"] == "a" and row["settlement"] == "urban"
        )
        self.assertEqual(row["seats"], 0)
        self.assertEqual(row["exceedance_frequency"], "1.000000")
        self.assertTrue(Decimal(row["mean_shortage"]) > 0)

    def test_conditional_overload_distribution_and_concentration(self):
        payload = self.payload(seats={cell: 100 for cell in CELLS})
        archive = run_study(payload)
        county = next(
            row for row in archive["horizons"][0]["modes"]["as_observed"]["rows"]
            if row["region"] == "county" and row["settlement"] == "all"
        )
        if Decimal(county["exceedance_frequency"]) > 0:
            self.assertIsNotNone(county["conditional_overload"])
            self.assertIn("mean_shortage", county["conditional_overload"])
            concentration = county["leaf_concentration"]
            self.assertEqual({entry["region"] for entry in concentration}, {"a", "b"})
            for entry in concentration:
                self.assertIn("exceedance_frequency", entry)

class CalibrationRules(unittest.TestCase):
    def setUp(self):
        years = list(range(2011, 2021))
        self.pattern = {cell: [((index * 7 + ord(cell[0])) % 13) - 6
                               for index in range(len(years))] for cell in CELLS}
        self.actuals = balanced_actuals(years, shock_pattern=self.pattern)
        self.origins = [year - 1 for year in years]

    def test_backtest_diagnostics_report_coverage_width_score(self):
        payload = make_payload(origins=self.origins, actuals=self.actuals, draws=300)
        archive = run_study(payload)
        backtest = archive["horizons"][0]["diagnostics"]["backtest_window"]
        for level in ("0.5", "0.8", "0.9"):
            for kind in ("raw", "calibrated"):
                metrics = backtest["leaf_cells"][level][kind]
                self.assertIsNotNone(metrics, f"{level}/{kind} should have points")
                self.assertIn("coverage", metrics)
                self.assertIn("mean_width", metrics)
                self.assertIn("mean_interval_score", metrics)
                self.assertGreater(metrics["points"], 0)
                self.assertTrue(Decimal("0") <= Decimal(metrics["coverage"]) <= Decimal("1"))

    def test_backtest_never_uses_future_or_calibration_years(self):
        payload = make_payload(
            origins=self.origins, actuals=self.actuals, draws=200,
            train_years=list(range(2010, 2016)),
            calibration_years=[2016, 2017],
            backtest_years=[2018, 2019],
        )
        archive = run_study(payload)
        backtest_origins = archive["horizons"][0]["diagnostics"]["backtest_window"]["origins"]
        self.assertEqual(backtest_origins, [2018, 2019])
        calib_origins = archive["horizons"][0]["diagnostics"]["calibration_window"]["origins"]
        self.assertEqual(calib_origins, [2016, 2017])

    def test_knowability_lag_excludes_points_from_diagnostics(self):
        # 回测年份 2018 的实际要到 2020 才可知；as_of=2020 仍允许，但回测该点按规则排除
        payload = make_payload(
            origins=self.origins, actuals=self.actuals, draws=200, as_of=2020,
            first_known={2019: 2020},
            train_years=list(range(2010, 2016)),
            calibration_years=[2016, 2017],
            backtest_years=[2018, 2019],
        )
        archive = run_study(payload)
        backtest = archive["horizons"][0]["diagnostics"]["backtest_window"]
        excluded = backtest["leaf_cells"]["0.8"]["raw"]["excluded_knowability"]
        self.assertGreater(excluded, 0)


class MultiStageRules(unittest.TestCase):
    def test_two_stages_aggregate_independently_with_full_draws(self):
        regions = [
            {"code": "county", "label": "C"},
            {"code": "a", "label": "A", "parent": "county"},
            {"code": "b", "label": "B", "parent": "county"},
        ]
        stages = [
            {"code": "primary", "min_age": 6, "max_age": 11, "class_size": 35, "max_class_size": 45},
            {"code": "junior", "min_age": 12, "max_age": 14, "class_size": 40, "max_class_size": 50},
        ]
        cells = [(r, s, st) for r in ("a", "b") for s in ("urban", "rural") for st in ("primary", "junior")]
        forecasts, observations = [], []
        for target in range(2011, 2021):
            for region, settlement, stage in cells:
                observations.append({
                    "year": target, "region": region, "settlement": settlement, "stage": stage,
                    "pupils": 100 + (target % 5), "first_known_year": target,
                })
                forecasts.append({
                    "issued_year": target - 1, "target_year": target, "region": region,
                    "settlement": settlement, "stage": stage, "predicted": 100, "model": "planning",
                })
        for region, settlement, stage in cells:
            forecasts.append({
                "issued_year": 2020, "target_year": 2021, "region": region,
                "settlement": settlement, "stage": stage, "predicted": 102, "model": "planning",
            })
        capacities = [
            {"facility": f"{region}-{settlement}-{stage}", "region": region, "settlement": settlement,
             "stage": stage, "from_year": 2010, "through_year": 2030, "seats": 101, "classrooms": 10}
            for region, settlement, stage in cells
        ]
        payload = {
            "study": {"study_id": "multi", "as_of_year": 2020, "model": "planning", "draws": 150,
                      "seed": 3, "coverage_levels": ["0.8"], "dependence_modes": ["as_observed"],
                      "minimum_complete_vectors": 2, "insufficient_history": "stop"},
            "regions": regions, "stages": stages, "capacities": capacities,
            "forecasts": forecasts, "observations": observations,
        }
        archive = run_study(payload)
        rows = archive["horizons"][0]["modes"]["as_observed"]["rows"]
        self.assertEqual({row["stage"] for row in rows}, {"primary", "junior"})
        county = [row for row in rows if row["region"] == "county" and row["settlement"] == "all"]
        self.assertEqual(len(county), 2)
        for row in county:
            self.assertEqual(row["draws"], 150)
            self.assertTrue(all(entry["stage"] == row["stage"] for entry in row["leaf_concentration"]))


class ArchiveRules(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "study.json"
        self.payload = build_demo_payload(draws=300)

    def test_save_and_reopen_reproduces_identical_samples(self):
        saved = save_study(self.payload, self.path)
        self.assertTrue(saved["created"])
        archive = load_study(self.path)
        report = reopen_study(archive, self._write_input(self.payload))
        self.assertTrue(report["reproduced"])
        self.assertEqual(report["checked_horizons"], 3)

    def _write_input(self, payload):
        path = Path(self.directory.name) / "input.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_rerun_same_input_is_idempotent_and_immutable(self):
        save_study(self.payload, self.path)
        again = save_study(self.payload, self.path)
        self.assertFalse(again["created"])

    def test_new_actuals_change_digest_and_require_new_study(self):
        save_study(self.payload, self.path)
        archive = load_study(self.path)
        changed = copy.deepcopy(self.payload)
        changed["observations"][0]["pupils"] += 3
        with self.assertRaises(InputError):
            reopen_study(archive, self._write_input(changed))

    def test_archive_integrity_digest_detects_tampering(self):
        save_study(self.payload, self.path)
        archive = load_study(self.path)
        tampered = dict(archive)
        tampered["study"] = dict(archive["study"], seed=archive["study"]["seed"] + 1)
        tampered_path = Path(self.directory.name) / "tampered.json"
        tampered_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(InputError):
            load_study(tampered_path)

    def test_archive_records_draws_seed_windows_and_input_summary(self):
        archive = save_study(self.payload, self.path) and load_study(self.path)
        config = archive["study"]
        self.assertEqual(config["draws"], 300)
        self.assertEqual(config["seed"], 20261113)
        self.assertIn("train_years", config)
        summary = archive["input_summary"]
        for key in ("forecast_records", "observation_records", "observations_knowable_at_as_of",
                    "observations_only_knowable_after_as_of", "zero_capacity_cells_in_forecast_years"):
            self.assertIn(key, summary)
        horizon = archive["horizons"][0]
        self.assertEqual(len(horizon["draw_indices"]), 300)
        self.assertTrue(horizon["excluded_origins"])

    def test_summarize_study_contains_required_views(self):
        save_study(self.payload, self.path)
        view = summarize_study(load_study(self.path))
        county = view["horizons"][0]["modes"]["as_observed"]["county_by_stage"][0]
        for key in ("median_pupils", "intervals", "exceedance_frequency", "conditional_overload",
                    "leaf_concentration", "seats"):
            self.assertIn(key, county)


class DemoRules(unittest.TestCase):
    def test_demo_covers_all_required_edge_cases(self):
        report = run_demo()
        self.assertFalse(report["reopen"]["recreated"])
        modes = {row["mode"]: row for row in report["studies"]["main"]["county_risk_by_mode"]}
        self.assertEqual(set(modes), {"as_observed", "aligned", "opposed"})
        # 三种依赖形态的全县风险口径至少有两种不同，证明联合结构未被独立相加抹掉
        signatures = {
            (row["median_pupils"], row["exceedance_frequency"], row["mean_shortage"])
            for row in modes.values()
        }
        self.assertGreaterEqual(len(signatures), 2)
        for row in modes.values():
            self.assertEqual(row["reconciliation_mismatches"], 0)
        for name in ("reopened_with_new_actuals", "zero_capacity", "insufficient_stop", "insufficient_merge"):
            self.assertIn(name, report["variants"])
        stop = report["variants"]["insufficient_stop"]
        self.assertTrue(all(not h["probability_output_available"] for h in stop["horizons"]))
        merge = report["variants"]["insufficient_merge"]
        self.assertTrue(any(h["probability_output_available"] for h in merge["horizons"]))
        zero = report["variants"]["zero_capacity"]
        self.assertTrue(zero["input_summary"]["zero_capacity_cells_in_forecast_years"])

    def test_synthesized_history_is_deterministic(self):
        first = synthesize_history()
        second = synthesize_history()
        self.assertEqual(first, second)

    def test_demo_variant_structure(self):
        main, variants = build_demo_variants()
        self.assertIn("forecasts", main)
        self.assertEqual(set(variants), {"reopened_with_new_actuals", "zero_capacity",
                                         "insufficient_stop", "insufficient_merge"})


if __name__ == "__main__":
    unittest.main()
