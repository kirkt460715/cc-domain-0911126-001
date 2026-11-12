import copy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from cohorts.records import Cell, InputError, Population, Region, region_tree
from forecast.backtest import ForecastPoint, evaluate, paired_comparison, rolling_origins
from forecast.batches import Batch, read_json
from forecast.estimation import cohort_ratios, estimate_births, estimate_progression, median, quantile, residual_diagnostics
from forecast.scenarios import Scenario, resolve_scenarios
from school_pressure.aggregation import aggregate, rural_urban_gaps, turning_points
from school_pressure.capacity import Capacity, assess, compare_pressure, validate_capacity
from school_pressure.stages import Stage, attendance_matrix, demand_by_stage, validate_stages


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "county.json"


class ForecastIntegration(unittest.TestCase):
    def setUp(self):
        self.payload = read_json(FIXTURE)

    def test_baseline_forecast_balances_every_transition(self):
        batch = Batch.parse(self.payload)
        result = batch.run("baseline")
        self.assertEqual(result["baseline_year"], 2026)
        self.assertEqual(result["through_year"], 2029)
        self.assertEqual(len(result["accounting"]), 3)
        self.assertTrue(all(row["balance"] == 0 for row in result["accounting"]))
        self.assertEqual(len(result["pressure"]), 64)
        for row in result["accounting"]:
            self.assertEqual(row["opening"], row["closing"])
            self.assertEqual(row["births"], 0)

    def test_migration_shifts_local_demand_without_changing_county_total(self):
        batch = Batch.parse(self.payload)
        first = batch.run("baseline")
        second = batch.run("migration")
        self.assertEqual(sum(row["count"] for row in first["population"]), sum(row["count"] for row in second["population"]))
        indexed_first = {(row["year"], row["region"], row["settlement"], row["age"]): row["count"] for row in first["population"]}
        indexed_second = {(row["year"], row["region"], row["settlement"], row["age"]): row["count"] for row in second["population"]}
        self.assertEqual(indexed_second[2027, "east", "urban", 11] - indexed_first[2027, "east", "urban", 11], 20)
        self.assertEqual(indexed_second[2027, "west", "rural", 11] - indexed_first[2027, "west", "rural", 11], -20)
        self.assertEqual(second["accounting"][0]["internal_moved"], 20)
        self.assertEqual(second["accounting"][1]["internal_moved"], 0)

    def test_batch_does_not_mutate_caller_payload(self):
        before = copy.deepcopy(self.payload)
        batch = Batch.parse(self.payload)
        batch.run_all()
        self.assertEqual(self.payload, before)
        self.payload["population"][0]["count"] = 999999
        self.assertNotEqual(batch.payload["population"][0]["count"], 999999)

    def test_canonical_batch_digest_ignores_object_key_order(self):
        batch = Batch.parse(self.payload)
        reordered = {key: self.payload[key] for key in reversed(list(self.payload))}
        self.assertEqual(batch.fingerprint, Batch.parse(reordered).fingerprint)

    def test_aggregate_population_on_parent_is_forbidden(self):
        self.payload["population"][0]["region"] = "county"
        with self.assertRaises(InputError):
            Batch.parse(self.payload)

    def test_mixed_baseline_years_are_forbidden(self):
        self.payload["population"][0]["year"] = 2025
        with self.assertRaises(InputError):
            Batch.parse(self.payload)

    def test_scenario_unknown_region_rejected_at_import(self):
        self.payload["scenarios"][1]["migrations"][0]["destination"] = "unknown"
        with self.assertRaises(InputError):
            Batch.parse(self.payload)

    def test_unknown_scenario_rejected(self):
        batch = Batch.parse(self.payload)
        with self.assertRaises(InputError):
            batch.run("missing")

    def test_inheritance_cycle_rejected(self):
        self.payload["scenarios"][0]["parent"] = "migration"
        with self.assertRaises(InputError):
            Batch.parse(self.payload)

    def test_child_overrides_parent_rate_without_mutating_parent(self):
        parent = Scenario.parse({"code": "base", "horizon": 2, "survival": [{"age": 6, "rate": "0.9"}]})
        child = Scenario.parse({"code": "child", "parent": "base", "horizon": 3, "survival": [{"age": 6, "rate": "0.8"}]})
        resolved = resolve_scenarios([child, parent])
        self.assertEqual(resolved["base"].survival[6], Decimal("0.9"))
        self.assertEqual(resolved["child"].survival[6], Decimal("0.8"))
        self.assertEqual(parent.survival[6], Decimal("0.9"))
        self.assertEqual(resolved["child"].horizon, 3)

    def test_inherited_events_outside_shorter_child_horizon_rejected(self):
        self.payload["scenarios"][0]["births"] = [{"year": 2029, "region": "east", "settlement": "urban", "count": 10}]
        self.payload["scenarios"][1]["horizon"] = 1
        with self.assertRaises(InputError):
            Batch.parse(self.payload)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"schema_version":1,"schema_version":2}')
            with self.assertRaises(InputError):
                read_json(path)

    def test_json_size_limit(self):
        with self.assertRaises(InputError):
            read_json(FIXTURE, byte_limit=10)

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite.json"
            path.write_text('{"rate":NaN}')
            with self.assertRaises(InputError):
                read_json(path)


class SchoolPressureRules(unittest.TestCase):
    def setUp(self):
        self.stage = Stage.parse({"code": "primary", "min_age": 6, "max_age": 11, "class_size": 30, "max_class_size": 40})
        self.stages = {"primary": self.stage}
        self.capacity = Capacity.parse({"facility": "school-a", "region": "a", "settlement": "urban", "stage": "primary", "from_year": 2026, "through_year": 2030, "seats": 100, "classrooms": 3})

    def demand(self, pupils, region="a", settlement="urban"):
        return {"year": 2026, "region": region, "settlement": settlement, "stage": "primary", "pupils": pupils}

    def test_overlapping_stage_ages_rejected(self):
        other = Stage.parse({"code": "junior", "min_age": 11, "max_age": 14, "class_size": 40})
        with self.assertRaises(InputError):
            validate_stages([self.stage, other])

    def test_demand_groups_age_before_rounding(self):
        stage = Stage.parse({"code": "primary", "min_age": 6, "max_age": 11, "class_size": 30, "participation": "0.5"})
        population = {Cell(2026, "a", "urban", 6): 1, Cell(2026, "a", "urban", 7): 1}
        rows, unmatched = demand_by_stage(population, [stage])
        self.assertEqual(rows[0]["pupils"], 1)
        self.assertEqual(rows[0]["expected_pupils"], "1.0")
        self.assertEqual(unmatched, 0)
        self.assertEqual(rows[0]["age_counts"], {"6": 1, "7": 1})

    def test_unmatched_ages_are_reported(self):
        rows, unmatched = demand_by_stage({Cell(2026, "a", "urban", 2): 12}, [self.stage])
        self.assertEqual(rows, [])
        self.assertEqual(unmatched, 12)

    def test_seat_surplus_can_coexist_with_classroom_shortage(self):
        row = assess([self.demand(95)], [self.capacity], self.stages)[0]
        self.assertEqual(row["seat_shortage"], 0)
        self.assertEqual(row["target_class_shortage"], 1)
        self.assertEqual(row["legal_class_shortage"], 0)
        self.assertEqual(row["status"], "watch")
        self.assertIn("small_class_target_unmet", row["reasons"])

    def test_legal_shortage_is_critical(self):
        row = assess([self.demand(121)], [self.capacity], self.stages)[0]
        self.assertEqual(row["seat_shortage"], 21)
        self.assertEqual(row["legal_class_shortage"], 1)
        self.assertEqual(row["status"], "critical")

    def test_positive_demand_without_supply_is_unserved(self):
        row = assess([self.demand(10)], [], self.stages)[0]
        self.assertEqual(row["status"], "unserved")
        self.assertIsNone(row["utilization"])
        self.assertEqual(row["seat_shortage"], 10)

    def test_supply_without_demand_is_spare(self):
        row = assess([self.demand(0, region="b")], [self.capacity], self.stages)
        supplied = next(item for item in row if item["region"] == "a")
        self.assertEqual(supplied["spare_seats"], 100)
        self.assertEqual(supplied["utilization"], "0.000000")
        self.assertEqual(supplied["status"], "adequate")

    def test_capacity_interval_is_inclusive(self):
        self.assertTrue(self.capacity.active(2026))
        self.assertTrue(self.capacity.active(2030))
        self.assertFalse(self.capacity.active(2025))
        self.assertFalse(self.capacity.active(2031))

    def test_same_facility_overlapping_intervals_rejected(self):
        regions = {"a": Region("a", "A")}
        with self.assertRaises(InputError):
            validate_capacity([self.capacity, self.capacity], regions, self.stages)

    def test_attendance_transfer_conserves_pupils(self):
        demand = [self.demand(50), self.demand(30, region="b", settlement="rural")]
        transfers = [{"year": 2026, "source": "b", "source_settlement": "rural", "destination": "a", "destination_settlement": "urban", "stage": "primary", "count": 10}]
        result = attendance_matrix(demand, transfers)
        by_region = {row["region"]: row for row in result}
        self.assertEqual(by_region["a"]["pupils"], 60)
        self.assertEqual(by_region["b"]["pupils"], 20)
        self.assertEqual(by_region["b"]["resident_pupils"], 30)
        self.assertEqual(sum(row["pupils"] for row in result), 80)

    def test_comparison_requires_same_cells(self):
        left = assess([self.demand(50)], [], self.stages)
        right = assess([self.demand(50, region="b")], [], self.stages)
        with self.assertRaises(InputError):
            compare_pressure(left, right)

    def test_comparison_reports_capacity_and_shortage_separately(self):
        left = assess([self.demand(120)], [], self.stages)
        right = assess([self.demand(120)], [self.capacity], self.stages)
        result = compare_pressure(left, right)[0]
        self.assertEqual(result["pupil_change"], 0)
        self.assertEqual(result["seat_change"], 100)
        self.assertEqual(result["shortage_change"], -100)

    def test_parent_reports_local_shortage_separately_from_net_gap(self):
        regions = region_tree([Region("county", "County"), Region("a", "A", "county"), Region("b", "B", "county")])
        rows = assess([self.demand(10), self.demand(50, region="b")], [self.capacity], self.stages)
        result = aggregate(rows, regions)
        parent = next(row for row in result if row["region"] == "county")
        self.assertEqual(parent["seat_shortage"], 50)
        self.assertEqual(parent["net_seat_gap"], -40)
        self.assertEqual(parent["locally_unmet_pupils"], 50)
        self.assertEqual(parent["spare_seats"], 90)

    def test_urban_rural_gap_does_not_treat_missing_as_zero(self):
        rows = assess([self.demand(10)], [], self.stages)
        gap = rural_urban_gaps(rows)[0]
        self.assertFalse(gap["complete"])
        self.assertEqual(gap["missing_settlements"], ["rural"])
        self.assertIsNone(gap["rural_minus_urban"])

    def test_trend_peak_handles_plateau(self):
        rows = [{**self.demand(count), "year": year, "seat_shortage": max(0, count - 20)} for year, count in [(2026, 10), (2027, 30), (2028, 30), (2029, 20)]]
        trend = turning_points(rows)[0]
        self.assertEqual(trend["peak_years"], [2027, 2028])
        self.assertEqual(trend["first_shortage_year"], 2027)
        self.assertEqual(trend["pupil_change"], 10)
        self.assertEqual([row["delta"] for row in trend["annual_changes"]], [20, 0, -10])

    def test_trend_rejects_missing_year(self):
        rows = [{**self.demand(10), "year": year, "seat_shortage": 0} for year in (2026, 2028)]
        with self.assertRaises(InputError):
            turning_points(rows)


class BacktestRules(unittest.TestCase):
    def observation(self, year, pupils, region="a"):
        return {"year": year, "region": region, "settlement": "urban", "stage": "primary", "pupils": pupils}

    def point(self, predicted, *, model="cohort", target=2027, region="a"):
        return ForecastPoint(2026, target, region, "urban", "primary", predicted, model)

    def test_metrics_preserve_bias_direction(self):
        backtest_result = evaluate([self.point(120)], [self.observation(2027, 100)], observed_through=2027)
        row = backtest_result["summary"][0]
        self.assertEqual(row["bias"], "20.000000")
        self.assertEqual(row["mae"], "20.000000")
        self.assertEqual(row["rmse"], "20.000000")
        self.assertEqual(row["wape"], "0.200000")

    def test_zero_actual_does_not_create_infinite_percentage(self):
        backtest_result = evaluate([self.point(10)], [self.observation(2027, 0)], observed_through=2027)
        self.assertIsNone(backtest_result["summary"][0]["wape"])
        self.assertIsNone(backtest_result["details"][0]["absolute_percentage_error"])
        self.assertEqual(backtest_result["summary"][0]["zero_actual_cells"], 1)

    def test_future_points_are_separate_from_missing_observations(self):
        points = [self.point(10), self.point(10, target=2028), self.point(10, region="b")]
        backtest_result = evaluate(points, [self.observation(2027, 10)], observed_through=2027)
        horizon_one = next(row for row in backtest_result["summary"] if row["horizon"] == 1)
        horizon_two = next(row for row in backtest_result["summary"] if row["horizon"] == 2)
        self.assertEqual(horizon_one["matched_cells"], 1)
        self.assertEqual(horizon_one["missing_cells"], 1)
        self.assertEqual(horizon_two["not_yet_observed_cells"], 1)
        self.assertEqual(horizon_two["missing_cells"], 0)

    def test_minimum_sample_suppresses_metrics(self):
        backtest_result = evaluate([self.point(10)], [self.observation(2027, 10)], observed_through=2027, minimum_cells=2)
        row = backtest_result["summary"][0]
        self.assertFalse(row["sufficient"])
        self.assertIsNone(row["mae"])
        self.assertIsNone(row["bias"])
        self.assertIsNone(row["rmse"])

    def test_same_year_forecast_is_not_backtest(self):
        with self.assertRaises(InputError):
            self.point(10, target=2026)

    def test_duplicate_points_rejected(self):
        point = self.point(10)
        with self.assertRaises(InputError):
            evaluate([point, point], [self.observation(2027, 10)], observed_through=2027)

    def test_future_observation_cannot_enter_backtest(self):
        with self.assertRaises(InputError):
            evaluate([self.point(10)], [self.observation(2028, 10)], observed_through=2027)

    def test_paired_comparison_uses_only_matched_cells(self):
        points = [self.point(120, model="left"), self.point(110, model="right"), self.point(100, model="left", region="b")]
        backtest_result = evaluate(points, [self.observation(2027, 100), self.observation(2027, 90, region="b")], observed_through=2027)
        comparison = paired_comparison(backtest_result, "left", "right")
        self.assertEqual(comparison["paired_cells"], 1)
        self.assertEqual(comparison["right_better"], 1)
        self.assertEqual(comparison["unpaired_right_missing"], 1)
        self.assertEqual(comparison["mean_absolute_error_difference"], "-10.000000")

    def test_rolling_training_never_includes_future_years(self):
        records = [Population(Cell(year, "a", "urban", 6), 10) for year in range(2024, 2029)]
        origins = rolling_origins(records, first_origin=2025, last_origin=2027, horizon=2)
        self.assertEqual(len(origins), 3)
        for origin in origins:
            self.assertTrue(all(row.cell.year <= origin["origin"] for row in origin["training"]))
            self.assertEqual({row.cell.year for row in origin["baseline"]}, {origin["origin"]})
            self.assertEqual(origin["target_years"], (origin["origin"] + 1, origin["origin"] + 2))

    def test_rolling_missing_origin_rejected(self):
        records = [Population(Cell(2025, "a", "urban", 6), 10)]
        with self.assertRaises(InputError):
            rolling_origins(records, first_origin=2025, last_origin=2026, horizon=1)


class EstimationRules(unittest.TestCase):
    def test_quantile_interpolates(self):
        self.assertEqual(quantile([0, 10], "0.25"), Decimal("2.50"))
        self.assertEqual(quantile([10, 0], 1), Decimal(10))
        self.assertIsNone(quantile([], "0.5"))
        self.assertEqual(median([Decimal(1), Decimal(3)]), Decimal(2))

    def test_small_denominator_ratio_is_excluded(self):
        population = {Cell(2025, "a", "rural", 5): 2, Cell(2026, "a", "rural", 6): 5}
        rows = cohort_ratios(population, through_year=2026)
        self.assertEqual(rows[0]["reason"], "small_denominator")
        self.assertIsNone(rows[0]["ratio"])

    def test_cohort_estimation_uses_age_advance(self):
        population = {Cell(2024, "a", "urban", 5): 100, Cell(2025, "a", "urban", 6): 110}
        rows = cohort_ratios(population, through_year=2025)
        self.assertEqual(rows[0]["ratio"], Decimal("1.1"))
        self.assertEqual(rows[0]["age"], 5)

    def test_local_insufficient_sample_uses_settlement_pool(self):
        population = {
            Cell(2024, "a", "urban", 5): 100,
            Cell(2025, "a", "urban", 6): 110,
            Cell(2024, "b", "urban", 5): 100,
            Cell(2025, "b", "urban", 6): 130,
        }
        result = estimate_progression(population, [("a", "urban", 5)], through_year=2025)
        self.assertEqual(result[0].level, "settlement_age")
        self.assertEqual(result[0].rate, Decimal("1.2"))
        self.assertEqual(result[0].observations, 2)

    def test_no_evidence_is_explicit_identity_fallback(self):
        result = estimate_progression({}, [("a", "rural", 5)], through_year=2025)
        self.assertEqual(result[0].level, "identity_fallback")
        self.assertEqual(result[0].observations, 0)
        self.assertEqual(result[0].rate, 1)

    def test_future_data_does_not_change_estimate(self):
        population = {Cell(2024, "a", "urban", 5): 100, Cell(2025, "a", "urban", 6): 110, Cell(2026, "a", "urban", 7): 10000}
        ratios = cohort_ratios(population, through_year=2025)
        self.assertEqual(len(ratios), 1)
        self.assertEqual(ratios[0]["year"], 2024)

    def test_birth_median_uses_only_trailing_window(self):
        records = [Population(Cell(year, "a", "rural", 0), count) for year, count in [(2022, 10000), (2023, 20), (2024, 30), (2025, 40), (2026, 5000)]]
        result = estimate_births(records, target_year=2026, through_year=2025, window=3)
        self.assertEqual(result[0]["count"], 30)
        self.assertEqual(result[0]["training_years"], [2023, 2024, 2025])
        self.assertEqual(result[0]["dispersion_mad"], "10")

    def test_residual_diagnostics_counts_direction(self):
        points = [ForecastPoint(2026, 2027, region, "urban", "primary", predicted, "m") for region, predicted in [("a", 90), ("b", 100), ("c", 120)]]
        actuals = [{"year": 2027, "region": region, "settlement": "urban", "stage": "primary", "pupils": 100} for region in ("a", "b", "c")]
        backtest_result = evaluate(points, actuals, observed_through=2027)
        diagnostics = residual_diagnostics(backtest_result)[0]
        self.assertEqual(diagnostics["positive_errors"], 1)
        self.assertEqual(diagnostics["negative_errors"], 1)
        self.assertEqual(diagnostics["zero_errors"], 1)
        self.assertEqual(diagnostics["signed_runs"], 2)
        self.assertEqual(diagnostics["median_error"], "0")
