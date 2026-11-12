from decimal import Decimal
import unittest

from cohorts.records import Cell, InputError, Population, Region, decimal, integer, population_index, region_tree
from cohorts.transitions import Birth, ExternalAdjustment, Migration, advance, apportion, migrate


class PrimitiveRules(unittest.TestCase):
    def test_boolean_is_not_a_count(self):
        with self.assertRaises(InputError):
            integer(True, "count")

    def test_fraction_is_not_an_integer(self):
        with self.assertRaises(InputError):
            integer(1.5, "count")

    def test_integer_bounds_are_inclusive(self):
        self.assertEqual(integer(0, "count", 0, 2), 0)
        self.assertEqual(integer(2, "count", 0, 2), 2)
        with self.assertRaises(InputError):
            integer(3, "count", 0, 2)

    def test_nonfinite_rates_are_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity", float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(InputError):
                    decimal(value, "rate")

    def test_rate_bounds(self):
        self.assertEqual(decimal("0.125", "rate"), Decimal("0.125"))
        with self.assertRaises(InputError):
            decimal("1.001", "rate")
        with self.assertRaises(InputError):
            decimal("-0.001", "rate")

    def test_population_unknown_fields(self):
        with self.assertRaises(InputError):
            Population.parse({"year": 2026, "region": "a", "settlement": "urban", "age": 6, "count": 4, "typo": 1})

    def test_population_duplicate_cells(self):
        record = Population(Cell(2026, "a", "urban", 6), 10)
        with self.assertRaises(InputError):
            population_index([record, record])

    def test_population_unknown_region(self):
        record = Population(Cell(2026, "a", "urban", 6), 10)
        with self.assertRaises(InputError):
            population_index([record], {"b": Region("b", "B")})

    def test_region_cycles(self):
        with self.assertRaises(InputError):
            region_tree([Region("a", "A", "b"), Region("b", "B", "a")])

    def test_region_missing_parent(self):
        with self.assertRaises(InputError):
            region_tree([Region("a", "A", "missing")])

    def test_region_duplicate(self):
        with self.assertRaises(InputError):
            region_tree([Region("a", "A"), Region("a", "Other")])

    def test_cell_advance(self):
        self.assertEqual(Cell(2026, "a", "rural", 5).next(), Cell(2027, "a", "rural", 6))
        with self.assertRaises(InputError):
            Cell(2200, "a", "rural", 5).next()


class ApportionmentRules(unittest.TestCase):
    def test_rounding_preserves_total(self):
        result = apportion(11, {"a": 1, "b": 1, "c": 1})
        self.assertEqual(result, {"a": 4, "b": 4, "c": 3})
        self.assertEqual(sum(result.values()), 11)

    def test_ties_do_not_depend_on_insertion_order(self):
        first = apportion(1, {"rural": 1, "urban": 1})
        second = apportion(1, {"urban": 1, "rural": 1})
        self.assertEqual(first, second)
        self.assertEqual(first["rural"], 1)

    def test_zero_total_and_zero_weights(self):
        self.assertEqual(apportion(0, {"a": 0, "b": 0}), {"a": 0, "b": 0})
        self.assertEqual(apportion(0, {}), {})

    def test_positive_total_needs_positive_weights(self):
        with self.assertRaises(InputError):
            apportion(1, {})
        with self.assertRaises(InputError):
            apportion(1, {"a": 0})

    def test_negative_weight_rejected(self):
        with self.assertRaises(InputError):
            apportion(10, {"a": -1, "b": 3})

    def test_survival_rounding_is_explicit(self):
        population = {Cell(2026, "a", "urban", 5): 3}
        result = advance(population, survival={5: Decimal("0.5")})
        self.assertEqual(result.population[Cell(2027, "a", "urban", 6)], 2)
        self.assertEqual(result.accounting["deaths"], 1)
        self.assertEqual(result.accounting["balance"], 0)


class MigrationRules(unittest.TestCase):
    def setUp(self):
        self.a = Cell(2026, "a", "rural", 10)
        self.b = Cell(2026, "b", "urban", 10)
        self.c = Cell(2026, "c", "urban", 10)

    def movement(self, source, destination, count):
        return Migration(2026, 10, source.region, source.settlement, destination.region, destination.settlement, count)

    def test_internal_movement_conserves_population(self):
        result, movements = migrate({self.a: 10, self.b: 20}, [self.movement(self.a, self.b, 6)])
        self.assertEqual(result[self.a], 4)
        self.assertEqual(result[self.b], 26)
        self.assertEqual(sum(result.values()), 30)
        self.assertEqual(movements[0]["count"], 6)

    def test_incoming_population_cannot_finance_same_year_outflow(self):
        flows = [self.movement(self.a, self.b, 10), self.movement(self.b, self.c, 15)]
        with self.assertRaises(InputError):
            migrate({self.a: 10, self.b: 10}, flows)

    def test_split_outflows_are_checked_together(self):
        flows = [self.movement(self.a, self.b, 6), self.movement(self.a, self.c, 5)]
        with self.assertRaises(InputError):
            migrate({self.a: 10}, flows)

    def test_duplicate_flow_is_not_silently_added(self):
        movement = self.movement(self.a, self.b, 2)
        with self.assertRaises(InputError):
            migrate({self.a: 10}, [movement, movement])

    def test_input_order_does_not_change_result(self):
        flows = [self.movement(self.a, self.b, 4), self.movement(self.a, self.c, 3)]
        first, _ = migrate({self.a: 10}, flows)
        second, _ = migrate({self.a: 10}, list(reversed(flows)))
        self.assertEqual(first, second)
        self.assertEqual(first[self.a], 3)

    def test_destination_can_be_new_cell(self):
        result, _ = migrate({self.a: 5}, [self.movement(self.a, self.b, 5)])
        self.assertEqual(result[self.a], 0)
        self.assertEqual(result[self.b], 5)

    def test_external_arrivals_and_departures_balance(self):
        adjustment = ExternalAdjustment(self.a, 7, 3, "synthetic boundary movement")
        result = advance({self.a: 10}, adjustments=[adjustment])
        self.assertEqual(sum(result.population.values()), 14)
        self.assertEqual(result.accounting["external_arrivals"], 7)
        self.assertEqual(result.accounting["external_departures"], 3)
        self.assertEqual(result.accounting["balance"], 0)

    def test_external_departure_cannot_borrow_from_arrivals(self):
        adjustment = ExternalAdjustment(self.a, 100, 11, "synthetic")
        with self.assertRaises(InputError):
            advance({self.a: 10}, adjustments=[adjustment])

    def test_births_do_not_age_in_birth_year(self):
        birth = Birth(2027, "a", "rural", 8)
        result = advance({self.a: 10}, births=[birth])
        self.assertEqual(result.population[birth.cell], 8)
        self.assertEqual(result.accounting["births"], 8)
        self.assertEqual(result.accounting["closing"], 18)

    def test_duplicate_birth_cell_rejected(self):
        birth = Birth(2027, "a", "rural", 8)
        with self.assertRaises(InputError):
            advance({self.a: 10}, births=[birth, birth])

    def test_birth_year_must_be_next_year(self):
        with self.assertRaises(InputError):
            advance({self.a: 10}, births=[Birth(2028, "a", "rural", 8)])

    def test_terminal_age_is_accounted_separately(self):
        old = Cell(2026, "a", "rural", 100)
        result = advance({old: 10}, survival={100: "0.8"})
        self.assertEqual(result.population, {})
        self.assertEqual(result.accounting["deaths"], 2)
        self.assertEqual(result.accounting["aged_out"], 8)
        self.assertEqual(result.accounting["closing"], 0)
        self.assertEqual(result.accounting["balance"], 0)

    def test_empty_year_can_be_advanced_with_explicit_calendar(self):
        result = advance({}, year=2026, births=[Birth(2027, "a", "rural", 4)])
        self.assertEqual(result.accounting["opening"], 0)
        self.assertEqual(result.accounting["closing"], 4)
        self.assertEqual(result.population[Cell(2027, "a", "rural", 0)], 4)

    def test_explicit_calendar_must_match_population(self):
        with self.assertRaises(InputError):
            advance({self.a: 10}, year=2025)

    def test_combined_transition_reconciles_all_components(self):
        migration = self.movement(self.a, self.b, 5)
        adjustment = ExternalAdjustment(self.b, 3, 1, "synthetic")
        births = [Birth(2027, "a", "rural", 4)]
        result = advance({self.a: 10, self.b: 10}, migrations=[migration], adjustments=[adjustment], births=births, survival={10: "0.5"})
        accounting = result.accounting
        self.assertEqual(accounting["opening"], 20)
        self.assertEqual(accounting["internal_moved"], 5)
        self.assertEqual(accounting["external_arrivals"], 3)
        self.assertEqual(accounting["external_departures"], 1)
        self.assertEqual(accounting["births"], 4)
        self.assertEqual(accounting["closing"], sum(result.population.values()))
        self.assertEqual(accounting["closing"], 20 + 3 - 1 - accounting["deaths"] + 4)
        self.assertEqual(accounting["balance"], 0)
