from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

from .records import Cell, InputError, decimal, fields, integer, name, required


def apportion(total, weights):
    integer(total, "total")
    if not weights:
        if total:
            raise InputError("weights", "cannot allocate positive total without buckets")
        return {}
    parsed = {key: decimal(value, "weight", high=Decimal(10**12)) for key, value in weights.items()}
    denominator = sum(parsed.values())
    if denominator == 0:
        if total:
            raise InputError("weights", "all weights are zero")
        return {key: 0 for key in weights}
    exact = {key: Decimal(total) * weight / denominator for key, weight in parsed.items()}
    floors = {key: int(value.to_integral_value(rounding=ROUND_FLOOR)) for key, value in exact.items()}
    remaining = total - sum(floors.values())
    ranking = sorted(exact, key=lambda key: (-(exact[key] - floors[key]), str(key)))
    for key in ranking[:remaining]:
        floors[key] += 1
    if sum(floors.values()) != total:
        raise ArithmeticError("apportionment failed conservation")
    return floors


@dataclass(frozen=True)
class Migration:
    year: int
    age: int
    source: str
    source_settlement: str
    destination: str
    destination_settlement: str
    count: int

    @classmethod
    def parse(cls, record):
        allowed = {
            "year", "age", "source", "source_settlement",
            "destination", "destination_settlement", "count",
        }
        fields(record, allowed, "migration")
        result = cls(
            integer(required(record, "year"), "year", 1900, 2199),
            integer(required(record, "age"), "age", 0, 99),
            name(required(record, "source"), "source"),
            required(record, "source_settlement"),
            name(required(record, "destination"), "destination"),
            required(record, "destination_settlement"),
            integer(required(record, "count"), "count"),
        )
        result.source_cell()
        result.destination_cell()
        if result.source_cell() == result.destination_cell():
            raise InputError("migration", "source and destination are identical")
        return result

    def source_cell(self):
        return Cell(self.year, self.source, self.source_settlement, self.age)

    def destination_cell(self):
        return Cell(self.year, self.destination, self.destination_settlement, self.age)


@dataclass(frozen=True)
class ExternalAdjustment:
    cell: Cell
    arrivals: int
    departures: int
    reason: str

    @classmethod
    def parse(cls, record):
        fields(record, {"year", "region", "settlement", "age", "arrivals", "departures", "reason"}, "adjustment")
        return cls(
            Cell.parse(record),
            integer(record.get("arrivals", 0), "arrivals"),
            integer(record.get("departures", 0), "departures"),
            name(required(record, "reason"), "reason"),
        )


@dataclass(frozen=True)
class Birth:
    year: int
    region: str
    settlement: str
    count: int

    @classmethod
    def parse(cls, record):
        fields(record, {"year", "region", "settlement", "count"}, "birth")
        cell = Cell.parse({**record, "age": 0})
        return cls(cell.year, cell.region, cell.settlement, integer(required(record, "count"), "count"))

    @property
    def cell(self):
        return Cell(self.year, self.region, self.settlement, 0)


@dataclass(frozen=True)
class TransitionResult:
    population: dict
    accounting: dict
    movements: tuple


def migrate(population, migrations, *, year=None):
    year_set = {cell.year for cell in population}
    if not year_set and year is not None:
        year_set.add(integer(year, "year", 1900, 2199))
    if len(year_set) != 1:
        raise InputError("population", "migration requires one year")
    actual_year = next(iter(year_set))
    if year is not None and actual_year != year:
        raise InputError("year", "does not match population year")
    year = actual_year
    outflows = {}
    inflows = {}
    seen = set()
    movements = []
    for movement in migrations:
        if movement.year != year:
            raise InputError("migration.year", "does not match population year")
        source = movement.source_cell()
        destination = movement.destination_cell()
        key = (source, destination)
        if key in seen:
            raise InputError("migration", "duplicate origin-destination flow")
        seen.add(key)
        outflows[source] = outflows.get(source, 0) + movement.count
        inflows[destination] = inflows.get(destination, 0) + movement.count
        movements.append({"source": source.as_dict(), "destination": destination.as_dict(), "count": movement.count})
    for source, count in outflows.items():
        if count > population.get(source, 0):
            raise InputError("migration.count", f"outflow exceeds original stock at {source}")
    result = {}
    for cell in sorted(set(population) | set(inflows)):
        result[cell] = population.get(cell, 0) - outflows.get(cell, 0) + inflows.get(cell, 0)
    if sum(result.values()) != sum(population.values()):
        raise ArithmeticError("internal migration changed total population")
    return result, tuple(movements)


def advance(population, *, migrations=(), adjustments=(), births=(), survival=None, year=None):
    if not population and year is None:
        raise InputError("population", "empty baseline")
    if population:
        actual_year = next(iter(population)).year
        if year is not None and actual_year != year:
            raise InputError("year", "does not match baseline")
        year = actual_year
    integer(year, "year", 1900, 2199)
    if any(cell.year != year for cell in population):
        raise InputError("population", "mixed years")
    original = sum(population.values())
    migrated, movements = migrate(population, migrations, year=year)
    arrivals = 0
    departures = 0
    seen_adjustments = set()
    for adjustment in adjustments:
        cell = adjustment.cell
        if cell.year != year:
            raise InputError("adjustment.year", "does not match baseline")
        if cell in seen_adjustments:
            raise InputError("adjustment", "multiple adjustments for same cell")
        seen_adjustments.add(cell)
        available = migrated.get(cell, 0)
        if adjustment.departures > available:
            raise InputError("adjustment.departures", "exceeds available stock")
        migrated[cell] = available + adjustment.arrivals - adjustment.departures
        arrivals += adjustment.arrivals
        departures += adjustment.departures
    result = {}
    deaths = 0
    age_out = 0
    survival = survival or {}
    for cell, count in sorted(migrated.items()):
        rate = decimal(survival.get(cell.age, 1), "survival")
        split = apportion(count, {"alive": rate, "dead": 1 - rate})
        deaths += split["dead"]
        if cell.age == 100:
            age_out += split["alive"]
            continue
        result[cell.next()] = split["alive"]
    born = 0
    seen_births = set()
    for birth in births:
        if birth.year != year + 1:
            raise InputError("birth.year", "births must belong to forecast year")
        if birth.cell in seen_births:
            raise InputError("birth", "duplicate birth cell")
        seen_births.add(birth.cell)
        result[birth.cell] = birth.count
        born += birth.count
    expected = original + arrivals - departures - deaths - age_out + born
    actual = sum(result.values())
    if actual != expected:
        raise ArithmeticError("transition accounting does not balance")
    accounting = {
        "from_year": year,
        "to_year": year + 1,
        "opening": original,
        "internal_moved": sum(movement["count"] for movement in movements),
        "external_arrivals": arrivals,
        "external_departures": departures,
        "deaths": deaths,
        "aged_out": age_out,
        "births": born,
        "closing": actual,
        "balance": actual - expected,
    }
    return TransitionResult(result, accounting, movements)
