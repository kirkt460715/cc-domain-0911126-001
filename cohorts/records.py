from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping


class InputError(ValueError):
    def __init__(self, field, message, value=None):
        self.field = field
        self.value = value
        super().__init__(f"{field}: {message}")


def required(mapping, field):
    if not isinstance(mapping, Mapping):
        raise InputError(field, "record must be an object")
    if field not in mapping:
        raise InputError(field, "required field missing")
    return mapping[field]


def name(value, field):
    if not isinstance(value, str):
        raise InputError(field, "expected text", value)
    value = value.strip()
    if not value or len(value) > 100:
        raise InputError(field, "text length outside 1..100", value)
    if any(ord(character) < 32 for character in value):
        raise InputError(field, "control characters are forbidden", value)
    return value


def integer(value, field, low=0, high=10**12):
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(field, "expected integer", value)
    if not low <= value <= high:
        raise InputError(field, f"outside {low}..{high}", value)
    return value


def decimal(value, field, low=Decimal(0), high=Decimal(1)):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise InputError(field, "expected finite decimal", value)
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise InputError(field, "invalid decimal", value) from exc
    if not result.is_finite() or not low <= result <= high:
        raise InputError(field, f"outside {low}..{high}", value)
    return result


def fields(record, allowed, label):
    if not isinstance(record, Mapping):
        raise InputError(label, "record must be an object")
    unknown = set(record) - set(allowed)
    if unknown:
        raise InputError(label, f"unknown fields: {sorted(unknown)}")


@dataclass(frozen=True, order=True)
class Cell:
    year: int
    region: str
    settlement: str
    age: int

    def __post_init__(self):
        integer(self.year, "year", 1900, 2200)
        name(self.region, "region")
        if self.settlement not in ("urban", "rural"):
            raise InputError("settlement", "expected urban or rural")
        integer(self.age, "age", 0, 100)

    @classmethod
    def parse(cls, record):
        return cls(
            integer(required(record, "year"), "year", 1900, 2200),
            name(required(record, "region"), "region"),
            required(record, "settlement"),
            integer(required(record, "age"), "age", 0, 100),
        )

    def next(self):
        if self.year == 2200 or self.age == 100:
            raise InputError("cell", "cannot advance beyond supported range")
        return Cell(self.year + 1, self.region, self.settlement, self.age + 1)

    def as_dict(self):
        return {
            "year": self.year,
            "region": self.region,
            "settlement": self.settlement,
            "age": self.age,
        }


@dataclass(frozen=True)
class Population:
    cell: Cell
    count: int

    def __post_init__(self):
        integer(self.count, "count")

    @classmethod
    def parse(cls, record):
        fields(record, {"year", "region", "settlement", "age", "count"}, "population")
        return cls(Cell.parse(record), integer(required(record, "count"), "count"))

    def as_dict(self):
        return {**self.cell.as_dict(), "count": self.count}


@dataclass(frozen=True)
class Region:
    code: str
    label: str
    parent: str | None = None

    @classmethod
    def parse(cls, record):
        fields(record, {"code", "label", "parent"}, "region")
        code = name(required(record, "code"), "code")
        label = name(required(record, "label"), "label")
        parent = record.get("parent")
        if parent is not None:
            parent = name(parent, "parent")
        if parent == code:
            raise InputError("parent", "region cannot contain itself")
        return cls(code, label, parent)


def region_tree(regions):
    by_code = {}
    for region in regions:
        if region.code in by_code:
            raise InputError("regions", f"duplicate region {region.code}")
        by_code[region.code] = region
    for code, region in by_code.items():
        seen = {code}
        parent = region.parent
        while parent is not None:
            if parent not in by_code:
                raise InputError("regions", f"unknown parent {parent}")
            if parent in seen:
                raise InputError("regions", "region hierarchy contains a cycle")
            seen.add(parent)
            parent = by_code[parent].parent
    return by_code


def population_index(records, regions=None):
    result = {}
    for population in records:
        if regions is not None and population.cell.region not in regions:
            raise InputError("region", f"unknown region {population.cell.region}")
        if population.cell in result:
            raise InputError("population", f"duplicate cell {population.cell}")
        result[population.cell] = population.count
    if not result:
        raise InputError("population", "at least one population cell required")
    return result


def ensure_single_year(population):
    years = {cell.year for cell in population}
    if len(years) != 1:
        raise InputError("population", "forecast baseline must contain exactly one year")
    return next(iter(years))


def expand_ancestors(region, regions):
    if region not in regions:
        raise InputError("region", f"unknown region {region}")
    result = []
    while region is not None:
        result.append(region)
        region = regions[region].parent
    return tuple(result)
