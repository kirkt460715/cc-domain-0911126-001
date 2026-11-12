from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json

from cohorts.records import InputError, Population, Region, fields, population_index, region_tree, required
from school_pressure.capacity import Capacity, validate_capacity
from school_pressure.stages import Stage, validate_stages
from .scenarios import Scenario, resolve_scenarios, run_scenario, validate_scenario_domain


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError("json", f"duplicate object key {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise InputError("json", f"nonfinite constant {value} is forbidden")


def read_json(path, *, byte_limit=16 * 1024 * 1024):
    with open(path, "rb") as source:
        raw = source.read(byte_limit + 1)
    if len(raw) > byte_limit:
        raise InputError("input", "document exceeds size limit")
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError("input", f"invalid UTF-8 JSON: {exc}") from exc


def collection(record, key, *, maximum=100000):
    value = required(record, key)
    if not isinstance(value, list):
        raise InputError(key, "expected array")
    if len(value) > maximum:
        raise InputError(key, f"too many records; maximum {maximum}")
    return value


@dataclass(frozen=True)
class Batch:
    payload: dict
    regions: dict
    population: dict
    stages: dict
    capacities: tuple
    scenarios: dict

    @classmethod
    def parse(cls, payload):
        fields(payload, {"schema_version", "regions", "population", "stages", "capacities", "scenarios"}, "batch")
        if required(payload, "schema_version") != 1 or isinstance(payload["schema_version"], bool):
            raise InputError("schema_version", "only version 1 is supported")
        regions = region_tree(Region.parse(row) for row in collection(payload, "regions", maximum=10000))
        population = population_index((Population.parse(row) for row in collection(payload, "population")), regions)
        region_children = {region.parent for region in regions.values() if region.parent is not None}
        if any(cell.region in region_children for cell in population):
            raise InputError("population", "population may be supplied only for leaf regions")
        stages = validate_stages(Stage.parse(row) for row in collection(payload, "stages", maximum=30))
        capacities = tuple(Capacity.parse(row) for row in collection(payload, "capacities"))
        validate_capacity(capacities, regions, stages)
        if any(capacity.region in region_children for capacity in capacities):
            raise InputError("capacity", "capacity may be supplied only for leaf regions")
        scenarios = resolve_scenarios(Scenario.parse(row) for row in collection(payload, "scenarios", maximum=100))
        for scenario in scenarios.values():
            validate_scenario_domain(scenario, population, regions, stages)
            for migration in scenario.migrations:
                if migration.source in region_children or migration.destination in region_children:
                    raise InputError("migration", "migration must use leaf regions")
            for birth in scenario.births:
                if birth.region in region_children:
                    raise InputError("birth", "births must use leaf regions")
        canonical_payload = copy.deepcopy(payload)
        return cls(canonical_payload, regions, population, stages, capacities, scenarios)

    @property
    def fingerprint(self):
        encoded = json.dumps(self.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def run(self, scenario, *, progress=None):
        if scenario not in self.scenarios:
            raise InputError("scenario", f"unknown scenario {scenario}")
        result = run_scenario(self.scenarios[scenario], self.population, self.regions, self.stages, self.capacities, progress=progress)
        result["batch_digest"] = self.fingerprint
        return result

    def run_all(self):
        return {code: self.run(code) for code in sorted(self.scenarios)}
