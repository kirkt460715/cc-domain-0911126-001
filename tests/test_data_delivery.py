import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from cohorts.records import InputError
from cohorts.tabular import export_bundle, read_bundle, read_table
from forecast.batches import Batch, read_json
from forecast.quality import QualityPolicy, audit, input_findings, output_findings
from releases.api import ApiServer
from releases.backup import backup, restore
from releases.exports import export_release, read_export_result, verify_export
from releases.ledger import ConflictError, Ledger
from releases.queries import PressureQuery, pressure_page, pressure_totals, release_facets
from releases.worker import Worker, queue_status
from school_pressure.portfolio import compare_scenarios, envelope, rank_hotspots, scenario_summary


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "county.json"


class DeliveryCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root / "ledger.sqlite3"
        self.batch = Batch.parse(read_json(FIXTURE))

    def release(self):
        with Ledger(self.database) as ledger:
            batch_id, _ = ledger.put_batch(self.batch.payload)
            run_id, _ = ledger.create_run(batch_id, "baseline", "release-request")
            revision = ledger.claim(run_id, "test", clock=100, duration=100)
            ledger.complete(run_id, "test", revision, self.batch.run("baseline"), clock=101)
            ledger.publish(run_id, "annual")
            return ledger.get_release("annual")


class TabularDelivery(DeliveryCase):
    def test_csv_roundtrip_preserves_analysis(self):
        directory = self.root / "bundle"
        names = export_bundle(self.batch.payload, directory)
        self.assertIn("population.csv", names)
        self.assertIn("migrations.csv", names)
        payload, issues = read_bundle(directory)
        self.assertEqual(issues, ())
        imported = Batch.parse(payload)
        self.assertEqual(imported.run("baseline")["population"], self.batch.run("baseline")["population"])
        self.assertEqual(imported.run("migration")["pressure"], self.batch.run("migration")["pressure"])

    def test_missing_required_file_rejects_bundle(self):
        directory = self.root / "bundle"
        export_bundle(self.batch.payload, directory)
        (directory / "population.csv").unlink()
        with self.assertRaises(InputError):
            read_bundle(directory)

    def test_header_typo_is_not_silently_ignored(self):
        path = self.root / "population.csv"
        path.write_text("year,region,settlement,age,counts\n2026,a,urban,6,10\n")
        with self.assertRaises(InputError):
            read_table(path, "population")

    def test_duplicate_headers_rejected(self):
        path = self.root / "regions.csv"
        path.write_text("code,label,label\na,A,A\n")
        with self.assertRaises(InputError):
            read_table(path, "regions")

    def test_invalid_row_does_not_create_partial_bundle(self):
        directory = self.root / "bundle"
        export_bundle(self.batch.payload, directory)
        path = directory / "population.csv"
        with path.open("a") as output:
            output.write("2026,east,urban,6,not-an-integer\n")
        payload, issues = read_bundle(directory)
        self.assertIsNone(payload)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].field, "count")
        self.assertEqual(issues[0].table, "population")

    def test_quoted_newline_and_comma_are_one_label(self):
        path = self.root / "regions.csv"
        path.write_text('code,label\na,"First,\nSecond"\n')
        table = read_table(path, "regions")
        self.assertEqual(table.issues, ())
        self.assertEqual(table.rows[0]["label"], "First,\nSecond")

    def test_noncanonical_integer_is_reported(self):
        path = self.root / "population.csv"
        path.write_text("year,region,settlement,age,count\n2026,a,urban,06,10\n")
        table = read_table(path, "population")
        self.assertEqual(table.rows, ())
        self.assertEqual(table.issues[0].field, "age")
        with self.assertRaises(InputError):
            table.require_valid()

    def test_row_width_mismatch_keeps_physical_line(self):
        path = self.root / "regions.csv"
        path.write_text("code,label\na,A,extra\n")
        table = read_table(path, "regions")
        self.assertEqual(table.issues[0].line, 2)
        self.assertEqual(table.issues[0].field, "row")

    def test_optional_empty_parent_is_omitted(self):
        path = self.root / "regions.csv"
        path.write_text("code,label,parent\na,A,\n")
        table = read_table(path, "regions")
        self.assertEqual(table.require_valid(), ({"code": "a", "label": "A"},))

    def test_invalid_utf8_is_rejected(self):
        path = self.root / "regions.csv"
        path.write_bytes(b"code,label\na,\xff\n")
        with self.assertRaises(InputError):
            read_table(path, "regions")


class QualityAndPortfolio(DeliveryCase):
    def test_audit_valid_results_with_assumption_warning(self):
        results = self.batch.run_all()
        report = audit(self.batch, results.values())
        self.assertTrue(report["passed"])
        self.assertEqual(report["counts"]["error"], 0)
        self.assertIn("large_migration_assumption", [finding["code"] for finding in report["findings"]])

    def test_missing_age_is_not_explicit_zero(self):
        payload = copy.deepcopy(self.batch.payload)
        payload["population"] = [row for row in payload["population"] if not (row["region"] == "east" and row["settlement"] == "urban" and row["age"] == 6)]
        batch = Batch.parse(payload)
        findings = input_findings(batch)
        missing = [finding for finding in findings if finding.code == "missing_age_cells"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].evidence["missing_ages"], [6])

    def test_corrupted_account_is_an_error(self):
        result = self.batch.run("baseline")
        result["accounting"][0]["closing"] += 1
        findings = output_findings(self.batch, result)
        codes = {finding.code for finding in findings}
        self.assertIn("unbalanced_accounting", codes)
        self.assertIn("population_account_disagreement", codes)

    def test_corrupted_pressure_is_an_error(self):
        result = self.batch.run("baseline")
        result["pressure"][0]["seat_shortage"] += 1
        findings = output_findings(self.batch, result)
        self.assertIn("pressure_arithmetic", {finding.code for finding in findings})

    def test_foreign_batch_result_rejected(self):
        result = self.batch.run("baseline")
        result["batch_digest"] = "different"
        with self.assertRaises(InputError):
            audit(self.batch, [result])

    def test_policy_normalizes_decimal_text(self):
        policy = QualityPolicy(migration_fraction_warning="0.2", capacity_utilization_warning="1.1")
        report = audit(self.batch, self.batch.run_all().values(), policy)
        self.assertTrue(report["passed"])
        self.assertGreater(report["counts"]["warning"], 0)

    def test_summary_recomputes_utilization_from_totals(self):
        result = self.batch.run("baseline")
        summary = scenario_summary(result)[0]
        first_year = [row for row in result["pressure"] if row["year"] == 2026]
        self.assertEqual(summary["pupils"], sum(row["pupils"] for row in first_year))
        self.assertEqual(summary["locally_unmet_pupils"], sum(row["seat_shortage"] for row in first_year))
        self.assertEqual(summary["net_seat_gap"], summary["pupils"] - summary["seats"])

    def test_hotspots_are_sorted_and_bounded(self):
        result = self.batch.run("baseline")
        hotspots = rank_hotspots(result, year=2027, limit=3)
        self.assertEqual(len(hotspots), 3)
        self.assertEqual([row["rank"] for row in hotspots], [1, 2, 3])
        self.assertTrue(all(row["year"] == 2027 for row in hotspots))
        self.assertEqual([row["seat_shortage"] for row in hotspots], sorted((row["seat_shortage"] for row in hotspots), reverse=True))

    def test_envelope_is_scenario_range_not_probability_interval(self):
        results = self.batch.run_all()
        ranges = envelope(results.values())
        cell = next(row for row in ranges if (row["year"], row["region"], row["settlement"], row["stage"]) == (2027, "east", "urban", "primary"))
        self.assertEqual(cell["scenario_count"], 2)
        self.assertEqual(cell["pupil_range"], 20)
        self.assertEqual(cell["highest_demand_scenarios"], ["migration"])

    def test_scenario_comparison_counts_reallocation(self):
        results = self.batch.run_all()
        comparison = compare_scenarios(results["baseline"], results["migration"])
        year = next(row for row in comparison["summary"] if row["year"] == 2027)
        self.assertEqual(year["pupil_change"], 0)
        self.assertEqual(year["seat_change"], 0)
        self.assertTrue(any(row["pupil_change"] == 20 for row in comparison["differences"]))


class QueryAndExport(DeliveryCase):
    def test_pages_cover_each_cell_exactly_once(self):
        release = self.release()
        query = PressureQuery(limit=7)
        collected = []
        cursor = None
        while True:
            page = pressure_page(release, query, cursor=cursor)
            collected.extend(page["rows"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        keys = [tuple(row[field] for field in ("year", "region", "settlement", "stage")) for row in collected]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys), 64)
        self.assertEqual(keys, sorted(keys))

    def test_cursor_is_bound_to_filters(self):
        release = self.release()
        first = pressure_page(release, PressureQuery(limit=2))
        with self.assertRaises(InputError):
            pressure_page(release, PressureQuery(limit=2, regions=("east",)), cursor=first["next_cursor"])

    def test_cursor_allows_page_size_change(self):
        release = self.release()
        first = pressure_page(release, PressureQuery(limit=2))
        second = pressure_page(release, PressureQuery(limit=5), cursor=first["next_cursor"])
        self.assertEqual(second["returned_cells"], 5)
        self.assertNotEqual(first["rows"][-1], second["rows"][0])

    def test_projection_does_not_mutate_release(self):
        release = self.release()
        page = pressure_page(release, PressureQuery(years=(2027,), columns=("year", "pupils")))
        self.assertTrue(all(set(row) == {"year", "pupils"} for row in page["rows"]))
        page["rows"][0]["pupils"] = -1
        self.assertTrue(all(row["pupils"] >= 0 for row in release["result"]["pressure"]))

    def test_totals_apply_filters_before_grouping(self):
        release = self.release()
        totals = pressure_totals(release, PressureQuery(years=(2027,), regions=("east",)), group_by=("settlement",))
        self.assertEqual(len(totals), 2)
        self.assertEqual({row["settlement"] for row in totals}, {"urban", "rural"})
        self.assertEqual(sum(row["cells"] for row in totals), 8)

    def test_facets_respect_current_filter(self):
        release = self.release()
        facets = release_facets(release, PressureQuery(years=(2027,), regions=("east",)))
        self.assertEqual(facets["year"], [2027])
        self.assertEqual(facets["region"], ["east"])
        self.assertEqual(facets["settlement"], ["rural", "urban"])

    def test_bad_cursor_is_rejected(self):
        release = self.release()
        with self.assertRaises(InputError):
            pressure_page(release, cursor="this is not base64")

    def test_export_is_integrity_checked_and_idempotent(self):
        release = self.release()
        destination = self.root / "export"
        with Ledger(self.database) as ledger:
            manifest, created = export_release(ledger, "annual", destination)
            repeated, recreated = export_release(ledger, "annual", destination)
        self.assertTrue(created)
        self.assertFalse(recreated)
        self.assertEqual(manifest, repeated)
        self.assertEqual(verify_export(destination), manifest)
        self.assertEqual(read_export_result(destination), release["result"])
        self.assertTrue((destination / "pressure.csv").read_text().startswith("year,region,settlement,stage"))

    def test_tampered_export_is_rejected(self):
        self.release()
        destination = self.root / "export"
        with Ledger(self.database) as ledger:
            export_release(ledger, "annual", destination)
        with (destination / "population.csv").open("a") as output:
            output.write("tampered\n")
        with self.assertRaises(ConflictError):
            verify_export(destination)

    def test_unlisted_file_is_rejected(self):
        self.release()
        destination = self.root / "export"
        with Ledger(self.database) as ledger:
            export_release(ledger, "annual", destination)
        (destination / "unexpected.txt").write_text("extra")
        with self.assertRaises(ConflictError):
            verify_export(destination)


class BackupAndWorker(DeliveryCase):
    def test_backup_restores_identical_release(self):
        release = self.release()
        destination = self.root / "backup.sqlite3"
        with Ledger(self.database) as ledger:
            receipt = backup(ledger, destination)
        restored = self.root / "restored.sqlite3"
        result = restore(destination, restored, expected_sha256=receipt["sha256"])
        self.assertEqual(result["integrity"]["releases"], 1)
        with Ledger(restored) as ledger:
            self.assertEqual(ledger.get_release("annual"), release)

    def test_restore_rejects_checksum_mismatch(self):
        self.release()
        with Ledger(self.database) as ledger:
            backup(ledger, self.root / "backup.sqlite3")
        with self.assertRaises(ConflictError):
            restore(self.root / "backup.sqlite3", self.root / "restored.sqlite3", expected_sha256="0" * 64)
        self.assertFalse((self.root / "restored.sqlite3").exists())

    def test_backup_does_not_overwrite_existing_target(self):
        self.release()
        destination = self.root / "existing"
        destination.write_text("keep")
        with Ledger(self.database) as ledger:
            with self.assertRaises(ConflictError):
                backup(ledger, destination)
        self.assertEqual(destination.read_text(), "keep")

    def test_worker_drains_pending_and_recovers_expired_runs(self):
        with Ledger(self.database) as ledger:
            batch_id, _ = ledger.put_batch(self.batch.payload)
            pending, _ = ledger.create_run(batch_id, "baseline", "pending")
            expired, _ = ledger.create_run(batch_id, "migration", "expired")
            ledger.claim(expired, "lost", clock=50, duration=10)
        before = queue_status(self.database, clock=100)
        self.assertEqual(before["expired_running"], 1)
        outcome = Worker(self.database, clock=lambda: 100).drain()
        self.assertEqual({row["run_id"] for row in outcome["completed"]}, {pending, expired})
        self.assertEqual(outcome["failed"], [])
        self.assertEqual(outcome["conflicted"], [])
        self.assertEqual(queue_status(self.database, clock=100)["counts"]["complete"], 2)

    def test_worker_leaves_active_lease_alone(self):
        with Ledger(self.database) as ledger:
            batch_id, _ = ledger.put_batch(self.batch.payload)
            run_id, _ = ledger.create_run(batch_id, "baseline", "active")
            ledger.claim(run_id, "other", clock=90, duration=100)
        outcome = Worker(self.database, clock=lambda: 100).drain()
        self.assertEqual(outcome["candidate_count"], 0)
        with Ledger(self.database) as ledger:
            self.assertEqual(ledger.get_run(run_id)["lease_owner"], "other")

    def test_worker_complete_run_is_reused(self):
        release = self.release()
        outcome = Worker(self.database, clock=lambda: 300).execute(release["run_id"])
        self.assertTrue(outcome.reused)
        self.assertEqual(outcome.result_digest, release["digest"])


class ApiDelivery(DeliveryCase):
    def setUp(self):
        super().setUp()
        self.server = ApiServer(("127.0.0.1", 0), self.database)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, *, raw=None, content_type="application/json"):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        try:
            content = raw if raw is not None else json.dumps(body).encode() if body is not None else None
            connection.request(method, path, body=content, headers={"Content-Type": content_type})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_api_full_analysis_release_and_query(self):
        status, imported = self.request("POST", "/batches", self.batch.payload)
        self.assertEqual(status, 201)
        status, run = self.request("POST", "/runs", {"batch_id": imported["batch_id"], "scenario": "baseline", "request_key": "api-request"})
        self.assertEqual(status, 201)
        status, completed = self.request("POST", "/execute", {"run_id": run["run_id"]})
        self.assertEqual(status, 200)
        self.assertFalse(completed["reused"])
        status, released = self.request("POST", "/releases", {"run_id": run["run_id"], "label": "api-annual"})
        self.assertEqual(status, 201)
        self.assertTrue(released["created"])
        status, page = self.request("GET", "/pressure?label=api-annual&year=2027&limit=3")
        self.assertEqual(status, 200)
        self.assertEqual(page["matched_cells"], 16)
        self.assertEqual(page["returned_cells"], 3)
        self.assertIsNotNone(page["next_cursor"])
        status, repeated = self.request("POST", "/execute", {"run_id": run["run_id"]})
        self.assertEqual(status, 200)
        self.assertTrue(repeated["reused"])
        self.assertEqual(repeated["digest"], completed["digest"])

    def test_duplicate_json_keys_rejected_by_api(self):
        status, error = self.request("POST", "/batches", raw=b'{"schema_version":1,"schema_version":2}')
        self.assertEqual(status, 422)
        self.assertEqual(error["error"], "invalid_input")

    def test_api_rejects_content_type(self):
        status, error = self.request("POST", "/batches", self.batch.payload, content_type="text/plain")
        self.assertEqual(status, 422)
        self.assertEqual(error["field"], "body")

    def test_api_rejects_unknown_query_parameters(self):
        status, error = self.request("GET", "/health?unexpected=value")
        self.assertEqual(status, 422)
        self.assertEqual(error["field"], "query")

    def test_api_unknown_route_is_json_404(self):
        status, error = self.request("GET", "/missing")
        self.assertEqual(status, 404)
        self.assertEqual(error, {"error": "route_not_found"})

    def test_api_missing_run_id_is_input_error(self):
        status, error = self.request("POST", "/execute", {})
        self.assertEqual(status, 422)
        self.assertEqual(error["field"], "run_id")

    def test_api_pending_run_cannot_be_released(self):
        _, imported = self.request("POST", "/batches", self.batch.payload)
        _, run = self.request("POST", "/runs", {"batch_id": imported["batch_id"], "scenario": "baseline", "request_key": "pending-api"})
        status, error = self.request("POST", "/releases", {"run_id": run["run_id"], "label": "premature"})
        self.assertEqual(status, 409)
        self.assertEqual(error["error"], "state_conflict")
