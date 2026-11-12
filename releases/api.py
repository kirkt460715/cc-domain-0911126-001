from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

from cohorts.records import InputError, fields, required
from forecast.batches import Batch, reject_constant, unique_object
from forecast.quality import audit
from school_pressure.portfolio import scenario_summary
from .ledger import ConflictError, Ledger, canonical
from .queries import PressureQuery, pressure_page, pressure_totals, release_facets


class ApiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, database, *, maximum_requests=16, maximum_body=16 * 1024 * 1024):
        self.database = Path(database)
        self.maximum_body = maximum_body
        self.capacity = threading.BoundedSemaphore(maximum_requests)
        with Ledger(self.database) as ledger:
            ledger.verify()
        super().__init__(address, ApiHandler)


class ApiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CohortAnalysis/1"

    def log_message(self, format, *args):
        return

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def respond(self, status, payload):
        content = (canonical(payload) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def read_body(self):
        if self.headers.get("Transfer-Encoding"):
            raise InputError("body", "transfer encoding is not supported")
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            raise InputError("body", "Content-Length is required")
        length = int(length)
        if length > self.server.maximum_body:
            raise InputError("body", "request exceeds size limit")
        if self.headers.get_content_type() != "application/json":
            raise InputError("body", "Content-Type must be application/json")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise InputError("body", "request body ended early")
        try:
            return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise InputError("body", "invalid UTF-8 JSON") from exc

    def route(self):
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise InputError("path", "expected a local request target")
        try:
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=100)
        except ValueError as exc:
            raise InputError("query", "invalid query string") from exc
        return parsed.path, query

    def single(self, query, key, *, default=None, required_value=False):
        values = query.get(key)
        if values is None:
            if required_value:
                raise InputError(key, "required query parameter missing")
            return default
        if len(values) != 1 or not values[0]:
            raise InputError(key, "expected one nonempty parameter")
        return values[0]

    def allowed_query(self, query, keys):
        unknown = set(query) - set(keys)
        if unknown:
            raise InputError("query", f"unknown parameters {sorted(unknown)}")

    def pressure_query(self, query):
        def many(key):
            values = query.get(key, [])
            if any(not value for value in values):
                raise InputError(key, "empty filter value")
            return tuple(values)
        try:
            years = tuple(int(value) for value in many("year"))
            minimum = int(self.single(query, "minimum_shortage", default="0"))
            limit = int(self.single(query, "limit", default="100"))
        except ValueError as exc:
            raise InputError("query", "numeric filter is not an integer") from exc
        return PressureQuery(
            years=years,
            regions=many("region"),
            settlements=many("settlement"),
            stages=many("stage"),
            statuses=many("status"),
            minimum_shortage=minimum,
            limit=limit,
        )

    def get(self, ledger, path, query):
        if path == "/health":
            self.allowed_query(query, ())
            return 200, {"status": "ok"}
        if path == "/integrity":
            self.allowed_query(query, ())
            return 200, ledger.verify()
        if path == "/batch":
            self.allowed_query(query, ("id",))
            return 200, ledger.get_batch(self.single(query, "id", required_value=True))
        if path == "/run":
            self.allowed_query(query, ("id",))
            return 200, ledger.get_run(self.single(query, "id", required_value=True))
        if path == "/history":
            self.allowed_query(query, ("entity", "after", "limit"))
            entity = self.single(query, "entity", required_value=True)
            try:
                after = int(self.single(query, "after", default="0"))
                limit = int(self.single(query, "limit", default="100"))
            except ValueError as exc:
                raise InputError("history", "pagination fields must be integers") from exc
            return 200, {"history": ledger.history(entity, after=after, limit=limit)}
        if path == "/release":
            self.allowed_query(query, ("label",))
            return 200, ledger.get_release(self.single(query, "label", required_value=True))
        if path in ("/pressure", "/totals", "/facets"):
            allowed = ("label", "year", "region", "settlement", "stage", "status", "minimum_shortage", "limit", "cursor", "group_by")
            self.allowed_query(query, allowed)
            release = ledger.get_release(self.single(query, "label", required_value=True))
            pressure_query = self.pressure_query(query)
            if path == "/pressure":
                return 200, pressure_page(release, pressure_query, cursor=self.single(query, "cursor"))
            if path == "/totals":
                return 200, {"totals": pressure_totals(release, pressure_query, group_by=tuple(query.get("group_by", ())))}
            return 200, release_facets(release, pressure_query)
        if path == "/summary":
            self.allowed_query(query, ("label",))
            release = ledger.get_release(self.single(query, "label", required_value=True))
            return 200, {"summary": scenario_summary(release["result"])}
        return 404, {"error": "route_not_found"}

    def post(self, ledger, path, query, body):
        self.allowed_query(query, ())
        if path == "/batches":
            batch = Batch.parse(body)
            batch_id, created = ledger.put_batch(batch.payload)
            return 201 if created else 200, {"batch_id": batch_id, "batch_digest": batch.fingerprint, "created": created}
        if path == "/runs":
            fields(body, {"batch_id", "scenario", "request_key"}, "run")
            batch_id = required(body, "batch_id")
            batch = Batch.parse(ledger.get_batch(batch_id)["payload"])
            scenario = required(body, "scenario")
            if scenario not in batch.scenarios:
                raise InputError("scenario", "unknown scenario")
            run_id, created = ledger.create_run(batch_id, scenario, required(body, "request_key"))
            return 201 if created else 200, {"run_id": run_id, "created": created}
        if path == "/execute":
            fields(body, {"run_id"}, "execute")
            run_id = required(body, "run_id")
            run = ledger.get_run(run_id)
            if run["state"] == "complete":
                return 200, {"run_id": run_id, "reused": True, "digest": run["result_digest"]}
            batch = Batch.parse(ledger.get_batch(run["batch_id"])["payload"])
            owner = "api-" + uuid.uuid4().hex
            revision = ledger.claim(run_id, owner, clock=int(time.time()), duration=300)
            try:
                result = batch.run(run["scenario"])
                quality = audit(batch, [result])
                if not quality["passed"]:
                    raise InputError("result", "quality checks found invalid output")
                result["quality"] = quality
            except (InputError, ArithmeticError) as exc:
                ledger.fail(run_id, owner, revision, str(exc), clock=int(time.time()))
                raise
            fingerprint = ledger.complete(run_id, owner, revision, result, clock=int(time.time()))
            return 200, {"run_id": run_id, "reused": False, "digest": fingerprint}
        if path == "/releases":
            fields(body, {"run_id", "label"}, "release")
            release, created = ledger.publish(required(body, "run_id"), required(body, "label"))
            return 201 if created else 200, {"release": release, "created": created}
        return 404, {"error": "route_not_found"}

    def dispatch(self, method):
        if not self.server.capacity.acquire(blocking=False):
            self.close_connection = True
            self.respond(503, {"error": "server_busy"})
            return
        try:
            path, query = self.route()
            body = self.read_body() if method == "POST" else None
            with Ledger(self.server.database) as ledger:
                if method == "GET":
                    status, payload = self.get(ledger, path, query)
                else:
                    status, payload = self.post(ledger, path, query, body)
            self.respond(status, payload)
        except InputError as exc:
            self.close_connection = True
            self.respond(422, {"error": "invalid_input", "field": exc.field, "message": str(exc)})
        except ConflictError as exc:
            self.close_connection = True
            self.respond(409, {"error": "state_conflict", "message": str(exc)})
        except (ConnectionError, TimeoutError):
            self.close_connection = True
        finally:
            self.server.capacity.release()

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")


def serve(database, *, host="127.0.0.1", port=8080):
    with ApiServer((host, port), database) as server:
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            return 0
