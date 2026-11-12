from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid

from cohorts.records import InputError
from cohorts.tabular import read_bundle, export_bundle
from forecast.backtest import ForecastPoint, evaluate, paired_comparison
from forecast.batches import Batch, read_json
from forecast.estimation import residual_diagnostics
from forecast.quality import audit
from releases.exports import atomic_text, export_release, verify_export
from releases.ledger import ConflictError, Ledger, canonical
from school_pressure.aggregation import aggregate, rural_urban_gaps, turning_points
from school_pressure.portfolio import compare_scenarios, envelope, rank_hotspots, scenario_summary


def execute(batch, code):
    result = batch.run(code)
    result["regional_pressure"] = aggregate(result["pressure"], batch.regions)
    result["urban_rural_gaps"] = rural_urban_gaps(result["regional_pressure"])
    result["trends"] = turning_points(result["pressure"])
    return result


def run_command(args):
    batch = Batch.parse(read_json(args.input))
    with Ledger(args.database) as ledger:
        batch_id, imported = ledger.put_batch(batch.payload)
        run_id, created = ledger.create_run(batch_id, args.scenario, args.request_key)
        current = ledger.get_run(run_id)
        if current["state"] == "complete":
            return {"run_id": run_id, "reused": True, "result_digest": current["result_digest"]}
        owner = "cli-" + uuid.uuid4().hex
        revision = ledger.claim(run_id, owner, clock=int(time.time()), duration=300)
        try:
            result = execute(batch, args.scenario)
        except (InputError, ArithmeticError) as exc:
            ledger.fail(run_id, owner, revision, str(exc), clock=int(time.time()))
            raise
        fingerprint = ledger.complete(run_id, owner, revision, result, clock=int(time.time()))
        return {"batch_id": batch_id, "run_id": run_id, "imported": imported, "created": created, "reused": False, "result_digest": fingerprint}


def demo_command(args):
    batch = Batch.parse(read_json(args.fixture))
    with tempfile.TemporaryDirectory(prefix="cohort-demo-") as directory:
        with Ledger(Path(directory) / "ledger.sqlite3") as ledger:
            batch_id, created = ledger.put_batch(batch.payload)
            if not created:
                raise ArithmeticError("fresh demo batch unexpectedly existed")
            results = {}
            for code in sorted(batch.scenarios):
                result = execute(batch, code)
                if any(row["balance"] != 0 for row in result["accounting"]):
                    raise ArithmeticError("demo population does not balance")
                run_id, _ = ledger.create_run(batch_id, code, "demo-" + code)
                revision = ledger.claim(run_id, "demo", clock=100, duration=100)
                fingerprint = ledger.complete(run_id, "demo", revision, result, clock=101)
                release, published = ledger.publish(run_id, "demo-" + code)
                if not published or ledger.get_release(release["label"])["digest"] != fingerprint:
                    raise ArithmeticError("demo release failed integrity check")
                results[code] = {
                    "through_year": result["through_year"],
                    "pressure_cells": len(result["pressure"]),
                    "shortage": sum(row["seat_shortage"] for row in result["pressure"]),
                    "result_digest": fingerprint,
                }
            integrity = ledger.verify()
            return {"scenarios": results, "integrity": integrity}


def parser():
    root = argparse.ArgumentParser(prog="cohort_cli")
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("input")
    run = commands.add_parser("run")
    run.add_argument("input")
    run.add_argument("--database", required=True)
    run.add_argument("--scenario", required=True)
    run.add_argument("--request-key", required=True)
    release = commands.add_parser("release")
    release.add_argument("--database", required=True)
    release.add_argument("--run-id", required=True)
    release.add_argument("--label", required=True)
    show = commands.add_parser("show")
    show.add_argument("--database", required=True)
    show.add_argument("--label", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--database", required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--fixture", required=True)
    bundle = commands.add_parser("bundle-import")
    bundle.add_argument("directory")
    bundle.add_argument("--output", required=True)
    unpack = commands.add_parser("bundle-export")
    unpack.add_argument("input")
    unpack.add_argument("--directory", required=True)
    quality = commands.add_parser("audit")
    quality.add_argument("input")
    report = commands.add_parser("portfolio")
    report.add_argument("input")
    report.add_argument("--baseline")
    report.add_argument("--limit", type=int, default=20)
    export = commands.add_parser("export")
    export.add_argument("--database", required=True)
    export.add_argument("--label", required=True)
    export.add_argument("--directory", required=True)
    exported = commands.add_parser("verify-export")
    exported.add_argument("directory")
    backtest = commands.add_parser("backtest")
    backtest.add_argument("input")
    backtest.add_argument("--observed-through", required=True, type=int)
    backtest.add_argument("--minimum-cells", type=int, default=1)
    backtest.add_argument("--left")
    backtest.add_argument("--right")
    server = commands.add_parser("serve")
    server.add_argument("--database", required=True)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8080)
    worker = commands.add_parser("work")
    worker.add_argument("--database", required=True)
    worker.add_argument("--retry-failed", action="store_true")
    worker.add_argument("--limit", type=int, default=100)
    queue = commands.add_parser("queue")
    queue.add_argument("--database", required=True)
    snapshot = commands.add_parser("backup")
    snapshot.add_argument("--database", required=True)
    snapshot.add_argument("--output", required=True)
    recover = commands.add_parser("restore")
    recover.add_argument("--source", required=True)
    recover.add_argument("--database", required=True)
    recover.add_argument("--sha256", required=True)
    return root


def analytics_command(args):
    if args.command == "bundle-import":
        payload, issues = read_bundle(args.directory)
        if issues:
            raise InputError("bundle", canonical({"row_issues": [issue.as_dict() for issue in issues]}))
        batch = Batch.parse(payload)
        atomic_text(args.output, canonical(batch.payload) + "\n")
        return {"output": args.output, "batch_digest": batch.fingerprint}
    if args.command == "bundle-export":
        batch = Batch.parse(read_json(args.input))
        filenames = export_bundle(batch.payload, args.directory)
        return {"directory": args.directory, "files": filenames}
    if args.command == "verify-export":
        return verify_export(args.directory)
    if args.command == "backtest":
        payload = read_json(args.input)
        points = []
        for row in payload["forecasts"]:
            points.append(ForecastPoint(
                row["issued_year"], row["target_year"], row["region"],
                row["settlement"], row["stage"], row["predicted"], row["model"],
            ))
        result = evaluate(points, payload["observations"], observed_through=args.observed_through, minimum_cells=args.minimum_cells)
        result["diagnostics"] = residual_diagnostics(result)
        if bool(args.left) != bool(args.right):
            raise InputError("models", "provide both --left and --right")
        if args.left:
            result["comparison"] = paired_comparison(result, args.left, args.right)
        return result
    batch = Batch.parse(read_json(args.input))
    results = batch.run_all()
    if args.command == "audit":
        return audit(batch, results.values())
    if args.baseline is not None and args.baseline not in results:
        raise InputError("baseline", "unknown baseline scenario")
    output = {
        "batch_digest": batch.fingerprint,
        "summary": [row for result in results.values() for row in scenario_summary(result)],
        "scenario_ranges": envelope(results.values()),
        "hotspots": {code: rank_hotspots(result, limit=args.limit) for code, result in results.items()},
    }
    if args.baseline:
        output["comparisons"] = [compare_scenarios(results[args.baseline], result) for code, result in results.items() if code != args.baseline]
    return output


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "serve":
            from releases.api import serve
            return serve(args.database, host=args.host, port=args.port)
        elif args.command == "work":
            from releases.worker import Worker
            result = Worker(args.database).drain(include_failed=args.retry_failed, limit=args.limit)
        elif args.command == "queue":
            from releases.worker import queue_status
            result = queue_status(args.database)
        elif args.command == "backup":
            from releases.backup import backup
            with Ledger(args.database) as ledger:
                result = backup(ledger, args.output)
        elif args.command == "restore":
            from releases.backup import restore
            result = restore(args.source, args.database, expected_sha256=args.sha256)
        elif args.command == "check":
            batch = Batch.parse(read_json(args.input))
            result = {"batch_digest": batch.fingerprint, "scenarios": sorted(batch.scenarios), "cells": len(batch.population)}
        elif args.command == "run":
            result = run_command(args)
        elif args.command == "demo":
            result = demo_command(args)
        elif args.command in ("bundle-import", "bundle-export", "audit", "portfolio", "verify-export", "backtest"):
            result = analytics_command(args)
        else:
            with Ledger(args.database) as ledger:
                if args.command == "release":
                    release, created = ledger.publish(args.run_id, args.label)
                    result = {"release": release, "created": created}
                elif args.command == "show":
                    result = ledger.get_release(args.label)
                elif args.command == "export":
                    manifest, created = export_release(ledger, args.label, args.directory)
                    result = {"manifest": manifest, "created": created}
                else:
                    result = ledger.verify()
        print(canonical(result))
        return 0
    except (InputError, ConflictError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
