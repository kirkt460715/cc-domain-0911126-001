"""联合预测研究的归档与重开校验。

研究以单个 JSON 文件原子写入，文件内容携带输入摘要、输入摘要、抽样参数、
种子、重采样索引、训练/校准/回测年份、排除理由与结果摘要。重开同一研究时，
以归档的输入摘要重新执行并比对，保证得到相同样本；输入发生变化（例如补到
新的实际值）时摘要不同，必须另建新研究。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cohorts.records import InputError, name
from forecast.batches import read_json, unique_object, reject_constant
from releases.exports import atomic_text, safe_component
from releases.ledger import canonical
from .engine import run_study


def _archive_digest(archive):
    return hashlib.sha256(canonical(archive).encode("utf-8")).hexdigest()


def save_study(payload, destination, *, overwrite=False):
    """执行研究并原子写入归档；同一路径上的研究不可变。"""
    archive = run_study(payload)
    destination = Path(destination)
    safe_component(destination.name)
    if destination.exists() and not overwrite:
        existing = load_study(destination)
        if existing["study"]["study_id"] != archive["study"]["study_id"]:
            raise InputError("archive", "path already holds a different study id")
        if existing["input_digest"] != archive["input_digest"]:
            raise InputError(
                "archive",
                "input changed (for example newly available actuals); create a new study instead of overwriting",
            )
        if existing["archive_digest"] != _archive_digest(archive):
            raise InputError("archive", "re-execution produced a different archive; refusing to overwrite")
        return {"path": str(destination), "study_id": archive["study"]["study_id"], "created": False,
                "archive_digest": existing["archive_digest"]}
    archive = dict(archive)
    archive["archive_digest"] = _archive_digest(archive)
    atomic_text(destination, canonical(archive) + "\n")
    return {"path": str(destination), "study_id": archive["study"]["study_id"], "created": True,
            "archive_digest": archive["archive_digest"]}


def load_study(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise InputError("archive", "study archive must be a regular file")
    with open(path, "rb") as source:
        raw = source.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024:
        raise InputError("archive", "study archive exceeds size limit")
    try:
        archive = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError("archive", f"invalid UTF-8 JSON: {exc}") from exc
    stored = archive.get("archive_digest")
    if not isinstance(stored, str):
        raise InputError("archive", "archive_digest missing")
    check = hashlib.sha256(canonical({key: value for key, value in archive.items() if key != "archive_digest"}).encode("utf-8")).hexdigest()
    if check != stored:
        raise InputError("archive", "archive integrity check failed")
    return archive


def reopen_study(archive, input_path):
    """用归档时的输入摘要校验当前输入，并重算以复现同一样本。

    输入一致且重算结果逐字段相符时返回复现报告；输入已变（含补到的新实际值）
    则报错并提示另建研究。
    """
    payload = read_json(input_path)
    recomputed = run_study(payload)
    if recomputed["input_digest"] != archive["input_digest"]:
        raise InputError(
            "input",
            "input digest differs from the archived study; updated actuals require a new study",
        )
    comparison = _compare(archive, recomputed)
    if comparison["mismatches"]:
        raise InputError("archive", f"re-execution did not reproduce the archived samples: {comparison['mismatches']}")
    return {
        "study_id": archive["study"]["study_id"],
        "input_digest": archive["input_digest"],
        "reproduced": True,
        "checked_horizons": comparison["checked_horizons"],
        "sample_digests": comparison["sample_digests"],
    }


def _compare(archive, recomputed):
    archived_by_key = {item["horizon_group"][0] if len(item["horizon_group"]) == 1 else tuple(item["horizon_group"]): item for item in archive["horizons"]}
    recomputed_by_key = {tuple(item["horizon_group"]): item for item in recomputed["horizons"]}
    mismatches = []
    sample_digests = {}
    for key, current in recomputed_by_key.items():
        stored = archived_by_key.get(key[0] if len(key) == 1 else key)
        if stored is None:
            mismatches.append({"horizon_group": list(key), "reason": "missing_in_archive"})
            continue
        if stored.get("draw_indices") != current.get("draw_indices"):
            mismatches.append({"horizon_group": list(key), "reason": "draw_indices differ"})
        for mode in current.get("modes", {}):
            old_digest = stored.get("modes", {}).get(mode, {}).get("sample_digest")
            new_digest = current["modes"][mode]["sample_digest"]
            sample_digests[f"{key}:{mode}"] = new_digest
            if old_digest != new_digest:
                mismatches.append({"horizon_group": list(key), "mode": mode, "reason": "sample digest differs"})
    return {"mismatches": mismatches, "checked_horizons": len(recomputed_by_key), "sample_digests": sample_digests}


def summarize_study(archive):
    """给出适合命令行浏览的精简视图。"""
    result = {
        "study_id": archive["study"]["study_id"],
        "input_digest": archive["input_digest"],
        "archive_digest": archive["archive_digest"],
        "input_summary": archive["input_summary"],
        "config": archive["study"],
        "horizons": [],
    }
    for horizon in archive["horizons"]:
        item = {
            "horizon_group": horizon["horizon_group"],
            "target_year": horizon["target_year"],
            "probability_output_available": horizon["probability_output_available"],
            "reason": horizon.get("reason"),
            "complete_vector_origins": horizon["complete_vector_origins"],
            "excluded_origins": horizon["excluded_origins"],
            "windows": horizon.get("windows"),
            "modes": {},
        }
        for mode, content in horizon.get("modes", {}).items():
            county_rows = [
                row for row in content["rows"]
                if row["region"] == archive["root_region"] and row["settlement"] == "all"
            ]
            item["modes"][mode] = {
                "reconciliation": content["reconciliation"],
                "clamp_events": content["clamp_events"],
                "county_by_stage": [
                    {
                        "stage": row["stage"],
                        "median_pupils": row["median_pupils"],
                        "intervals": row["intervals"],
                        "seats": row["seats"],
                        "exceedance_frequency": row["exceedance_frequency"],
                        "conditional_overload": row["conditional_overload"],
                        "leaf_concentration": row.get("leaf_concentration"),
                    }
                    for row in county_rows
                ],
            }
        if "diagnostics" in horizon:
            item["diagnostics"] = horizon["diagnostics"]
        result["horizons"].append(item)
    return result


def validate_study_id(value):
    return name(value, "study_id")
