"""联合预测研究的可复现演示。

为了让“同向 / 反向地区误差”产生可核对的不同全县风险，演示历史由确定性
伪随机序列合成：每个（地区、城乡、学段）单元拥有独立的冲击序列，其中城乡
冲击方向相反、东西两个叶子地区彼此不同；预测值在实际值附近做小幅偏移。
实际值的首次可知年份默认等于目标年份，只有 2026-2028 的修订要等到下一年
才可知，用于演示“未来才可知的修订不进入旧研究”。

边界情景：

* 未来才可知的修订：2026-2028 的实际值 first_known_year 在研究 as_of 之后，
  只能在更晚的研究中进入，重开旧研究仍得到相同样本。
* 容量为零：一个叶子单元在部分预测年份没有任何生效容量，正需求必然超载。
* 历史不足：把最少完整向量门槛调高，演示跨度停止概率输出或按说明合并。
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from forecast.batches import read_json
from studies.archive import load_study, save_study


STAGE = {"code": "primary", "min_age": 6, "max_age": 11, "class_size": 35, "max_class_size": 45}
REGIONS = ("east", "west")
SETTLEMENTS = ("urban", "rural")
STAGE_CODE = "primary"
FIRST_TARGET = 2011
LAST_TARGET = 2029
MAX_HORIZON = 3
AS_OF = 2025
LATE_AS_OF = 2028


def _cell_stream(seed_text):
    """确定性 LCG，保证同输入永远得到同序列，且无需第三方依赖。"""
    import hashlib
    state = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    while True:
        state = (1103515245 * state + 12345) % (2 ** 31)
        # 映射到 [-18, 18] 的整数冲击
        yield (state % 37) - 18


def synthesize_history(*, seed=20261113, lag_from=2028):
    """生成 forecasts / observations；冲击在单元间异质、城乡反向。"""
    base = {"east": 130, "west": 95}
    # 先取每个单元的潜在冲击序列；城乡冲击负相关但不完全抵消
    # （rural 取 urban 的反向 60%），两个叶子地区使用独立序列。
    cell_shocks = {}
    for region in REGIONS:
        for settlement in SETTLEMENTS:
            stream = _cell_stream(f"{seed}|{region}|{settlement}")
            values = []
            for target in range(FIRST_TARGET, LAST_TARGET + 1):
                raw = next(stream)
                if settlement == "urban":
                    values.append(raw)
                else:
                    values.append(-(abs(raw) * 6 // 10) * (1 if raw >= 0 else -1))
            cell_shocks[(region, settlement)] = values
    forecasts = []
    observations = []
    for index, target in enumerate(range(FIRST_TARGET, LAST_TARGET + 1)):
        for region in REGIONS:
            for settlement in SETTLEMENTS:
                shock = cell_shocks[(region, settlement)][index]
                actual = base[region] + shock
                # 持续性点预测：沿用上一目标年的潜在冲击，首年用基准值
                prior_shock = 0 if index == 0 else cell_shocks[(region, settlement)][index - 1]
                predicted = base[region] + prior_shock
                first_known = target + 1 if target >= lag_from else target
                observations.append({
                    "year": target, "region": region, "settlement": settlement,
                    "stage": STAGE_CODE, "pupils": actual, "first_known_year": first_known,
                })
                for horizon in range(1, MAX_HORIZON + 1):
                    issued = target - horizon
                    if issued < 2010:
                        continue
                    forecasts.append({
                        "issued_year": issued, "target_year": target,
                        "region": region, "settlement": settlement, "stage": STAGE_CODE,
                        "predicted": predicted, "model": "planning",
                    })
    forecasts.sort(key=lambda row: (row["issued_year"], row["target_year"], row["region"], row["settlement"]))
    return {"forecasts": forecasts, "observations": observations}


def _regions():
    return [
        {"code": "county", "label": "合成县"},
        {"code": "east", "label": "合成东区", "parent": "county"},
        {"code": "west", "label": "合成西区", "parent": "county"},
    ]


def _capacities(cells, years, *, seats, zero_capacity_cells=(), zero_years=()):
    result = []
    first_year, last_year = min(years), max(years)
    zero_cells = set(zero_capacity_cells)
    zero_years = tuple(sorted(zero_years))
    for region, settlement, stage in cells:
        cell_seats = seats.get((region, settlement, stage), seats) if isinstance(seats, dict) else seats
        facility = f"{region}-{settlement}-{stage}"
        if (region, settlement, stage) in zero_cells and zero_years:
            gap_start, gap_end = min(zero_years), max(zero_years)
            # 容量空档：空档前一段 + 空档后一段，空档内无任何生效容量
            if first_year < gap_start:
                result.append({
                    "facility": facility, "region": region, "settlement": settlement, "stage": stage,
                    "from_year": first_year, "through_year": gap_start - 1,
                    "seats": cell_seats, "classrooms": 100,
                })
            if gap_end < last_year:
                result.append({
                    "facility": facility + "-reopen", "region": region, "settlement": settlement, "stage": stage,
                    "from_year": gap_end + 1, "through_year": last_year,
                    "seats": cell_seats, "classrooms": 100,
                })
            continue
        result.append({
            "facility": facility, "region": region, "settlement": settlement, "stage": stage,
            "from_year": first_year, "through_year": last_year,
            "seats": cell_seats, "classrooms": 100,
        })
    return result


# 与各叶子单元的基准需求规模相称、并留出统一缓冲的固定容量；
# 使各叶子与全县的超越频率都落在 (0,1) 内，便于核对概率口径。
DEMO_SEATS = {
    ("east", "urban", STAGE_CODE): 133,
    ("east", "rural", STAGE_CODE): 132,
    ("west", "urban", STAGE_CODE): 105,
    ("west", "rural", STAGE_CODE): 94,
}


def build_demo_payload(history_path=None, *, study_id="joint-demo", draws=400, seed=20261113,
                       as_of_year=AS_OF, minimum_complete_vectors=3,
                       insufficient_history="stop", merge_horizons=(),
                       dependence_modes=("as_observed", "aligned", "opposed"),
                       seats=None, zero_capacity_cells=(), zero_years=(), use_shipped_history=False,
                       max_horizon=MAX_HORIZON):
    if seats is None:
        seats = dict(DEMO_SEATS)
    if use_shipped_history and history_path is not None:
        history = read_json(Path(history_path))
    else:
        history = synthesize_history(seed=seed)
    forecasts = [row for row in history["forecasts"] if row["target_year"] - row["issued_year"] <= max_horizon]
    history = {"forecasts": forecasts, "observations": history["observations"]}
    cells = sorted({
        (row["region"], row["settlement"], row["stage"]) for row in history["forecasts"]
    })
    target_years = sorted({row["target_year"] for row in history["forecasts"]})
    return {
        "study": {
            "study_id": study_id,
            "as_of_year": as_of_year,
            "model": "planning",
            "draws": draws,
            "seed": seed,
            "coverage_levels": ("0.5", "0.8", "0.9"),
            "dependence_modes": list(dependence_modes),
            "minimum_complete_vectors": minimum_complete_vectors,
            "insufficient_history": insufficient_history,
            "merge_horizons": [list(group) for group in merge_horizons],
        },
        "regions": _regions(),
        "stages": [STAGE],
        "capacities": _capacities(cells, target_years, seats=seats,
                                  zero_capacity_cells=zero_capacity_cells, zero_years=zero_years),
        "forecasts": history["forecasts"],
        "observations": history["observations"],
    }


def build_demo_variants(history_path=None):
    """返回（主研究, 变体字典）；变体覆盖需求中要求的可核对边界。"""
    main = build_demo_payload(history_path)
    variants = {
        # 同一历史在 2028 重开：2026-2027 修订此时可知，2028-2029 仍属未来修订；
        # 旧研究（as_of=2025）输入摘要保持不变，重开仍得到相同样本。
        "reopened_with_new_actuals": build_demo_payload(
            history_path, study_id="joint-demo-late", as_of_year=LATE_AS_OF, max_horizon=1,
        ),
        # 容量为零：west/rural 在 2027-2028 无任何生效容量
        "zero_capacity": build_demo_payload(
            history_path, study_id="joint-demo-zero-capacity",
            zero_capacity_cells=[("west", "rural", STAGE_CODE)], zero_years=(2027, 2028),
        ),
        # 历史不足：门槛高于任一跨度的完整向量数，停止概率输出
        "insufficient_stop": build_demo_payload(
            history_path, study_id="joint-demo-insufficient-stop",
            minimum_complete_vectors=99, dependence_modes=("as_observed",),
        ),
        # 同样的历史不足，但选择有说明的合并策略
        "insufficient_merge": build_demo_payload(
            history_path, study_id="joint-demo-insufficient-merge",
            minimum_complete_vectors=99, insufficient_history="merge_horizons",
            dependence_modes=("as_observed",),
        ),
    }
    return main, variants


def _county_risk_row(archive, horizon, mode):
    entry = next(item for item in archive["horizons"] if item["horizon_group"] == [horizon])
    row = next(
        row for row in entry["modes"][mode]["rows"]
        if row["region"] == archive["root_region"] and row["settlement"] == "all"
    )
    return {
        "mode": mode,
        "exceedance_frequency": row["exceedance_frequency"],
        "mean_shortage": row["mean_shortage"],
        "median_pupils": row["median_pupils"],
        "clamp_events": entry["modes"][mode]["clamp_events"],
        "reconciliation_mismatches": entry["modes"][mode]["reconciliation"]["mismatches"],
    }


def run_demo(history_path=None):
    """在临时目录内归档全部研究并返回可人工核对的演示摘要。"""
    main_payload, variants = build_demo_variants(history_path)
    report = {"studies": {}, "variants": {}}
    with tempfile.TemporaryDirectory(prefix="joint-study-demo-") as directory:
        main_path = Path(directory) / "main.json"
        saved = save_study(main_payload, main_path)
        archived = load_study(main_path)
        # 重开校验：同输入必须复现完全相同的样本摘要，且不重写文件
        reopened = save_study(main_payload, main_path)
        report["reopen"] = {"recreated": reopened["created"], "archive_digest": reopened["archive_digest"]}
        horizon = archived["horizons"][0]["horizon_group"][0]
        report["studies"]["main"] = {
            "archive_digest": saved["archive_digest"],
            "county_risk_by_mode": [
                _county_risk_row(archived, horizon, mode) for mode in main_payload["study"]["dependence_modes"]
            ],
            "excluded_origins": archived["horizons"][0]["excluded_origins"],
            "input_summary": archived["input_summary"],
            "backtest_diagnostics": archived["horizons"][0]["diagnostics"]["backtest_window"],
        }
        for name, payload in variants.items():
            output = Path(directory) / f"{name}.json"
            save_study(payload, output)
            study = load_study(output)
            report["variants"][name] = _variant_summary(study)
    return report


def _variant_summary(study):
    horizons = []
    for entry in study["horizons"]:
        item = {
            "horizon_group": entry["horizon_group"],
            "probability_output_available": entry["probability_output_available"],
            "reason": entry.get("reason"),
            "merged": entry["merged"],
            "merge_note": entry["merge_note"],
            "excluded_origins": entry["excluded_origins"],
        }
        if entry["probability_output_available"]:
            first_mode = next(iter(entry["modes"]))
            county = next(
                row for row in entry["modes"][first_mode]["rows"]
                if row["region"] == study["root_region"] and row["settlement"] == "all"
            )
            item["county"] = {
                "mode": first_mode,
                "exceedance_frequency": county["exceedance_frequency"],
                "conditional_overload": county["conditional_overload"],
                "leaf_concentration": county.get("leaf_concentration"),
            }
        horizons.append(item)
    return {"input_summary": study["input_summary"], "horizons": horizons}
