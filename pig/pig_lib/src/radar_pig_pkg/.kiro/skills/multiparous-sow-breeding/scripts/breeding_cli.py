#!/usr/bin/env python3
"""Standalone multiparous-sow breeding model CLI.

The script is intentionally self-contained: it does not import radar_pig_pkg and
never resolves files from the process working directory. Every external path
argument must be absolute. Only trusted pickle files may be loaded.

Per-pig data problems (missing label row, culled pig, broken dates, wrong row
count) are skipped and reported, matching the original Python pipeline's
tolerance while keeping every skip explicit and counted. Dataset-level problems
(missing label file, unknown experiment, no usable pig) fail loudly.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
LOGGER = logging.getLogger("multiparous-sow-breeding")
ROWS_PER_HOUR = 120
HOURS_PER_DAY = 24
ROWS_PER_DAY = ROWS_PER_HOUR * HOURS_PER_DAY
MIN_DAYS = 3
REQUIRED_COLUMNS = ("normalize_act", "new_br", "is_up")
LABEL_COLUMNS = (
    "experiment",
    "pig_id",
    "date",
    "label",
    "eva_label",
    "oneday_label",
    "towday_label",
)
FEATURE_COLUMNS = (
    "act_01", "act_02", "act_03", "act_04", "act_05", "act_06",
    "act_07", "act_08", "act_09", "act_10", "act_11", "act_12",
    "act_13", "act_14", "act_15", "act_16", "act_17", "act_18",
    "br_01", "br_02", "br_03", "br_04", "br_05", "br_06",
    "br_07", "br_08", "br_09", "br_10", "br_11", "br_12",
    "br_13", "br_14", "br_15", "br_16", "br_17", "br_18",
    "br_19", "br_20",
    "up_01", "up_02", "up_03", "up_04", "up_05", "up_06",
    "up_07", "up_08", "up_09", "up_10", "up_11", "up_12",
    "up_13", "up_14", "up_15", "up_16",
    "sleep_01", "sleep_02", "sleep_03", "sleep_04", "sleep_05",
    "sleep_06", "sleep_07", "sleep_08", "sleep_09",
)
INTERVAL_POSITIVE_DAYS = {
    "-2_0": {-2, -1, 0},
    "-1_0": {-1, 0},
    "-1_1": {-1, 0, 1},
}


class PipelineError(RuntimeError):
    """Fatal, actionable input or pipeline failure."""


class SkipPig(Exception):
    """Non-fatal per-pig problem; the pig is excluded and reported."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


@dataclass
class PigRecord:
    experiment: str
    pig_id: str
    radar_id: str
    dates: list[str]
    raw_data: pd.DataFrame
    oestrus_text: str
    breeding_text: str
    event_date: datetime | None


@dataclass
class FeatureReport:
    mode: str
    experiments: list[str]
    selected_pigs: int = 0
    usable_pigs: int = 0
    skipped_pigs: int = 0
    skipped_by_category: dict[str, int] = field(default_factory=dict)
    skipped_details: list[dict[str, str]] = field(default_factory=list)
    generated_samples: int = 0
    skipped_nonfinite_samples: int = 0
    warnings: list[str] = field(default_factory=list)


class JsonEncoder(json.JSONEncoder):
    """Encode NumPy, pandas, dataclass and Path values in result files."""

    def default(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            number = float(value)
            return number if np.isfinite(number) else None
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        return super().default(value)


def _json_dump(data: Any, path: Path | None = None) -> str:
    text = json.dumps(data, cls=JsonEncoder, ensure_ascii=False, indent=2)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return text


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(
            f"路径必须是绝对路径，不能依赖当前工作目录: {value}"
        )
    return path.resolve()


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise PipelineError(f"{description}不存在或不是文件: {path}")


def _require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise PipelineError(f"{description}不存在或不是目录: {path}")


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "experiment"


def _model_info_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_info.json")


def _load_dataset(feature_file: Path) -> list[Any]:
    _require_file(feature_file, "特征 pickle")
    LOGGER.warning("仅应加载可信来源的 pickle；pickle 反序列化可执行任意代码")
    try:
        with feature_file.open("rb") as stream:
            dataset = pickle.load(stream)
    except Exception as exc:
        raise PipelineError(f"无法加载特征 pickle {feature_file}: {exc}") from exc
    if not isinstance(dataset, (list, tuple)):
        raise PipelineError(
            "特征 pickle 顶层必须是 list/tuple，元素格式为 "
            "(experiment, pig_id, dates, DataFrame)"
        )
    return list(dataset)


def _load_label_tables(
    label_dir: Path, experiments: Sequence[str]
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    _require_directory(label_dir, "标签目录")
    tables: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for experiment in experiments:
        label_file = label_dir / f"{experiment}.csv"
        if not label_file.is_file():
            errors.append(f"缺少实验标签文件: {label_file}")
            continue
        try:
            table = pd.read_csv(label_file, header=0, dtype=str)
        except Exception as exc:
            errors.append(f"无法读取标签文件 {label_file}: {exc}")
            continue
        if table.shape[1] < 3:
            errors.append(
                f"标签文件至少需要 3 列（雷达 ID、发情日期、配种日期）: {label_file}"
            )
            continue
        tables[experiment] = table
    return tables, errors


def _parse_date(value: Any) -> datetime:
    return datetime.strptime(str(value).strip(), "%Y%m%d")


def _parse_event_date(value: str, context: str) -> datetime | None:
    text = value.strip()
    if text == "0":
        return None
    if len(text) < 8:
        raise SkipPig("label_malformed", f"{context} 标签值无法解析: {text!r}")
    try:
        return _parse_date(text[:8])
    except ValueError as exc:
        raise SkipPig("label_malformed", f"{context} 标签日期无效: {text!r} ({exc})")


def _prepare_pig(
    item: Any,
    index: int,
    label_tables: dict[str, pd.DataFrame],
    mode: str,
    warnings: list[str],
) -> PigRecord:
    """Validate one dataset item. Raises SkipPig or PipelineError."""
    if not isinstance(item, (list, tuple)) or len(item) != 4:
        raise PipelineError(
            f"dataset[{index}] 必须是长度为 4 的 tuple/list "
            "(experiment, pig_id, dates, DataFrame)"
        )
    experiment, pig_id, date_values, raw_data = item
    experiment = str(experiment)
    pig_id = str(pig_id)
    context = f"{experiment}/{pig_id}"

    parts = pig_id.split("_")
    if len(parts) < 2 or not parts[1].strip():
        raise SkipPig("pig_id_format", f"{context} pig_id 缺少雷达 ID 段")
    radar_id = parts[1].strip()

    if not isinstance(date_values, (list, tuple)):
        raise SkipPig("dates_type", f"{context} dates 不是 list/tuple")
    dates = [str(value).strip() for value in date_values]
    if len(dates) < MIN_DAYS:
        raise SkipPig("dates_too_few", f"{context} 只有 {len(dates)} 天，少于 {MIN_DAYS} 天")
    try:
        parsed_dates = [_parse_date(value) for value in dates]
    except ValueError as exc:
        raise SkipPig("dates_format", f"{context} 日期格式不是 YYYYMMDD ({exc})")
    if dates != sorted(dates):
        raise SkipPig("dates_unsorted", f"{context} 日期未升序排列")
    for previous, current in zip(parsed_dates, parsed_dates[1:]):
        if current - previous != timedelta(days=1):
            raise SkipPig(
                "dates_gap",
                f"{context} 日期不连续: {previous:%Y%m%d} -> {current:%Y%m%d}",
            )

    if not isinstance(raw_data, pd.DataFrame):
        raise SkipPig("raw_type", f"{context} 第 4 项不是 pandas.DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in raw_data.columns]
    if missing:
        raise SkipPig("raw_columns", f"{context} 缺少特征列 {missing}")
    expected_rows = len(dates) * ROWS_PER_DAY
    if len(raw_data) != expected_rows:
        raise SkipPig(
            "raw_rows",
            f"{context} 行数应为 {expected_rows}（{len(dates)}×2880），实际 {len(raw_data)}",
        )

    table = label_tables.get(experiment)
    if table is None:
        raise PipelineError(f"{context} 对应的标签表缺失")
    matches = table.iloc[:, 0].astype(str).str.strip() == radar_id
    match_count = int(matches.sum())
    if match_count == 0:
        raise SkipPig("label_missing", f"{context} 标签首列没有雷达 ID {radar_id}")
    if match_count > 1:
        warnings.append(f"{context} 标签中雷达 ID {radar_id} 重复 {match_count} 次，使用第一行")
    row = table.loc[matches].iloc[0].tolist()
    oestrus_text = str(row[1]).strip()
    breeding_text = str(row[2]).strip()

    if breeding_text == "-1":
        raise SkipPig("culled", f"{context} 配种标签为 -1（淘汰）")
    if mode == "test" and len(oestrus_text) >= 8 and breeding_text == "0":
        raise SkipPig("oestrus_not_bred", f"{context} 发情未配种，测试剔除")
    event_date = _parse_event_date(breeding_text, context)
    if event_date is not None and event_date <= parsed_dates[1]:
        raise SkipPig(
            "label_too_early",
            f"{context} 配种日 {event_date:%Y%m%d} 不晚于第二个数据日 {dates[1]}",
        )

    return PigRecord(
        experiment=experiment,
        pig_id=pig_id,
        radar_id=radar_id,
        dates=dates,
        raw_data=raw_data,
        oestrus_text=oestrus_text,
        breeding_text=breeding_text,
        event_date=event_date,
    )


def _iterate_pigs(
    dataset: Sequence[Any],
    label_tables: dict[str, pd.DataFrame],
    experiments: Sequence[str],
    mode: str,
    report: FeatureReport,
) -> list[PigRecord]:
    experiment_set = set(experiments)
    records: list[PigRecord] = []
    categories: Counter[str] = Counter()
    for index, item in enumerate(dataset):
        if isinstance(item, (list, tuple)) and len(item) == 4:
            if str(item[0]) not in experiment_set:
                continue
        report.selected_pigs += 1
        try:
            records.append(
                _prepare_pig(item, index, label_tables, mode, report.warnings)
            )
        except SkipPig as skip:
            categories[skip.category] += 1
            report.skipped_pigs += 1
            report.skipped_details.append(
                {"category": skip.category, "reason": skip.message}
            )
    report.usable_pigs = len(records)
    report.skipped_by_category = dict(sorted(categories.items()))
    return records


def inspect_inputs(
    dataset: Sequence[Any],
    label_dir: Path,
    experiments: Sequence[str],
    mode: str = "train",
) -> dict[str, Any]:
    """Inspect every selected pig and separate fatal errors from skips."""
    requested = list(dict.fromkeys(str(name) for name in experiments))
    if not requested:
        raise PipelineError("experiments 不能为空")

    label_tables, errors = _load_label_tables(label_dir, requested)
    present: Counter[str] = Counter()
    for item in dataset:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            present[str(item[0])] += 1
    for experiment in requested:
        if present[experiment] == 0:
            errors.append(f"特征 pickle 中没有请求的实验: {experiment}")

    report = FeatureReport(mode=mode, experiments=requested)
    usable_by_experiment: Counter[str] = Counter()
    if not errors:
        records = _iterate_pigs(dataset, label_tables, requested, mode, report)
        for record in records:
            usable_by_experiment[record.experiment] += 1
        for experiment in requested:
            if usable_by_experiment[experiment] == 0:
                errors.append(f"实验 {experiment} 没有可用猪（全部被跳过）")

    return {
        "valid": not errors,
        "mode": mode,
        "requested_experiments": requested,
        "dataset_items": len(dataset),
        "selected_pigs": report.selected_pigs,
        "usable_pigs": report.usable_pigs,
        "usable_pigs_by_experiment": dict(sorted(usable_by_experiment.items())),
        "present_experiments": dict(sorted(present.items())),
        "skipped_pigs": report.skipped_pigs,
        "skipped_by_category": report.skipped_by_category,
        "skipped_details": report.skipped_details,
        "warnings": report.warnings,
        "errors": errors,
    }


def _raise_for_invalid_inputs(report: dict[str, Any]) -> None:
    if report["errors"]:
        joined = "\n- ".join(report["errors"])
        raise PipelineError(f"输入预检失败:\n- {joined}")
    if report["skipped_pigs"]:
        LOGGER.warning(
            "跳过 %s 头猪: %s",
            report["skipped_pigs"],
            report["skipped_by_category"],
        )
    for warning in report["warnings"]:
        LOGGER.warning(warning)


def _count_continuous_ones(values: Sequence[float]) -> int:
    count = 0
    in_segment = False
    for value in values:
        if value == 1:
            if not in_segment:
                count += 1
                in_segment = True
        else:
            in_segment = False
    return count


def _kl_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    epsilon = 1e-10
    p = np.asarray(left, dtype=np.float32) + epsilon
    q = np.asarray(right, dtype=np.float32) + epsilon
    return float(np.sum(p * np.log(p / q)))


def _compute_activity_features(
    act_raw: np.ndarray,
    cur_morning: range,
    cur_evening: range,
    day1_morning: range,
    day1_evening: range,
    day2_morning: range,
    day2_evening: range,
) -> np.ndarray:
    current_morning = act_raw[cur_morning, :]
    previous_morning = act_raw[day1_morning, :]
    previous_evening = act_raw[day1_evening, :]
    two_days_morning = act_raw[day2_morning, :]
    two_days_evening = act_raw[day2_evening, :]
    values = [
        np.nansum(current_morning) - np.nansum(previous_morning),
        np.nansum(current_morning) - np.nansum(two_days_morning),
        np.nansum(previous_evening) - np.nansum(two_days_evening),
        np.sum(current_morning > 4) - np.sum(previous_morning > 4),
        np.sum(current_morning > 4) - np.sum(two_days_morning > 4),
        np.sum(previous_evening > 4) - np.sum(two_days_evening > 4),
        np.sum(current_morning > 15) - np.sum(previous_morning > 15),
        np.sum(current_morning > 15) - np.sum(two_days_morning > 15),
        np.sum(previous_evening > 15) - np.sum(two_days_evening > 15),
        np.nansum(current_morning) / (np.nansum(previous_morning) + 0.1),
        np.nansum(current_morning) / (np.nansum(two_days_morning) + 0.1),
        np.nansum(previous_evening) / (np.nansum(two_days_evening) + 0.1),
        np.sum(current_morning > 4) / (np.sum(previous_morning > 4) + 0.1),
        np.sum(current_morning > 4) / (np.sum(two_days_morning > 4) + 0.1),
        np.sum(previous_evening > 4) / (np.sum(two_days_evening > 4) + 0.1),
        np.sum(current_morning > 15) / (np.sum(previous_morning > 15) + 0.1),
        np.sum(current_morning > 15) / (np.sum(two_days_morning > 15) + 0.1),
        np.sum(previous_evening > 15) / (np.sum(two_days_evening > 15) + 0.1),
    ]
    _ = cur_evening
    return np.asarray(values, dtype=float)


def _compute_breathing_features(
    br_raw: np.ndarray,
    cur_morning: range,
    cur_evening: range,
    day1_morning: range,
    day1_evening: range,
    day2_morning: range,
    day2_evening: range,
) -> np.ndarray:
    current_morning = br_raw[cur_morning, :]
    previous_morning = br_raw[day1_morning, :]
    previous_evening = br_raw[day1_evening, :]
    two_days_morning = br_raw[day2_morning, :]
    two_days_evening = br_raw[day2_evening, :]

    median_current = np.nanmedian(current_morning)
    median_previous_evening = np.nanmedian(previous_evening)
    median_previous_morning = np.nanmedian(previous_morning)
    median_two_days_evening = np.nanmedian(two_days_evening)
    median_two_days_morning = np.nanmedian(two_days_morning)
    values = [
        median_current,
        median_previous_evening,
        median_previous_morning,
        median_two_days_evening,
        median_two_days_morning,
        median_current - median_previous_morning,
        median_current - median_two_days_morning,
        median_previous_morning - median_two_days_morning,
        median_previous_evening - median_two_days_evening,
        median_current / (median_previous_morning + 0.01),
        median_current / (median_two_days_morning + 0.01),
        median_previous_morning / (median_two_days_morning + 0.01),
        median_previous_evening / (median_two_days_evening + 0.01),
        (median_current + median_previous_evening)
        / (median_previous_morning + median_two_days_evening + 0.01),
        np.nanpercentile(current_morning, 75)
        / (np.nanpercentile(current_morning, 25) + 0.01),
        np.nanpercentile(previous_morning, 75)
        / (np.nanpercentile(previous_morning, 25) + 0.01),
        np.nanpercentile(two_days_morning, 75)
        / (np.nanpercentile(two_days_morning, 25) + 0.01),
        np.nanpercentile(previous_evening, 75)
        / (np.nanpercentile(previous_evening, 25) + 0.01),
        np.sum(current_morning > 40),
        np.sum(previous_morning > 40),
    ]
    _ = cur_evening
    return np.asarray(values, dtype=float)


def _compute_movement_features(
    up_raw: np.ndarray,
    cur_morning: range,
    cur_evening: range,
    day1_morning: range,
    day1_evening: range,
    day2_morning: range,
    day2_evening: range,
) -> np.ndarray:
    current_morning = up_raw[cur_morning, :]
    previous_morning = up_raw[day1_morning, :]
    previous_evening = up_raw[day1_evening, :]
    two_days_morning = up_raw[day2_morning, :]
    two_days_evening = up_raw[day2_evening, :]
    standing_values = [
        np.nansum(current_morning),
        np.nansum(current_morning) - np.nansum(previous_morning),
        np.nansum(current_morning) - np.nansum(two_days_morning),
        np.nansum(previous_evening) - np.nansum(two_days_evening),
    ]

    # Preserve np.diff's default axis from the original Python source so the
    # 63-dimensional feature values stay compatible with existing models.
    up_diff = np.diff(up_raw)
    diff_current = up_diff[cur_morning, :]
    diff_previous = up_diff[day1_morning, :]
    diff_previous_evening = up_diff[day1_evening, :]
    diff_two_days = up_diff[day2_morning, :]
    diff_two_days_evening = up_diff[day2_evening, :]

    transitions_current = _count_continuous_ones(diff_current.flatten().tolist())
    transitions_previous = _count_continuous_ones(diff_previous.flatten().tolist())
    transition_ratio = transitions_current / (transitions_previous + 0.1)
    transition_delta = transitions_current - transitions_previous
    rule_flag = (
        (transitions_previous == 0 and transition_ratio >= 30)
        or (transitions_previous == 1 and transition_ratio >= 40)
        or (transitions_previous == 2 and transition_ratio >= 20)
        or (transitions_previous == 3 and transition_ratio >= 15)
        or (transitions_previous == 4 and transition_ratio >= 10)
        or (transitions_previous == 5 and transition_ratio >= 8)
        or (transitions_previous == 6 and transition_ratio >= 6)
        or (10 >= transitions_previous >= 7 and transition_ratio >= 4)
        or (transitions_previous > 10 and transition_ratio >= 2.5)
        or (transitions_previous > 25 and transition_ratio >= 2)
        or (transitions_previous > 35 and transition_ratio >= 1.5)
        or (transitions_previous > 50 and transition_delta >= 25)
    )
    # The original source derives this from a zero array, so it is always 0.
    transitions_two_days = _count_continuous_ones(
        np.zeros_like(diff_previous).flatten().tolist()
    )
    values = standing_values + [
        np.sum(diff_current != 0),
        np.sum(diff_current != 0) / (np.sum(diff_previous != 0) + 0.1),
        np.sum(diff_current != 0) / (np.sum(diff_two_days != 0) + 0.1),
        np.sum(diff_previous_evening != 0) / (np.sum(diff_two_days_evening != 0) + 0.1),
        transitions_current,
        transitions_previous,
        transitions_current >= 30 and transitions_previous == 0,
        transition_ratio,
        transition_delta,
        rule_flag,
        transitions_two_days,
        transitions_current >= transitions_previous >= transitions_two_days
        and transitions_current != 0,
    ]
    _ = cur_evening
    return np.asarray(values, dtype=float)


def _sleep_distribution(values: Sequence[float]) -> list[float]:
    if not values:
        raise SkipPig("sleep_window", "睡眠特征窗口为空")
    result = [0, 0, 0, 0]
    index = 0
    while index < len(values):
        duration = 0
        while index < len(values) and values[index] != 0:
            index += 1
        while index < len(values) and values[index] != 1:
            index += 1
            duration += 1
        if duration >= 120:
            result[3] += duration
        elif duration >= 60:
            result[2] += duration
        elif duration >= 20:
            result[1] += duration
        elif duration > 0:
            result[0] += duration
    result.append(len(values) - sum(result))
    return [value / len(values) for value in result]


def _compute_sleep_features(
    up_raw: np.ndarray,
    day1_hour_index: int,
    current_hour_index: int,
    day2_hour_index: int,
) -> np.ndarray:
    current_values = up_raw[
        day1_hour_index + 20 : current_hour_index + 7
    ].flatten().tolist()
    previous_values = up_raw[
        day2_hour_index + 20 : day1_hour_index + 7
    ].flatten().tolist()
    current = _sleep_distribution(current_values)
    previous = _sleep_distribution(previous_values)
    return np.asarray(
        [
            current[0] - previous[0],
            current[1] - previous[1],
            current[2] - previous[2],
            current[3] - previous[3],
            _kl_divergence(current, previous),
            current[0],
            current[1],
            current[2],
            current[3],
        ],
        dtype=float,
    )


def _compute_feature_set(
    act_raw: np.ndarray,
    br_raw: np.ndarray,
    up_raw: np.ndarray,
    current_day_index: int,
) -> np.ndarray:
    current_hour = current_day_index * HOURS_PER_DAY
    day1_hour = (current_day_index - 1) * HOURS_PER_DAY
    day2_hour = (current_day_index - 2) * HOURS_PER_DAY
    selections = (
        range(current_hour, current_hour + 7),
        range(current_hour + 20, current_hour + 24),
        range(day1_hour, day1_hour + 7),
        range(day1_hour + 20, day1_hour + 24),
        range(day2_hour, day2_hour + 7),
        range(day2_hour + 20, day2_hour + 24),
    )
    features = np.concatenate(
        [
            _compute_activity_features(act_raw, *selections),
            _compute_breathing_features(br_raw, *selections),
            _compute_movement_features(up_raw, *selections),
            _compute_sleep_features(up_raw, day1_hour, current_hour, day2_hour),
        ]
    )
    if features.shape != (len(FEATURE_COLUMNS),):
        raise PipelineError(
            f"内部特征维度错误: 期望 {len(FEATURE_COLUMNS)}，实际 {features.shape}"
        )
    return features


def _generate_label_info(
    interval_type: str,
    record: PigRecord,
    date_text: str,
    current_date: datetime,
) -> list[Any]:
    positive_days = INTERVAL_POSITIVE_DAYS[interval_type]
    if record.event_date is None:
        delta_days: int | None = None
    else:
        delta_days = (current_date - record.event_date).days
    return [
        record.experiment,
        record.pig_id,
        date_text,
        int(delta_days == 0),
        int(delta_days in positive_days),
        int(delta_days == -1),
        int(delta_days == -2),
    ]


def _hourly_matrix(raw_data: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(raw_data[column], errors="coerce").to_numpy(dtype=float)
    return values.reshape(-1, ROWS_PER_HOUR)


def generate_features(
    dataset: Sequence[Any],
    label_dir: Path,
    experiments: Sequence[str],
    mode: str,
    interval_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureReport]:
    """Generate the source-compatible default 63-dimensional feature set."""
    if mode not in {"train", "test"}:
        raise PipelineError(f"mode 必须是 train 或 test，实际为 {mode}")
    if interval_type not in INTERVAL_POSITIVE_DAYS:
        raise PipelineError(f"不支持的 interval_type: {interval_type}")

    requested = list(dict.fromkeys(str(name) for name in experiments))
    label_tables, errors = _load_label_tables(label_dir, requested)
    if errors:
        raise PipelineError("; ".join(errors))

    report = FeatureReport(mode=mode, experiments=requested)
    records = _iterate_pigs(dataset, label_tables, requested, mode, report)
    if not records:
        raise PipelineError(
            f"实验 {requested} 没有可用猪；跳过原因: {report.skipped_by_category}"
        )

    all_features: list[np.ndarray] = []
    all_labels: list[list[Any]] = []
    extra_skips: Counter[str] = Counter()
    for record in records:
        try:
            act_raw = _hourly_matrix(record.raw_data, "normalize_act")
            br_raw = _hourly_matrix(record.raw_data, "new_br")
            up_raw = _hourly_matrix(record.raw_data, "is_up")
        except ValueError as exc:
            extra_skips["reshape"] += 1
            report.skipped_details.append(
                {
                    "category": "reshape",
                    "reason": f"{record.experiment}/{record.pig_id} 无法按小时重塑: {exc}",
                }
            )
            continue

        pig_features: list[np.ndarray] = []
        pig_labels: list[list[Any]] = []
        try:
            for day_index in range(MIN_DAYS - 1, len(record.dates)):
                features = _compute_feature_set(act_raw, br_raw, up_raw, day_index)
                if mode == "train" and not np.isfinite(features).all():
                    report.skipped_nonfinite_samples += 1
                    continue
                pig_features.append(features)
                pig_labels.append(
                    _generate_label_info(
                        interval_type,
                        record,
                        record.dates[day_index],
                        _parse_date(record.dates[day_index]),
                    )
                )
        except SkipPig as skip:
            extra_skips[skip.category] += 1
            report.skipped_details.append(
                {"category": skip.category, "reason": skip.message}
            )
            continue
        except IndexError as exc:
            extra_skips["feature_window"] += 1
            report.skipped_details.append(
                {
                    "category": "feature_window",
                    "reason": f"{record.experiment}/{record.pig_id} 特征窗口越界: {exc}",
                }
            )
            continue
        all_features.extend(pig_features)
        all_labels.extend(pig_labels)

    if extra_skips:
        merged = Counter(report.skipped_by_category)
        merged.update(extra_skips)
        report.skipped_by_category = dict(sorted(merged.items()))
        report.skipped_pigs += sum(extra_skips.values())
        report.usable_pigs -= sum(extra_skips.values())

    features_df = pd.DataFrame(all_features, columns=list(FEATURE_COLUMNS), dtype=float)
    labels_df = pd.DataFrame(all_labels, columns=list(LABEL_COLUMNS))
    report.generated_samples = len(features_df)
    if len(features_df) != len(labels_df):
        raise PipelineError("内部错误：特征与标签样本数不一致")
    if features_df.empty:
        raise PipelineError(
            f"实验 {requested} 没有生成任何样本；跳过原因: {report.skipped_by_category}"
        )
    return features_df, labels_df, report


def _label_distribution(labels: pd.Series) -> dict[str, int]:
    counts = labels.astype(int).value_counts().to_dict()
    return {str(label): int(counts.get(label, 0)) for label in (0, 1)}


def train_pipeline(
    feature_file: Path,
    label_dir: Path,
    experiments: Sequence[str],
    model_output: Path,
    interval_type: str,
    test_size: float,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    nthread: int,
    verbose_eval: int,
    stratify: bool,
) -> dict[str, Any]:
    if not 0 < test_size < 1:
        raise PipelineError("test_size 必须在 0 与 1 之间")
    dataset = _load_dataset(feature_file)
    inspection = inspect_inputs(dataset, label_dir, experiments, mode="train")
    _raise_for_invalid_inputs(inspection)
    features_df, labels_df, feature_report = generate_features(
        dataset, label_dir, experiments, "train", interval_type
    )
    y = labels_df["eva_label"].astype(int)
    if y.nunique() < 2:
        raise PipelineError(
            f"训练标签必须同时包含 0 和 1，实际 {_label_distribution(y)}"
        )

    try:
        x_train, x_val, y_train, y_val = train_test_split(
            features_df,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y if stratify else None,
        )
    except ValueError as exc:
        raise PipelineError(f"训练/验证集拆分失败: {exc}") from exc
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise PipelineError(
            "训练集和验证集都必须包含正负标签；请增加数据或使用 --stratify"
        )

    feature_names = list(FEATURE_COLUMNS)
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=feature_names)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.015,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "nthread": nthread,
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "eval")],
        verbose_eval=False if verbose_eval <= 0 else verbose_eval,
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
    )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_output))
    reloaded = xgb.Booster()
    reloaded.load_model(str(model_output))

    info = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(SCRIPT_PATH),
        "feature_file": str(feature_file),
        "label_dir": str(label_dir),
        "experiments": list(experiments),
        "multiparous_scope_confirmed": True,
        "label_type": "breeding",
        "interval_type": interval_type,
        "parameters": params,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "test_size": test_size,
        "stratified_split": stratify,
        "best_score": float(model.best_score),
        "best_iteration": int(model.best_iteration),
        "feature_dim": len(FEATURE_COLUMNS),
        "feature_columns": feature_names,
        "all_samples": len(features_df),
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "all_label_distribution": _label_distribution(y),
        "train_label_distribution": _label_distribution(y_train),
        "val_label_distribution": _label_distribution(y_val),
        "input_inspection": inspection,
        "feature_report": asdict(feature_report),
        "model_path": str(model_output),
    }
    info_path = _model_info_path(model_output)
    info["info_path"] = str(info_path)
    _json_dump(info, info_path)
    return info


def _calculate_sign(prediction: np.ndarray, label: np.ndarray) -> str:
    predicted = np.flatnonzero(prediction == 1)
    actual = np.flatnonzero(label == 1)
    pred_day = int(predicted[0]) if len(predicted) else -1
    true_day = int(actual[-1]) if len(actual) else -1
    if true_day == -1 and pred_day == -1:
        return "未配种且未预测"
    if true_day == -1 and pred_day != -1:
        return "未配种但预测"
    if true_day != -1 and pred_day != -1:
        return str(pred_day - true_day)
    return "未预测"


def _sample_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probability: np.ndarray
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = (
        float(roc_auc_score(y_true, y_probability))
        if len(np.unique(y_true)) == 2
        else None
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "confusion_matrix": [[int(tp), int(fp)], [int(fn), int(tn)]],
    }


def _pig_metrics(
    labels_df: pd.DataFrame, y_pred: np.ndarray, y_probability: np.ndarray
) -> dict[str, Any]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    pig_ids: dict[str, list[str]] = {"tp": [], "fp": [], "fn": [], "tn": []}
    sign_count: Counter[str] = Counter()
    pig_results: list[dict[str, Any]] = []

    for pig_id in labels_df["pig_id"].unique():
        mask = (labels_df["pig_id"] == pig_id).to_numpy()
        event_label = labels_df.loc[mask, "label"].to_numpy(dtype=int)
        evaluation_label = labels_df.loc[mask, "eva_label"].to_numpy(dtype=int)
        prediction = y_pred[mask].astype(int, copy=True)
        probability = y_probability[mask].astype(float, copy=True)
        sign = _calculate_sign(prediction, event_label)
        sign_count[sign] += 1
        pig_results.append(
            {
                "pig_id": str(pig_id),
                "date": labels_df.loc[mask, "date"].astype(str).tolist(),
                "label": event_label.tolist(),
                "eva_label": evaluation_label.tolist(),
                "pred": prediction.tolist(),
                "pred_prob": [round(float(value), 6) for value in probability],
                "sign": sign,
            }
        )

        positive_indices = np.flatnonzero(prediction == 1)
        first_prediction = np.zeros_like(prediction)
        if len(positive_indices):
            first_prediction[positive_indices[0]] = 1
        if event_label.sum() > 0:
            category = "tp" if np.any(first_prediction & evaluation_label) else "fn"
        else:
            category = "fp" if first_prediction.sum() > 0 else "tn"
        counts[category] += 1
        pig_ids[category].append(str(pig_id))

    positive_total = counts["tp"] + counts["fn"]
    predicted_total = counts["tp"] + counts["fp"]
    all_total = sum(counts.values())
    return {
        "accuracy": (counts["tp"] + counts["tn"]) / all_total if all_total else 0.0,
        "precision": counts["tp"] / predicted_total if predicted_total else 0.0,
        "recall": counts["tp"] / positive_total if positive_total else 0.0,
        "confusion_matrix": [
            [counts["tp"], counts["fp"]],
            [counts["fn"], counts["tn"]],
        ],
        "confusion_matrix_pigs": [
            pig_ids["tp"],
            pig_ids["fp"],
            pig_ids["fn"],
            pig_ids["tn"],
        ],
        "sign_count": dict(sign_count),
        "pig_results": pig_results,
    }


def test_pipeline(
    feature_file: Path,
    label_dir: Path,
    experiments: Sequence[str],
    model_path: Path,
    output_dir: Path,
    interval_type: str,
    threshold: float,
) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise PipelineError("threshold 必须在 0 与 1 之间")
    _require_file(model_path, "模型文件")
    dataset = _load_dataset(feature_file)
    inspection = inspect_inputs(dataset, label_dir, experiments, mode="test")
    _raise_for_invalid_inputs(inspection)

    model = xgb.Booster()
    try:
        model.load_model(str(model_path))
    except Exception as exc:
        raise PipelineError(f"无法加载模型 {model_path}: {exc}") from exc

    info_path = _model_info_path(model_path)
    if info_path.is_file():
        try:
            model_info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PipelineError(f"无法解析模型信息 {info_path}: {exc}") from exc
        expected_columns = model_info.get("feature_columns")
        if expected_columns and list(expected_columns) != list(FEATURE_COLUMNS):
            raise PipelineError("模型信息中的特征列与内置 63 维特征不一致")
        expected_interval = model_info.get("interval_type")
        if expected_interval and expected_interval != interval_type:
            raise PipelineError(
                f"模型使用 interval_type={expected_interval}，测试请求为 {interval_type}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(FEATURE_COLUMNS)
    experiment_results: list[dict[str, Any]] = []
    for experiment in dict.fromkeys(str(name) for name in experiments):
        features_df, labels_df, feature_report = generate_features(
            dataset, label_dir, [experiment], "test", interval_type
        )
        matrix = features_df.to_numpy(dtype=float)
        probability = np.asarray(
            model.predict(xgb.DMatrix(features_df, feature_names=feature_names)),
            dtype=float,
        )
        prediction = (probability > threshold).astype(int)
        nonfinite_rows = ~np.isfinite(matrix).all(axis=1)
        prediction[nonfinite_rows] = 0
        probability[nonfinite_rows] = 0.0

        y_true = labels_df["eva_label"].to_numpy(dtype=int)
        sample_metrics = _sample_metrics(y_true, prediction, probability)
        pig_metrics = _pig_metrics(labels_df, prediction, probability)

        experiment_dir = output_dir / _safe_filename(experiment)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        predictions = labels_df.copy()
        predictions["pred"] = prediction
        predictions["pred_probability"] = probability
        predictions["forced_negative_for_nonfinite"] = nonfinite_rows
        predictions_path = experiment_dir / "sample_predictions.csv"
        predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        pigs_path = experiment_dir / "pig_results.json"
        _json_dump(pig_metrics["pig_results"], pigs_path)
        metrics_path = experiment_dir / "metrics.json"

        result = {
            "experiment": experiment,
            "samples": len(features_df),
            "pigs": int(labels_df["pig_id"].nunique()),
            "forced_negative_samples": int(nonfinite_rows.sum()),
            "threshold": threshold,
            "sample_label_distribution": _label_distribution(labels_df["eva_label"]),
            "sample_metrics": sample_metrics,
            "pig_metrics": {
                key: value for key, value in pig_metrics.items() if key != "pig_results"
            },
            "feature_report": asdict(feature_report),
            "predictions_path": str(predictions_path),
            "pig_results_path": str(pigs_path),
            "metrics_path": str(metrics_path),
        }
        _json_dump(result, metrics_path)
        experiment_results.append(result)

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(SCRIPT_PATH),
        "feature_file": str(feature_file),
        "label_dir": str(label_dir),
        "model_path": str(model_path),
        "model_info_path": str(info_path) if info_path.is_file() else None,
        "output_dir": str(output_dir),
        "experiments": list(experiments),
        "multiparous_scope_confirmed": True,
        "label_type": "breeding",
        "interval_type": interval_type,
        "threshold": threshold,
        "input_inspection": inspection,
        "results": experiment_results,
    }
    summary_path = output_dir / "test_summary.json"
    summary["summary_path"] = str(summary_path)
    _json_dump(summary, summary_path)
    return summary


def _create_synthetic_inputs(base_dir: Path) -> tuple[Path, Path]:
    label_dir = base_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    dataset: list[tuple[str, str, list[str], pd.DataFrame]] = []
    start = datetime(2024, 1, 1)
    dates = [(start + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(6)]

    for experiment_index, experiment in enumerate(("synthetic_train", "synthetic_test")):
        label_rows: list[dict[str, str]] = []
        for pig_index in range(8):
            radar_id = str(1000 + experiment_index * 100 + pig_index)
            pig_id = f"pig_{radar_id}"
            positive = pig_index % 2 == 0
            event_index = 4
            rng = np.random.default_rng(2024 + experiment_index * 100 + pig_index)
            row_count = len(dates) * ROWS_PER_DAY
            activity = np.maximum(rng.normal(2.0, 0.8, row_count), 0.0)
            breathing = rng.normal(22.0, 1.5, row_count)
            standing = (rng.random(row_count) > 0.72).astype(float)
            if positive:
                for day_index in (2, 3, 4):
                    start_row = day_index * ROWS_PER_DAY
                    end_row = (day_index + 1) * ROWS_PER_DAY
                    activity[start_row:end_row] += (day_index - 1) * 1.5
                    standing[start_row:end_row] = (
                        rng.random(ROWS_PER_DAY) > (0.72 - 0.08 * (day_index - 1))
                    ).astype(float)
            dataset.append(
                (
                    experiment,
                    pig_id,
                    list(dates),
                    pd.DataFrame(
                        {
                            "normalize_act": activity.astype(np.float32),
                            "new_br": breathing.astype(np.float32),
                            "is_up": standing.astype(np.float32),
                        }
                    ),
                )
            )
            label_rows.append(
                {
                    "radar_id": radar_id,
                    "oestrus_date": dates[event_index - 2] if positive else "0",
                    "breeding_date": dates[event_index] if positive else "0",
                }
            )
        # A pig that exists only in the label file, plus one dataset pig without a
        # label row, so self-check also covers the real-world skip paths.
        label_rows.append({"radar_id": "9999", "oestrus_date": "0", "breeding_date": "0"})
        dataset.append(
            (
                experiment,
                f"pig_unlabeled{experiment_index}",
                list(dates),
                pd.DataFrame(
                    {
                        "normalize_act": np.zeros(len(dates) * ROWS_PER_DAY, dtype=np.float32),
                        "new_br": np.full(len(dates) * ROWS_PER_DAY, 20.0, dtype=np.float32),
                        "is_up": np.zeros(len(dates) * ROWS_PER_DAY, dtype=np.float32),
                    }
                ),
            )
        )
        pd.DataFrame(label_rows).to_csv(
            label_dir / f"{experiment}.csv", index=False, encoding="utf-8"
        )

    feature_file = base_dir / "synthetic_features.pkl"
    with feature_file.open("wb") as stream:
        pickle.dump(dataset, stream)
    return feature_file, label_dir


def self_check(output_dir: Path | None) -> dict[str, Any]:
    def run(base_dir: Path) -> dict[str, Any]:
        feature_file, label_dir = _create_synthetic_inputs(base_dir)
        model_path = base_dir / "model" / "breeding.model"
        train_info = train_pipeline(
            feature_file=feature_file,
            label_dir=label_dir,
            experiments=["synthetic_train"],
            model_output=model_path,
            interval_type="-2_0",
            test_size=0.25,
            seed=42,
            num_boost_round=20,
            early_stopping_rounds=5,
            nthread=1,
            verbose_eval=0,
            stratify=True,
        )
        test_info = test_pipeline(
            feature_file=feature_file,
            label_dir=label_dir,
            experiments=["synthetic_test"],
            model_path=model_path,
            output_dir=base_dir / "test-results",
            interval_type="-2_0",
            threshold=0.52,
        )
        if train_info["feature_dim"] != len(FEATURE_COLUMNS):
            raise PipelineError("self-check 训练特征维度不是 63")
        if not test_info["results"]:
            raise PipelineError("self-check 未生成测试结果")
        skipped = train_info["feature_report"]["skipped_by_category"]
        if "label_missing" not in skipped:
            raise PipelineError("self-check 未覆盖缺少标签行的跳过路径")
        return {
            "status": "PASS",
            "script": str(SCRIPT_PATH),
            "working_directory_independent": True,
            "artifacts": str(base_dir),
            "train_samples": train_info["all_samples"],
            "test_samples": test_info["results"][0]["samples"],
            "feature_dim": train_info["feature_dim"],
            "skip_paths_covered": skipped,
            "summary_path": test_info["summary_path"],
        }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return run(output_dir)
    with tempfile.TemporaryDirectory(
        prefix="multiparous-sow-breeding-", dir=str(SCRIPT_DIR)
    ) as temp:
        result = run(Path(temp).resolve())
        result["artifacts"] = None
        result["summary_path"] = None
        return result


def _add_common_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--feature-file",
        required=True,
        type=_absolute_path,
        help="可信特征 pickle 的绝对路径",
    )
    parser.add_argument(
        "--label-dir",
        required=True,
        type=_absolute_path,
        help="包含 <experiment>.csv 的标签目录绝对路径",
    )
    parser.add_argument(
        "--experiments",
        required=True,
        nargs="+",
        help="仅包含经产猪的实验名称列表",
    )
    parser.add_argument(
        "--confirm-multiparous-data",
        required=True,
        action="store_true",
        help="明确确认特征文件与实验列表只包含目标经产猪数据",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "自包含的经产猪配种模型校验、训练和测试工具。"
            "脚本不导入 radar_pig_pkg，所有外部路径必须是绝对路径。"
        )
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="校验 PKL 与标签契约")
    _add_common_input_arguments(validate_parser)
    validate_parser.add_argument(
        "--mode",
        choices=("train", "test"),
        default="train",
        help="test 模式额外剔除发情未配种的猪",
    )

    train_parser = subparsers.add_parser("train", help="生成 63 维特征并训练模型")
    _add_common_input_arguments(train_parser)
    train_parser.add_argument(
        "--model-output", required=True, type=_absolute_path, help="模型输出绝对路径"
    )
    train_parser.add_argument(
        "--interval-type", choices=tuple(INTERVAL_POSITIVE_DAYS), default="-2_0"
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-boost-round", type=int, default=1000)
    train_parser.add_argument("--early-stopping-rounds", type=int, default=10)
    train_parser.add_argument("--nthread", type=int, default=4)
    train_parser.add_argument("--verbose-eval", type=int, default=10)
    train_parser.add_argument(
        "--stratify",
        action="store_true",
        help="按 eva_label 分层拆分；原 Python 入口默认不分层",
    )

    test_parser = subparsers.add_parser("test", help="逐实验测试并保存完整结果")
    _add_common_input_arguments(test_parser)
    test_parser.add_argument(
        "--model-path", required=True, type=_absolute_path, help="模型绝对路径"
    )
    test_parser.add_argument(
        "--output-dir", required=True, type=_absolute_path, help="测试结果目录绝对路径"
    )
    test_parser.add_argument(
        "--interval-type", choices=tuple(INTERVAL_POSITIVE_DAYS), default="-2_0"
    )
    test_parser.add_argument("--threshold", type=float, default=0.52)

    check_parser = subparsers.add_parser(
        "self-check", help="用临时合成数据验证完整训练和测试链"
    )
    check_parser.add_argument(
        "--output-dir",
        type=_absolute_path,
        default=None,
        help="可选绝对目录；省略时使用并清理临时目录",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.command == "validate":
            dataset = _load_dataset(args.feature_file)
            report = inspect_inputs(
                dataset, args.label_dir, args.experiments, mode=args.mode
            )
            print(_json_dump(report))
            return 0 if report["valid"] else 2
        if args.command == "train":
            print(
                _json_dump(
                    train_pipeline(
                        feature_file=args.feature_file,
                        label_dir=args.label_dir,
                        experiments=args.experiments,
                        model_output=args.model_output,
                        interval_type=args.interval_type,
                        test_size=args.test_size,
                        seed=args.seed,
                        num_boost_round=args.num_boost_round,
                        early_stopping_rounds=args.early_stopping_rounds,
                        nthread=args.nthread,
                        verbose_eval=args.verbose_eval,
                        stratify=args.stratify,
                    )
                )
            )
            return 0
        if args.command == "test":
            print(
                _json_dump(
                    test_pipeline(
                        feature_file=args.feature_file,
                        label_dir=args.label_dir,
                        experiments=args.experiments,
                        model_path=args.model_path,
                        output_dir=args.output_dir,
                        interval_type=args.interval_type,
                        threshold=args.threshold,
                    )
                )
            )
            return 0
        if args.command == "self-check":
            print(_json_dump(self_check(args.output_dir)))
            return 0
        parser.error(f"未知命令: {args.command}")
    except PipelineError as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("用户中断")
        return 130
    except Exception:
        LOGGER.exception("未预期错误")
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
