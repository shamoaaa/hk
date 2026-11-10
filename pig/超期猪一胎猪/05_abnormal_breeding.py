"""
统计经产猪配种结果中超出 -1 到 0 区间的猪只明细
排除：疾病猪、数据异常猪
输出：设备号、耳号、算法配种时间、人工配种时间、配种差、所属单元、胎次等

适用范围：长塘 20260301 及以后的所有批次
"""

import os
import openpyxl
from datetime import datetime, date
import pandas as pd

BASE_DIR = r"e:\5.毫米波雷达\一步到位\excel 3.0\长塘"

# 20260301及以后的目录
MIN_DATE = "20260301"


def find_result_files(base_dir):
    """查找所有人工算法结果汇总文件，返回 [(batch_date, filepath, sub_dir_name), ...]"""
    results = []
    for folder in sorted(os.listdir(base_dir)):
        if folder < MIN_DATE:
            continue
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        # 直接在目录下查找
        _find_in_dir(folder_path, folder, "", results)
        # 子目录（如 三线2栋、四线2栋、二线2栋）
        for sub in os.listdir(folder_path):
            sub_path = os.path.join(folder_path, sub)
            if os.path.isdir(sub_path):
                _find_in_dir(sub_path, folder, sub, results)
    return results


def _find_in_dir(dir_path, batch_date, sub_name, results):
    """在指定目录下查找人工算法结果汇总xlsx"""
    for f in os.listdir(dir_path):
        if not f.endswith(".xlsx"):
            continue
        if f.startswith("~$"):
            continue
        # 匹配人工算法结果汇总文件（排除副本，优先新算法）
        if "人工算法结果汇总" in f:
            # 跳过副本
            if "副本" in f:
                continue
            # 跳过旧算法（如果有新算法版本存在）
            if "旧算法" in f:
                new_algo_exists = any(
                    "新算法" in x and x.endswith(".xlsx") and not x.startswith("~$")
                    for x in os.listdir(dir_path)
                )
                if new_algo_exists:
                    continue
            # 跳过重跑版本之前的（如果有重跑版本）
            if "重跑" not in f:
                rerun_exists = any(
                    "重跑" in x and "副本" not in x and x.endswith(".xlsx") and not x.startswith("~$")
                    for x in os.listdir(dir_path)
                )
                if rerun_exists:
                    continue
            results.append((batch_date, os.path.join(dir_path, f), sub_name))


def parse_date_int(val):
    """将 20260527 格式的整数/字符串转为 date 对象"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        val = int(val)
    elif isinstance(val, str):
        val = val.strip()
        if not val.isdigit():
            return None
        val = int(val)
    else:
        return None
    try:
        return date(val // 10000, (val % 10000) // 100, val % 100)
    except (ValueError, OverflowError):
        return None


def detect_columns(headers):
    """
    根据表头检测各列的位置索引（0-based）。
    返回 dict: {field_name: col_index}
    """
    col_map = {}
    for i, h in enumerate(headers):
        if h is None:
            continue
        h = h.strip()
        if h in ("设备号", "雷达编号"):
            col_map["设备号"] = i
        elif h == "猪舍栏位":
            col_map["栏位"] = i
        elif h == "所属单元":
            col_map["所属单元"] = i
        elif h in ("耳号", "种猪耳号"):
            col_map["耳号"] = i
        elif h == "种猪性质":
            col_map["种猪性质"] = i
        elif h == "胎次":
            col_map["胎次"] = i
        elif h == "猪只品种":
            col_map["品种"] = i
        elif h == "是否强配":
            col_map["是否强配"] = i
        elif h in ("配种时间", "配种日期"):
            col_map["配种时间"] = i
        elif h in ("算法配种时间", "算法配种日期"):
            col_map["算法配种时间"] = i
        elif h == "配种差":
            col_map["配种差"] = i
        elif h == "疾病因素":
            col_map["疾病因素"] = i
        elif h == "是否超期猪":
            col_map["是否超期猪"] = i
        elif h == "数据是否异常":
            col_map["数据是否异常"] = i
    return col_map


def get_cell(row_data, col_map, field, default=None):
    """从行数据中获取指定字段值"""
    idx = col_map.get(field)
    if idx is None or idx >= len(row_data):
        return default
    return row_data[idx]


def compute_breeding_diff(breeding_time, algo_breeding_time):
    """计算配种差 = 算法配种时间 - 人工配种时间（天数）"""
    d1 = parse_date_int(breeding_time)
    d2 = parse_date_int(algo_breeding_time)
    if d1 is None or d2 is None:
        return None
    return (d2 - d1).days


def compute_breeding_day(breeding_time, batch_date_str):
    """
    计算猪在第几天配种。
    文件夹日期为第2天，所以第1天 = 文件夹日期 - 1天。
    配种天数 = 人工配种时间 - 第1天的日期 + 1
    即：配种天数 = 人工配种时间 - (文件夹日期 - 1) 
    例如文件夹20260525，第1天=20260524，如果配种时间=20260526，则是第3天配种
    """
    breeding_date = parse_date_int(breeding_time)
    if breeding_date is None:
        return None
    # 文件夹日期 = 第2天
    folder_date = parse_date_int(int(batch_date_str))
    if folder_date is None:
        return None
    # 第1天 = 文件夹日期 - 1
    from datetime import timedelta
    day1 = folder_date - timedelta(days=1)
    # 配种在第几天
    return (breeding_date - day1).days


def process_file(batch_date, filepath, sub_name):
    """处理单个人工算法结果汇总文件，返回异常猪明细列表"""
    records = []
    try:
        wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
    except Exception as e:
        print(f"  [WARN] 无法打开 {filepath}: {e}")
        return records

    # 优先使用 Sheet1（有些文件的 active sheet 是 Sheet2 汇总表）
    if "Sheet1" in wb.sheetnames:
        ws = wb["Sheet1"]
    else:
        ws = wb.worksheets[0]

    if ws is None or ws.max_row is None or ws.max_row < 2:
        wb.close()
        return records

    # 读取表头
    rows = list(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    if not rows:
        wb.close()
        return records
    headers = [str(c) if c else None for c in rows[0]]
    col_map = detect_columns(headers)

    # 必须有配种时间和算法配种时间列
    if "配种时间" not in col_map or "算法配种时间" not in col_map:
        print(f"  [WARN] {filepath} 缺少配种时间列，跳过")
        wb.close()
        return records

    # 遍历数据行
    for row in ws.iter_rows(min_row=2, values_only=True):
        row_data = list(row)

        # 过滤：只看经产猪（生产种猪）
        pig_type = get_cell(row_data, col_map, "种猪性质")
        if pig_type and "后备" in str(pig_type):
            continue

        # 过滤：排除疾病
        disease = get_cell(row_data, col_map, "疾病因素")
        if disease and str(disease).strip():
            continue

        # 过滤：排除数据异常
        abnormal = get_cell(row_data, col_map, "数据是否异常")
        if abnormal and str(abnormal).strip():
            continue

        # 获取配种时间
        breeding_time = get_cell(row_data, col_map, "配种时间")
        algo_breeding_time = get_cell(row_data, col_map, "算法配种时间")

        # 判断特殊情况
        # 先尝试读取文件中已有的配种差
        existing_diff = get_cell(row_data, col_map, "配种差")

        # 情况1: 误报 - 人工无配种时间，算法有
        if (breeding_time is None or str(breeding_time).strip() == "") and algo_breeding_time:
            diff_label = "误报"
            diff_val = None
        # 情况2: 未预测 - 人工有配种时间，算法无
        elif breeding_time and (algo_breeding_time is None or str(algo_breeding_time).strip() == ""):
            diff_label = "未预测"
            diff_val = None
        # 情况3: 都有，计算差值
        elif breeding_time and algo_breeding_time:
            diff_val = compute_breeding_diff(breeding_time, algo_breeding_time)
            if diff_val is None:
                # 尝试用已有值
                if existing_diff is not None and isinstance(existing_diff, (int, float)):
                    diff_val = int(existing_diff)
                else:
                    continue
            diff_label = str(diff_val)
        else:
            # 都没有，跳过
            continue

        # 如果在已有的配种差里看到"误报"/"未预测"字样
        if isinstance(existing_diff, str):
            if "误报" in existing_diff:
                diff_label = "误报"
                diff_val = None
            elif "未预测" in existing_diff:
                diff_label = "未预测"
                diff_val = None

        # 筛选：只保留超出 -1 到 0 区间的
        if diff_val is not None and -1 <= diff_val <= 0:
            continue  # 在正常范围内，跳过

        # 收集信息
        record = {
            "批次日期": batch_date,
            "子目录": sub_name if sub_name else "",
            "设备号": get_cell(row_data, col_map, "设备号", ""),
            "栏位": get_cell(row_data, col_map, "栏位", ""),
            "所属单元": get_cell(row_data, col_map, "所属单元", ""),
            "耳号": get_cell(row_data, col_map, "耳号", ""),
            "种猪性质": pig_type or "",
            "胎次": get_cell(row_data, col_map, "胎次", ""),
            "品种": get_cell(row_data, col_map, "品种", ""),
            "是否强配": get_cell(row_data, col_map, "是否强配", ""),
            "人工配种时间": breeding_time if breeding_time else "",
            "算法配种时间": algo_breeding_time if algo_breeding_time else "",
            "配种差": diff_label,
            "配种差(天)": diff_val if diff_val is not None else "",
            "配种在第几天": compute_breeding_day(breeding_time, batch_date) if breeding_time else "",
            "是否超期猪": get_cell(row_data, col_map, "是否超期猪", "") or "",
        }
        records.append(record)

    wb.close()
    return records


def try_enrich_from_查情配种记录(batch_dir, records):
    """尝试从查情配种记录补充胎次等缺失信息"""
    # 查找查情配种记录文件
    chaiqing_file = None
    for f in os.listdir(batch_dir):
        if f.startswith("~$"):
            continue
        if "查情配种记录" in f and f.endswith(".xlsx") and "副本" not in f:
            chaiqing_file = os.path.join(batch_dir, f)
            break

    if not chaiqing_file:
        return

    try:
        wb = openpyxl.load_workbook(chaiqing_file, data_only=True, read_only=True)
    except Exception:
        return

    ws = wb.active
    if ws is None or ws.max_row is None or ws.max_row < 2:
        wb.close()
        return

    # 建立 设备号->信息 的映射
    rows_data = list(ws.iter_rows(min_row=1, values_only=True))
    if not rows_data:
        wb.close()
        return

    headers = [str(c) if c else None for c in rows_data[0]]
    col_map = detect_columns(headers)

    # 构建查找表 (设备号 -> row_data)
    lookup = {}
    for row_data in rows_data[1:]:
        row_data = list(row_data)
        dev_id = get_cell(row_data, col_map, "设备号")
        if dev_id:
            lookup[str(dev_id)] = row_data

    # 补充缺失信息
    for rec in records:
        dev_id = str(rec["设备号"])
        if dev_id in lookup:
            row_data = lookup[dev_id]
            if not rec["胎次"]:
                rec["胎次"] = get_cell(row_data, col_map, "胎次", "")
            if not rec["品种"]:
                rec["品种"] = get_cell(row_data, col_map, "品种", "")
            if not rec["所属单元"]:
                rec["所属单元"] = get_cell(row_data, col_map, "所属单元", "")

    wb.close()


def main():
    print(f"扫描目录: {BASE_DIR}")
    print(f"筛选条件: 经产猪、配种差超出[-1, 0]、排除疾病和数据异常")
    print("=" * 80)

    file_list = find_result_files(BASE_DIR)
    print(f"找到 {len(file_list)} 个人工算法结果汇总文件\n")

    all_records = []

    for batch_date, filepath, sub_name in file_list:
        print(f"处理: {batch_date}/{sub_name} - {os.path.basename(filepath)}")
        records = process_file(batch_date, filepath, sub_name)

        # 尝试补充胎次等信息
        batch_dir = os.path.dirname(filepath)
        try_enrich_from_查情配种记录(batch_dir, records)

        all_records.extend(records)
        print(f"  -> 异常配种猪: {len(records)} 头")

    print("\n" + "=" * 80)
    print(f"总计: {len(all_records)} 头经产猪配种结果超出 [-1, 0] 区间")

    # 汇总统计
    if all_records:
        df = pd.DataFrame(all_records)

        # 按配种差分布统计
        print("\n--- 配种差分布 ---")
        print(df["配种差"].value_counts().to_string())

        # 按批次统计
        print("\n--- 按批次统计 ---")
        batch_summary = df.groupby(["批次日期", "子目录"]).size().reset_index(name="异常数量")
        print(batch_summary.to_string(index=False))

        # 输出到 Excel
        output_path = os.path.join(
            r"e:\5.毫米波雷达\一步到位\4.0",
            "异常配种明细_经产猪.xlsx"
        )
        df.to_excel(output_path, index=False, sheet_name="异常配种明细")
        print(f"\n明细已保存至: {output_path}")


if __name__ == "__main__":
    main()
