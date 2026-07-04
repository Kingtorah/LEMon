#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量校验 .pt.zip 文件完整性及数据内容有效性
功能：
1. 基础校验：ZIP文件是否损坏、是否包含.pt文件、能否被torch加载
2. 深度校验 (-f)：检查Tensor数据是否包含 NaN、Inf 或 全0
"""

import os
import sys
import argparse
import zipfile
from io import BytesIO
from pathlib import Path
import torch


def check_tensor_anomalies(data) -> str:
    """
    检查加载的数据中是否存在异常 (NaN, Inf, 全0)
    :param data: torch.load 加载出的数据 (通常是一个列表或元组)
    :return: 错误信息字符串，如果没有异常则返回 None
    """
    # 假设 data 是一个列表，结构参考 dataset_parser.py:
    # [lidar_proj, rgb_image, event_volume, bf_depth, af_depth, ...]

    if not isinstance(data, (list, tuple)):
        return "数据格式错误(非List/Tuple)"

    # 定义数据对应的名称（根据 dataset_parser.py 的逻辑推断）
    # 如果列表长度超过名称列表，后续的统称为 "Unknown_Tensor"
    tensor_names = ["LiDAR_Proj", "RGB_Image", "Event_Volume", "BF_Depth", "AF_Depth"]

    for idx, item in enumerate(data):
        # 仅检查 Tensor 类型的数据
        if not isinstance(item, torch.Tensor):
            continue

        name = tensor_names[idx] if idx < len(tensor_names) else f"Tensor_{idx}"

        # 1. 检查 NaN
        if torch.isnan(item).any():
            return f"{name} 包含 NaN"

        # 2. 检查 Inf
        if torch.isinf(item).any():
            return f"{name} 包含 Inf"

        # 3. 检查全 0 (对于某些数据如 RGB 或 Depth，全 0 通常意味着异常)
        # 注意：对于稀疏的 Event Volume，全 0 可能是合法的，但在数据集中极少见
        if item.numel() > 0 and torch.count_nonzero(item) == 0:
            return f"{name} 数据全为 0"

    return None


def check_pt_zip_file(file_path: Path, full_check: bool = False) -> dict:
    """
    检查单个 .pt.zip 文件的完整性
    :param file_path: 文件路径
    :param full_check: 是否进行深度内容检查（NaN/0检测）
    :return: 检查结果字典
    """
    result = {
        "file_path": str(file_path),
        "file_size": file_path.stat().st_size / 1024 / 1024,  # MB
        "is_valid": False,
        "error_msg": ""
    }

    # 1. 检查文件是否为空
    if result["file_size"] < 0.001:
        result["error_msg"] = "文件为空（小于1KB）"
        return result

    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            # 2. 校验ZIP结构
            bad_files = zf.testzip()
            if bad_files:
                result["error_msg"] = f"ZIP结构损坏: {bad_files}"
                return result

            # 3. 检查ZIP内容
            pt_files = [f for f in zf.namelist() if f.endswith('.pt')]
            if not pt_files:
                result["error_msg"] = "ZIP内未找到 .pt 文件"
                return result

            # 4. 加载 .pt 文件
            with zf.open(pt_files[0]) as f:
                buffer = BytesIO(f.read())
                # 加载数据
                data = torch.load(buffer, map_location='cpu')

                # 5. 深度内容检查 (如果开启)
                if full_check:
                    anomaly_msg = check_tensor_anomalies(data)
                    if anomaly_msg:
                        result["error_msg"] = f"数据内容异常: {anomaly_msg}"
                        return result

            # 如果所有步骤通过
            result["is_valid"] = True

    except zipfile.BadZipFile:
        result["error_msg"] = "无效的ZIP文件"
    except RuntimeError as e:
        result["error_msg"] = f"PT加载失败: {str(e)}"
    except Exception as e:
        result["error_msg"] = f"未知错误: {str(e)}"

    return result


def main():
    parser = argparse.ArgumentParser(description='批量校验 .pt.zip 文件完整性及数据质量')
    parser.add_argument('dataset_dir', type=str, help='数据集目录路径')
    # 新增 -f 参数
    parser.add_argument('-f', '--full-check', action='store_true',
                        help='开启深度检查：检测数据是否包含 NaN、Inf 或 全0')
    args = parser.parse_args()

    target_dir = Path(args.dataset_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"错误：无效的目录 -> {target_dir}")
        sys.exit(1)

    print(f"校验目录: {target_dir.absolute()}")
    print(f"深度检查模式 (-f): {'✅ 开启' if args.full_check else '⬜ 关闭'}")
    print("=" * 100)

    pt_zip_files = list(target_dir.glob("*.pt.zip"))
    if not pt_zip_files:
        print("未找到 .pt.zip 文件")
        sys.exit(0)

    stats = {
        "total": len(pt_zip_files),
        "valid": 0,
        "invalid": 0,
        "details": {
            "zip_error": 0,
            "load_error": 0,
            "content_error": 0,  # 新增内容错误统计
            "other": 0
        },
        "bad_files": []
    }

    for idx, file_path in enumerate(pt_zip_files, 1):
        # 打印进度，如果开启深度检查可能会比较慢，因此flush输出
        print(f"\r[{idx}/{stats['total']}] 检查中: {file_path.name} ...", end="", flush=True)

        result = check_pt_zip_file(file_path, full_check=args.full_check)

        if result["is_valid"]:
            stats["valid"] += 1
            # 只有出错时才换行打印，否则覆盖当前行（保持界面清爽）
            # 如果想看所有文件详情，可以去掉 \r 并修改下面的逻辑
        else:
            stats["invalid"] += 1
            stats["bad_files"].append(result)
            print(f"\n    ❌ 异常 | {result['error_msg']}")

            # 错误分类统计
            msg = result["error_msg"]
            if "ZIP" in msg:
                stats["details"]["zip_error"] += 1
            elif "PT加载失败" in msg:
                stats["details"]["load_error"] += 1
            elif "数据内容异常" in msg:  # 对应 -f 检查出的错误
                stats["details"]["content_error"] += 1
            else:
                stats["details"]["other"] += 1

    print(f"\n\n{'=' * 100}")
    print("📊 校验汇总报告")
    print(f"总文件数: {stats['total']}")
    print(f"✅ 正常: {stats['valid']}")
    print(f"❌ 异常: {stats['invalid']}")
    print("-" * 30)
    print(f"   - ZIP/文件损坏: {stats['details']['zip_error'] + stats['details']['other']}")
    print(f"   - PT加载失败:   {stats['details']['load_error']}")
    if args.full_check:
        print(f"   - 数据内容异常: {stats['details']['content_error']} (NaN/Inf/All-Zero)")

    if stats["bad_files"]:
        bad_list_path = target_dir / "bad_files_report.txt"
        with open(bad_list_path, 'w', encoding='utf-8') as f:
            f.write(f"异常文件列表 (Total: {len(stats['bad_files'])})\n")
            f.write(f"检查模式: {'深度检查' if args.full_check else '基础检查'}\n")
            f.write("-" * 60 + "\n")
            for item in stats["bad_files"]:
                f.write(f"{item['file_path']} | {item['error_msg']}\n")
        print(f"\n📝 异常文件列表已保存至: {bad_list_path}")


if __name__ == "__main__":
    main()