#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集解析工具：解析DELTA/ALED预处理后的.pt/.pt.zip文件
提取事件数据、LiDAR投影、深度数据等核心信息
新增功能：
1. 默认文件路径
2. 可选可视化功能（LiDAR/事件体/深度图）
"""
import os
import sys

# 添加项目根目录到Python路径（关键修复）
# 获取当前脚本（dataset_parser.py）的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（utils的上级目录）
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import argparse
import zipfile
from math import log
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from visualization.visualization import depth_image_to_img, event_volume_to_img, lidar_proj_to_img


def visualize_item(item_idx: int, lidar_proj, rgb_image, event_volume, bf_depth, af_depth):
    """
    可视化单个item的所有数据
    图片保存到utils/temp目录，不直接展示
    """
    # 定义保存目录（utils/temp）
    save_dir = os.path.join(current_dir, "temp")
    # 确保保存目录存在，不存在则创建
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(20, 15))
    plot_idx = 1

    # 1. LiDAR投影可视化
    if lidar_proj is not None and lidar_proj.numel() > 0:
        plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        lidar_img = lidar_proj_to_img(lidar_proj.unsqueeze(0))[0].permute(1, 2, 0).cpu()
        plt.imshow(lidar_img)
        plt.title(f'Item {item_idx} - LiDAR Projection')
        plt.axis('off')

    # 2. RGB图像
    if rgb_image is not None and rgb_image.numel() > 0:
        plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        rgb_img = rgb_image.permute(1, 2, 0).cpu()
        # 归一化RGB图像到[0,1]
        if rgb_img.max() > 1:
            rgb_img = rgb_img / 255.0
        plt.imshow(rgb_img)
        plt.title(f'Item {item_idx} - RGB Image')
        plt.axis('off')

    # 3. 事件体可视化
    if event_volume is not None and event_volume.numel() > 0:
        plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        event_img = event_volume_to_img(event_volume.unsqueeze(0))[0].permute(1, 2, 0).cpu()
        plt.imshow(event_img)
        plt.title(f'Item {item_idx} - Event Volume')
        plt.axis('off')

    # 4. BF Depth可视化
    if bf_depth is not None and bf_depth.numel() > 0:
        plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        bf_depth_img = depth_image_to_img(bf_depth.unsqueeze(0))[0].permute(1, 2, 0).cpu()
        plt.imshow(bf_depth_img)
        plt.title(f'Item {item_idx} - BF Depth')
        plt.axis('off')

    # 5. AF Depth可视化
    if af_depth is not None and af_depth.numel() > 0:
        plt.subplot(2, 3, plot_idx)
        plot_idx += 1
        af_depth_img = depth_image_to_img(af_depth.unsqueeze(0))[0].permute(1, 2, 0).cpu()
        plt.imshow(af_depth_img)
        plt.title(f'Item {item_idx} - AF Depth')
        plt.axis('off')

    plt.tight_layout()
    # 拼接完整保存路径
    save_path = os.path.join(save_dir, f'item_{item_idx}_visualization.png')
    # 保存图片（提高dpi，确保清晰度）
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # 移除plt.show()，不直接展示图片
    # 关闭画布释放资源
    plt.close()
    # 打印保存路径，方便查看
    print(f"可视化图片已保存至: {save_path}")


# ======================== 原有核心功能 ========================
def load_preprocessed_file(file_path: str) -> list:
    """加载预处理后的.pt或.pt.zip文件"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 处理压缩文件
    if file_path.suffixes == ['.pt', '.zip'] or file_path.suffix == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zf:
            pt_filename = [f for f in zf.namelist() if f.endswith('.pt')][0]
            with zf.open(pt_filename) as f:
                buffer = BytesIO(f.read())
                data = torch.load(buffer, map_location='cpu')
    # 处理普通.pt文件
    elif file_path.suffix == '.pt':
        data = torch.load(file_path, map_location='cpu')
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    return data


def analyze_tensor(tensor: torch.Tensor, name: str) -> dict:
    """分析单个张量的关键信息（增加零点数量统计）"""
    if tensor is None:
        return {
            'name': name,
            'exists': False,
            'shape': None,
            'dtype': None,
            'zero_count': 0,
            'non_zero_count': 0,
            'nan_count': 0,
            'min': None,
            'max': None,
            'mean': None
        }

    # 基础信息提取
    valid_mask = ~torch.isnan(tensor)
    valid_tensor = tensor[valid_mask]

    # 计算总元素数和零点数量
    total_elements = tensor.numel()
    zero_count = total_elements - int((tensor != 0).sum().item())

    return {
        'name': name,
        'exists': True,
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'total_elements': total_elements,
        'zero_count': zero_count,
        'non_zero_count': int((tensor != 0).sum().item()),
        'nan_count': int(torch.isnan(tensor).sum().item()),
        'min': valid_tensor.min().item() if valid_tensor.numel() > 0 else None,
        'max': valid_tensor.max().item() if valid_tensor.numel() > 0 else None,
        'mean': valid_tensor.mean().item() if valid_tensor.numel() > 0 else None
    }


def parse_dataset_file(file_path: str, print_details: bool = True, visualize: bool = False) -> dict:
    """
    解析预处理后的数据集文件
    :param file_path: 文件路径（.pt 或 .pt.zip）
    :param print_details: 是否打印详细信息
    :param visualize: 是否可视化数据
    :return: 解析结果字典
    """
    # 加载数据
    data = load_preprocessed_file(file_path)

    # 初始化结果
    result = {
        'file_path': file_path,
        'sequence_length': len(data),
        'items': []
    }

    # 遍历序列中的每个item
    for item_idx, item in enumerate(data):
        # 每个item结构: [lidar_proj, rgb_image, event_volume, bf_depth, af_depth]
        lidar_proj, rgb_image, event_volume, bf_depth, af_depth = item[:5]
        # print(lidar_proj)

        # 分析每个张量
        item_info = {
            'item_index': item_idx,
            'lidar_projection': analyze_tensor(lidar_proj, 'lidar_projection'),
            'rgb_image': analyze_tensor(rgb_image, 'rgb_image'),
            'event_volume': analyze_tensor(event_volume, 'event_volume'),
            'bf_depth': analyze_tensor(bf_depth, 'bf_depth'),
            'af_depth': analyze_tensor(af_depth, 'af_depth')
        }
        result['items'].append(item_info)

        # 可视化当前item（如果开启）
        if visualize:
            visualize_item(item_idx, lidar_proj, rgb_image, event_volume, bf_depth, af_depth)

    # 打印总结信息
    if print_details:
        print("=" * 70)
        print(f"文件解析结果: {file_path}")
        print("=" * 70)
        print(f"序列总长度: {result['sequence_length']}")
        print("\n" + "-" * 70)

        for item_info in result['items']:
            print(f"\nItem {item_info['item_index']}")
            for tensor_name, tensor_info in item_info.items():
                if tensor_name == 'item_index':
                    continue

                print(f"\n{tensor_name.upper()}:")
                if not tensor_info['exists']:
                    print("  不存在")
                    continue

                print(f"  形状: {tensor_info['shape']}")
                print(f"  数据类型: {tensor_info['dtype']}")
                print(f"  总元素数: {tensor_info['total_elements']}")
                print(f"  零点数量: {tensor_info['zero_count']}")
                print(f"  非零点数量: {tensor_info['non_zero_count']}")
                print(f"  NaN点数量: {tensor_info['nan_count']}")
                print(f"  最小值: {tensor_info['min']:.6f}" if tensor_info['min'] else "  最小值: N/A")
                print(f"  最大值: {tensor_info['max']:.6f}" if tensor_info['max'] else "  最大值: N/A")
                print(f"  平均值: {tensor_info['mean']:.6f}" if tensor_info['mean'] else "  平均值: N/A")
            print("-" * 70)

    return result


def main():
    """命令行接口"""
    # 默认文件路径
    DEFAULT_FILE_PATH = "/root/autodl-tmp/dataset/SLED/processed/train/Town07_00_seq0007.pt.zip"

    parser = argparse.ArgumentParser(description='解析DELTA/ALED预处理数据集文件')
    # 文件路径参数（可选，默认使用预设路径）
    parser.add_argument('file_path', type=str, nargs='?', default=DEFAULT_FILE_PATH,
                        help=f'预处理文件路径 (.pt 或 .pt.zip)，默认: {DEFAULT_FILE_PATH}')
    parser.add_argument('--no-print', action='store_true', help='不打印详细信息，仅返回数据')
    parser.add_argument('--visualize', '-v', action='store_true', help='可视化数据并保存图片到utils/temp')

    args = parser.parse_args()

    # 解析文件（支持可视化开关）
    parse_dataset_file(
        file_path=args.file_path,
        print_details=not args.no_print,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()