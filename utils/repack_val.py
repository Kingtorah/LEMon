# import os
# import torch
# import argparse
#
#
# def load_sequences_from_files(file_paths):
#     sequences = []
#     for file_path in file_paths:
#         sequences.extend(torch.load(file_path))
#     return sequences
#
#
# def save_new_sequences(sequences, output_path, prefix, sequences_per_file):
#     num_sequences = len(sequences)
#     num_new_sequences = num_sequences // sequences_per_file
#     for i in range(num_new_sequences):
#         new_sequence = sequences[i * sequences_per_file:(i + 1) * sequences_per_file]
#         torch.save(new_sequence, f"{output_path}/{prefix}_seq{i:04}.pt")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Reorganize .pt files")
#     parser.add_argument("--input_folder", type=str, nargs='?', default=r"E:\My_Project\DLProject\dataset\addevent_dis",
#                         help="Path to the folder containing the input .pt files")
#     parser.add_argument("--output_folder", type=str, nargs='?', default=r"E:\My_Project\DLProject\dataset\day1_test",help="Path to the folder to save the new .pt files")
#     parser.add_argument("--prefix", type=str, nargs='?', default="outdoor_day1",  help="Prefix for the new .pt files")
#     parser.add_argument("--sequences_per_file", type=int, nargs='?', default=5133, help="Number of sequences per new .pt file")
#     parser.add_argument("--num_files_to_read", type=int, nargs='?', default=1711, help="Number of .pt files to read from the end")
#
#     args = parser.parse_args()
#
#     # 获取所有 .pt 文件的路径
#     pt_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(".pt")]
#
#     # 排序文件列表以确保顺序一致
#     pt_files.sort()
#
#     # 选择最后 num_files_to_read 个 .pt 文件
#     pt_files = pt_files[-args.num_files_to_read:]
#
#     # 加载这些 .pt 文件中的所有序列
#     sequences = load_sequences_from_files(pt_files)
#
#     if args.sequences_per_file <= 0:
#         args.sequences_per_file = len(sequences)
#
#     # 确保序列总数是 sequences_per_file 的倍数，不足部分舍弃
#     sequences = sequences[:(len(sequences) // args.sequences_per_file) * args.sequences_per_file]
#
#     # 将这些序列保存为新的 .pt 文件，每个文件包含 sequences_per_file 对数据
#     save_new_sequences(sequences, args.output_folder, args.prefix, args.sequences_per_file)
#
#
# if __name__ == "__main__":
#     main()
import os
import glob
import zipfile
from io import BytesIO
import torch

# --- 1. 定义路径 ---
train_dir = "/root/autodl-tmp/dataset/MVSEC/processed/train/day2"
val_dir = "/root/autodl-tmp/dataset/MVSEC/processed/val/day2"

# 确保验证集目录存在
os.makedirs(val_dir, exist_ok=True)

# --- 2. 获取并排序所有的 pt.zip 文件 ---
# 原始代码使用了 f"{prefix}_seq{i:04}.pt.zip" 命名，自然排序即可保证时序正确
all_train_files = sorted(glob.glob(os.path.join(train_dir, "*.pt.zip")))

print(f"在训练集中共检测到 {len(all_train_files)} 个 .pt.zip 文件。")

if len(all_train_files) < 765:
    raise ValueError("训练集文件总数少于 765 个，无法执行划分！")

# 提取后 765 个文件作为验证集候选
val_files_to_process = all_train_files[-765:]
print(f"已锁定后 {len(val_files_to_process)} 个文件准备转换为验证集...")

# --- 3. 读取并展平所有数据帧 ---
all_frames = []
prefix = ""

print("正在解包并读取数据，请稍候...")
for file_path in val_files_to_process:
    filename_zip = os.path.basename(file_path)
    filename_pt = filename_zip[:-4]  # 去掉 .zip 后缀，得到内部的 .pt 文件名

    # 获取文件前缀 (例如 outdoor_day2) 以用于新文件名
    if not prefix:
        prefix = filename_pt.split("_seq")[0]

    with zipfile.ZipFile(file_path, "r") as z:
        pt_data = z.read(filename_pt)
        # 从字节流中加载 PyTorch 数据
        sequence = torch.load(BytesIO(pt_data))
        all_frames.extend(sequence)

print(f"成功提取了 {len(all_frames)} 帧独立的点云/事件/深度数据组合。")

# --- 4. 按 20 帧重新切分并打包到验证集 ---
val_seq_len = 20
nb_seq = len(all_frames) // val_seq_len

print(f"正在按每组 {val_seq_len} 帧重新打包为 {nb_seq} 个验证集文件...")

for i in range(nb_seq):
    # 切片获取 20 帧数据
    sequence = all_frames[i * val_seq_len: (i + 1) * val_seq_len]

    # 构造新文件名
    filename = f"{prefix}_val_seq{i:04}.pt"

    # 写入 BytesIO 缓存
    buffer = BytesIO()
    torch.save(sequence, buffer)

    # 压缩并保存到 val 目录
    zip_path = os.path.join(val_dir, f"{filename}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(filename, buffer.getvalue())

print(f"验证集打包完成！已保存至: {val_dir}")

# # --- 5. 清理原训练集中的多余文件 ---
# print("正在清理原训练集中的移出文件...")
# for file_path in val_files_to_process:
#     os.remove(file_path)
#
# print(f"清理完毕。目前训练集剩余文件数: {len(all_train_files) - 765}。")
