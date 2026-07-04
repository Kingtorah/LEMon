import torch
from torch import Tensor
import os

from models.aled import ALED
from models.delta import DELTA
from models.ledepth import LEDepth

def run_delta_example():
  """
  创建一个 LEDepth 模型实例，使用随机张量作为输入，
  执行一次前向传播，并打印输出的张量形状。
  """

  # --- 1. 定义模型参数和输入维度 ---
  # 这些参数应与您在 'train.py' 中初始化 LEDepth 时使用的参数相匹配
  LIDAR_CHANNELS = 1
  EVENT_CHANNELS = 4
  OUT_CHANNELS = 1
  SA_LAYERS = 2
  PATCH_SIZE = 4
  DIMENSIONALITY = 1024
  FFNN_DIMENSIONALITY = 4096
  NBR_HEADS = 4
  PROP_MEM_SIZE = 128

  # 输入数据维度 (Batch Size, Channels, Height, Width)
  BATCH_SIZE = 1
  H = 256  # 图像高度，应能被 PATCH_SIZE 整除
  W = 384  # 图像宽度，应能被 PATCH_SIZE 整除

  # 确保输入维度合理
  if H % PATCH_SIZE != 0 or W % PATCH_SIZE != 0:
    raise ValueError("Height and Width must be divisible by PATCH_SIZE.")

  H_P = H // PATCH_SIZE  # 经过 Patch 后的高度
  W_P = W // PATCH_SIZE  # 经过 Patch 后的宽度
  SEQUENCE_LENGTH = H_P * W_P # 序列长度 L

  # --- 2. 检查设备并创建模型 ---
  device = "cpu" # or "cuda:0"
  print(f"Using device: {device}")

  try:
    # 初始化 LEDepth 模型
    model = LEDepth(
      LIDAR_CHANNELS, EVENT_CHANNELS, OUT_CHANNELS, SA_LAYERS, PATCH_SIZE,
      DIMENSIONALITY, FFNN_DIMENSIONALITY, NBR_HEADS, PROP_MEM_SIZE
    ).to(device)
    model.eval() # 设置为评估模式
    print("LEDepth model initialized successfully.")

  except Exception as e:
    print(f"Error initializing LEDepth model: {e}")
    return

  # --- 3. 创建随机输入张量 ---
  # 模拟数据输入
  # event_input: (B, C_evt, H, W)
  event_input = torch.rand(BATCH_SIZE, EVENT_CHANNELS, H, W, device=device)
  # lidar_input: (B, C_lidar, H, W)
  # 注意: train.py 中检查了 lidar_proj_available，这里我们假设它存在
  lidar_input = torch.rand(BATCH_SIZE, LIDAR_CHANNELS, H, W, device=device)

  # central_mem: (B, L, D) - 首次运行时设为 None
  # L 是序列长度 (H_P * W_P)
  central_mem = None

  # prop_mem: (B, PROP_MEM_SIZE, D) - 首次运行时设为 None
  prop_mem = None

  # crop_positions: (B, 2) - 模拟 RandomCropAlignedWithPatches 的输出
  # 对于非裁剪输入，在 LEDepth.py 中会被设置为 (0, 0)
  crop_positions = torch.zeros(BATCH_SIZE, 2, dtype=torch.long, device=device)

  # --- 4. 执行前向传播 ---
  print("\n--- Running Forward Pass ---")
  try:
    with torch.no_grad():
      # 调用 LEDepth 的 forward 方法
      pred_depths, new_central_mem, new_prop_mem = model(
        lidar_input, event_input, central_mem, prop_mem, crop_positions
      )

    # --- 5. 打印输出形状 ---
    print("\n--- Output Shapes ---")
    print(f"1. Predicted Depths (pred_depths): {pred_depths.shape}")
    print(f"2. Updated Central Memory (new_central_mem): {new_central_mem.shape}")
    print(f"3. Updated Propagation Memory (new_prop_mem): {new_prop_mem.shape}")

    # 预期输出形状 (Expected Shapes for reference)
    print("\n--- Expected Shapes (B=1, H=256, W=384, P=4, D=1024, M=128) ---")
    print(f"1. Predicted Depths: ({BATCH_SIZE}, {OUT_CHANNELS}, {H}, {W})") # (1, 2, 256, 384)
    print(f"2. Updated Central Memory: ({BATCH_SIZE}, {SEQUENCE_LENGTH}, {DIMENSIONALITY})") # (1, 24576, 1024)
    print(f"3. Updated Propagation Memory: ({BATCH_SIZE}, {PROP_MEM_SIZE}, {DIMENSIONALITY})") # (1, 128, 1024)

  except Exception as e:
    print(f"An error occurred during the forward pass: {e}")

if __name__ == "__main__":
  run_delta_example()