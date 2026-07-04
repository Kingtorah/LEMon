import torch
from torch import Tensor
import os

# 假设您将上一条回答的代码保存为了 models/LEDepth.py
from models.ledepth import LEDepth


def run_ledpeth_example():
    """
  创建一个 LEDepth 模型实例，使用随机张量作为输入，
  执行一次前向传播，并打印输出的张量形状。
  """

    # --- 1. 定义模型参数和输入维度 ---
    # 这些参数与 LEDepth 的 __init__ 参数对应
    LIDAR_CHANNELS = 1
    EVENT_CHANNELS = 4
    OUT_CHANNELS = 1
    SA_LAYERS = 2
    PATCH_SIZE = 4  # 注意：Patch Size 越小，显存占用越高
    DIMENSIONALITY = 1024  # 为了测试运行速度，我稍微减小了维度（原代码是1024），你可以改回1024
    FFNN_DIMENSIONALITY = 4096  # 同上
    NBR_HEADS = 4
    PROP_MEM_SIZE = 128

    # 输入数据维度 (Batch Size, Channels, Height, Width)
    BATCH_SIZE = 1
    H = 256  # 图像高度
    W = 384  # 图像宽度

    # 确保输入维度合理
    if H % PATCH_SIZE != 0 or W % PATCH_SIZE != 0:
        raise ValueError("Height and Width must be divisible by PATCH_SIZE.")

    H_P = H // PATCH_SIZE  # 经过 Patch 后的高度
    W_P = W // PATCH_SIZE  # 经过 Patch 后的宽度
    SEQUENCE_LENGTH = H_P * W_P  # 序列长度 L

    # --- 2. 检查设备并创建模型 ---
    # 如果有显卡则使用显卡，否则使用 CPU
    device =  "cpu"
    print(f"Using device: {device}")

    try:
        # 初始化 LEDepth 模型
        model = LEDepth(
            lidar_channels=LIDAR_CHANNELS,
            event_channels=EVENT_CHANNELS,
            out_channels=OUT_CHANNELS,
            sa_layers=SA_LAYERS,
            patch_size=PATCH_SIZE,
            dimensionality=DIMENSIONALITY,
            ffnn_dimensionality=FFNN_DIMENSIONALITY,
            nbr_heads=NBR_HEADS,
            prop_mem_size=PROP_MEM_SIZE
        ).to(device)

        model.eval()  # 设置为评估模式
        print("LEDepth (Structure-Aware) model initialized successfully.")

    except Exception as e:
        print(f"Error initializing LEDepth model: {e}")
        # 打印详细错误栈以便调试
        import traceback
        traceback.print_exc()
        return

    # --- 3. 创建随机输入张量 ---
    # event_input: (B, C_evt, H, W)
    event_input = torch.rand(BATCH_SIZE, EVENT_CHANNELS, H, W, device=device)

    # lidar_input: (B, C_lidar, H, W)
    # 模拟稀疏 LiDAR 输入
    lidar_input = torch.rand(BATCH_SIZE, LIDAR_CHANNELS, H, W, device=device)
    # 模拟稀疏性：将 95% 的像素设为 0
    mask = torch.rand_like(lidar_input) > 0.05
    lidar_input[mask] = 0.0

    # central_mem: (B, L, D) - 首次运行时设为 None
    central_mem = None

    # prop_mem: (B, PROP_MEM_SIZE, D) - 首次运行时设为 None
    prop_mem = None

    # crop_positions: (B, 2)
    crop_positions = torch.zeros(BATCH_SIZE, 2, dtype=torch.long, device=device)

    # --- 4. 执行前向传播 ---
    print("\n--- Running Forward Pass ---")
    try:
        with torch.no_grad():
            # [关键修改] LEDepth 返回 4 个值：
            # 1. pred_depths: 最终融合后的深度图
            # 2. new_central_mem: 更新后的 GRU 中心记忆
            # 3. new_prop_mem: 更新后的传播记忆
            # 4. aux_depth: (新增) SSPL 辅助分支预测的深度，用于计算 Teacher/Student Loss
            pred_depths, new_central_mem, new_prop_mem, aux_depth = model(
                lidar_input, event_input, central_mem, prop_mem, crop_positions
            )

        # --- 5. 打印输出形状 ---
        print("\n--- Output Shapes ---")
        print(f"1. Predicted Depths (Final Output): {pred_depths.shape}")
        print(f"2. Updated Central Memory:          {new_central_mem.shape}")
        print(f"3. Updated Propagation Memory:      {new_prop_mem.shape}")
        print(f"4. SSPL Aux Depth (Student Branch): {aux_depth.shape}")

        # 预期输出形状验证
        print(f"\n--- Expected Shapes (B={BATCH_SIZE}, H={H}, W={W}, P={PATCH_SIZE}, D={DIMENSIONALITY}) ---")
        print(f"1. Final Depth: ({BATCH_SIZE}, {OUT_CHANNELS}, {H}, {W})")
        print(
            f"2. Central Mem: ({BATCH_SIZE}, {SEQUENCE_LENGTH}, {DIMENSIONALITY}) -> (Sequence Len = {H_P}*{W_P} = {SEQUENCE_LENGTH})")
        print(f"3. Prop Mem:    ({BATCH_SIZE}, {PROP_MEM_SIZE}, {DIMENSIONALITY})")
        print(f"4. Aux Depth:   ({BATCH_SIZE}, 1, {H}, {W})")  # 辅助头也会将 Patch 还原为图像尺寸

    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_ledpeth_example()