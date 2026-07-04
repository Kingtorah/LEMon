import torch
import torch.nn as nn
from models.ledepth_ssm_smst import LEDepth  # 假设您将上面的代码保存为此文件名


def test_ledepth_forward():
    print("=== Testing LEDepth with Crop Positions ===")

    # 1. 配置参数
    batch_size = 3
    height, width = 512, 512  # 输入图像尺寸
    patch_size = 4
    dim = 96
    device = 'cuda:0'

    print(f"Device: {device}")

    # 2. 实例化模型
    # 假设 max_res 为 (720, 1284)
    model = LEDepth(
        lidar_channels=1,
        event_channels=4,
        out_channels=1,
        dimensionality=dim,
        patch_size=patch_size
    ).to(device)

    print("Model instantiated.")

    # 3. 创建随机输入数据
    # LiDAR 输入 (B, 1, H, W)
    lidar_input = torch.randn(batch_size, 1, height, width).to(device)

    # Event 输入 (B, 1, H, W) - 这里假设简单的通道数为1，实际可能是 Voxel grid
    event_input = torch.randn(batch_size, 4, height, width).to(device)
    crop_positions = torch.tensor([[ 48, 176],
        [496,  64],
        [480,  64]], dtype=torch.float32).to(device)

    print(f"Input Shapes: LiDAR {lidar_input.shape}, Event {event_input.shape}")
    print(f"Crop Positions: \n{crop_positions}")

    # 第一次调用 (t=0)，prev_states 为 None
    depth_pred, student_pred, states = model(
        lidar_input=lidar_input,
        event_input=event_input,
        prev_states=None,
        crop_positions=crop_positions
    )

    print("\n--- Output Shapes (t=0) ---")
    print(f"Fused Depth: {depth_pred.shape}")  # 应该与输入 H, W 一致
    print(f"Student Depth: {student_pred.shape}")  # 应该与输入 H, W 一致

    # 检查状态字典 keys
    print(f"State keys: {states.keys()}")

    # 5. 测试连续帧 (t=1)
    # 再次传入随机数据和上一步的 states
    lidar_input_t1 = torch.randn(batch_size, 1, height, width).to(device)
    event_input_t1 = torch.randn(batch_size, 4, height, width).to(device)

    # 假设 t=1 时，裁剪窗口稍微移动了一点
    crop_positions_t1 = torch.tensor([[ 48, 176],
        [496,  64],
        [480,  64]], dtype=torch.float32).to(device)

    depth_pred_t1, _, _ = model(
        lidar_input=lidar_input_t1,
        event_input=event_input_t1,
        prev_states=states,
        crop_positions=crop_positions_t1
    )

    print("\n--- Output Shapes (t=1) ---")
    print(f"Fused Depth (t=1): {depth_pred_t1.shape}")
    print("Test Passed Successfully.")


if __name__ == "__main__":
    try:
        test_ledepth_forward()
    except Exception as e:
        print(f"\nTest Failed with error: {e}")
        import traceback

        traceback.print_exc()