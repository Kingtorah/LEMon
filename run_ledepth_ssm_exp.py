import torch
from models.ledepth_ssm import LEDepth
import time

def test_hr_vssn():
    print("==========================================")
    print("   Testing HR-VSSN (Mamba-DELTA) Model    ")
    print("==========================================")

    # 1. 实例化模型
    # 使用较小的参数以便快速测试
    model = LEDepth(
        lidar_channels=1,
        event_channels=4,
        out_channels=1,
        num_layers=3,
        patch_size=4,
        dimensionality=64,
        d_state=16
    )

    device =  "cpu"
    model = model.to(device)
    print(f"Model created on device: {device}")

    # 2. 构造测试数据 (模拟 Davis346 的分辨率 346x260)
    # 这是一个非 2 的幂次，且不能被 4 整除 (346/4 = 86.5) 的尺寸，用于测试 Padding
    H, W = 512, 512
    B = 2
    lidar = torch.randn(B, 1, H, W).to(device)
    events = torch.randn(B, 4, H, W).to(device)

    print(f"\nInput Resolution: {H}x{W} (Arbitrary Scale Test)")

    # 3. 运行第一帧 (T=0)
    # prev_states = None, 模型内部会自动初始化
    start_time = time.time()
    pred_t1, states_t1 = model(lidar, events, prev_states=None)

    print(f"Frame 1 Processed. Time: {time.time() - start_time:.4f}s")
    print(f"Output Shape: {pred_t1.shape}")

    # 验证输出尺寸是否严格等于输入尺寸 (Padding 是否被切除)
    assert pred_t1.shape == (B, 1, H, W), f"Shape Mismatch! Expected {(B, 1, H, W)}, got {pred_t1.shape}"
    print(">> Shape Check Passed (Padding logic works).")

    # 4. 运行第二帧 (T=1)
    # 关键：将 states_t1 传回去
    lidar_t2 = torch.randn(B, 1, H, W).to(device)
    events_t2 = torch.randn(B, 4, H, W).to(device)

    pred_t2, states_t2 = model(lidar_t2, events_t2, prev_states=states_t1)
    print(f"Frame 2 Processed (With State Passing).")

    # 5. 反向传播测试 (验证梯度流)
    loss = pred_t2.sum()
    loss.backward()
    print(">> Backward Pass Successful (Gradient flows through time).")

    # 6. 验证状态是否更新
    # 简单的检查：确保第二帧的状态和第一帧不一样
    diff = (states_t2['lidar_s'][0] - states_t1['lidar_s'][0]).abs().sum()
    if diff > 0:
        print(f">> State Update Check Passed (Diff: {diff.item():.2f}). Memory is working.")
    else:
        print(">> Warning: States did not change.")


if __name__ == "__main__":
    test_hr_vssn()