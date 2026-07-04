#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the PyTorch code for the ALED network, as originally described in the "Learning
to Estimate Two Dense Depths from LiDAR and Event Data" article (SCIA 2023).
"""

import torch
from torch import nn, Tensor
# import sys
# import os
#
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
#
# # 2. 把这两个路径强行塞进 Python 的环境变量里
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)
#
# from submodules.aled_submodules import ConvEncodingHead, Decoder
# from submodules.shared_submodules import ConvGRU, ResidualBasicEncoder
#
from models.submodules.aled_submodules import ConvEncodingHead, Decoder
from models.submodules.shared_submodules import ConvGRU, ResidualBasicEncoder


class ALED(nn.Module):
  """
  The ALED network, as described in the article.
  It is composed of 2 branches, one for the projected LiDAR clouds and one for the events.
  It uses convolutions for encoding/decoding, and Convolutional Gated Recurrent Units (ConvGRUs) for
  fusion, memory, and asynchronicity purposes.
  """

  def __init__(self, lidar_channels: int, event_channels: int, out_channels: int):
    super().__init__()

    # The encoding heads for the LiDAR, RGB, and event inputs
    self.lidar_head = ConvEncodingHead(lidar_channels, 32, 5, 1, 2)
    self.event_head = ConvEncodingHead(event_channels, 32, 5, 1, 2)

    # The 3 LiDAR encoders
    self.lidar_encoder1 = ResidualBasicEncoder(32, 64, 5, 2, 2, True)
    self.lidar_encoder2 = ResidualBasicEncoder(64, 128, 5, 2, 2, True)
    self.lidar_encoder3 = ResidualBasicEncoder(128, 256, 5, 2, 2, False)

    # The 3 event encoders
    self.event_encoder1 = ResidualBasicEncoder(32, 64, 5, 2, 2, True)
    self.event_encoder2 = ResidualBasicEncoder(64, 128, 5, 2, 2, True)
    self.event_encoder3 = ResidualBasicEncoder(128, 256, 5, 2, 2, False)

    # The 4 convGRU blocks for the LiDAR
    self.conv_gru_lidar0 = ConvGRU(32, 32+32, 3)
    self.conv_gru_lidar1 = ConvGRU(64, 64+64, 3)
    self.conv_gru_lidar2 = ConvGRU(128, 128+128, 3)
    self.conv_gru_lidar3 = ConvGRU(256, 256, 3)

    # The 4 convGRU blocks for the events
    self.conv_gru_events0 = ConvGRU(32, 32+32, 3)
    self.conv_gru_events1 = ConvGRU(64, 64+64, 3)
    self.conv_gru_events2 = ConvGRU(128, 128+128, 3)
    self.conv_gru_events3 = ConvGRU(256, 256, 3)

    # The 2 residual blocks
    self.residual_block1 = ResidualBasicEncoder(256, 256, 3, 1, 1, False)
    self.residual_block2 = ResidualBasicEncoder(256, 256, 3, 1, 1, False)

    # The 3 decoders
    self.decoder1 = Decoder(256, 128, 128, 2, 5, 1, 2)
    self.decoder2 = Decoder(128, 64, 64, 2, 5, 1, 2)
    self.decoder3 = Decoder(64, 32, 32, 2, 5, 1, 2)

    # The 3 convolutions used to reduce the number of channels after concatenating the decoded state
    # and the hidden state of the corresponding convGRU module
    self.conv_concat1 = nn.Conv2d(256, 128, 1)
    self.conv_concat2 = nn.Conv2d(128, 64, 1)
    self.conv_concat3 = nn.Conv2d(64, 32, 1)

    # The final prediction layer
    self.prediction_layer = nn.Conv2d(32, out_channels, 1)


  def forward(self, lidar_input: Tensor | None, event_input: Tensor | None,
              central_mems: list[Tensor] | None) -> tuple[Tensor, list[Tensor]]:
    """
    The shape of the LiDAR/event inputs should be (B, C, H, W), where B is the number of batches, C
    the number of channels, and H and W the height and width. If at a given time, an input is not
    available, its value should be set to None. Multiple inputs can be set to None, but at least one
    input should have a value to update the memories / predict the new depth map.
    The central memories should be None if they are not yet initialized, otherwise they should be an
    array of 4 Tensors, each of shape (B, C_H, H_S, W_S) where C_H is the number of channels of the
    hidden state, and H_S and W_S are the spatial size after applying the convolutions.
    """

    # PART 1: VERIFYING IF THE GIVEN MEMORY IS INITIALIZED, OTHERWISE INITIALIZE IT
    # We initialize the central memories if necessary, as a list of Nones
    if central_mems is None:
      central_mems = [None, None, None, None]

    # PART 2: ENCODING THE LIDAR DATA AND UPDATING THE MEMORIES (IF AVAILABLE)
    if lidar_input is not None:
      # We first apply the head, to go from M layers to 32, and give the result to the top level
      # convGRU to update its state
      encoded_lidar = self.lidar_head(lidar_input)
      central_mems[0] = self.conv_gru_lidar0(encoded_lidar, central_mems[0])

      # We apply the first encoder and give it to the convGRU to update its state
      encoded_lidar = self.lidar_encoder1(encoded_lidar)
      central_mems[1] = self.conv_gru_lidar1(encoded_lidar, central_mems[1])

      # We apply the second encoder and give it to the convGRU to update its state
      encoded_lidar = self.lidar_encoder2(encoded_lidar)
      central_mems[2] = self.conv_gru_lidar2(encoded_lidar, central_mems[2])

      # We apply the third encoder and give it to the convGRU to update its state
      encoded_lidar = self.lidar_encoder3(encoded_lidar)
      central_mems[3] = self.conv_gru_lidar3(encoded_lidar, central_mems[3])

    # PART 3: ENCODING THE EVENT DATA AND UPDATING THE MEMORIES (IF AVAILABLE)
    if event_input is not None:
      # We first apply the head, to go from M layers to 32, and give the result to the top level
      # convGRU to update its state
      encoded_event = self.event_head(event_input)
      central_mems[0] = self.conv_gru_events0(encoded_event, central_mems[0])

      # We apply the first encoder and give it to the convGRU to update its state
      encoded_event = self.event_encoder1(encoded_event)
      central_mems[1] = self.conv_gru_events1(encoded_event, central_mems[1])

      # We apply the second encoder and give it to the convGRU to update its state
      encoded_event = self.event_encoder2(encoded_event)
      central_mems[2] = self.conv_gru_events2(encoded_event, central_mems[2])

      # We apply the third encoder and give it to the convGRU to update its state
      encoded_event = self.event_encoder3(encoded_event)
      central_mems[3] = self.conv_gru_events3(encoded_event, central_mems[3])

    # PART 4: DECODING THE MEMORIES TO PREDICT THE FINAL DEPTH VALUES
    # The initial input for the decoding is the fourth and last central memory, on which we apply
    # the two residual blocks
    pred = self.residual_block1(central_mems[3])
    pred = self.residual_block2(pred)

    # We decompose the third central memory in two parts: a "prediction" part and an "upsampling
    # mask" part
    central_mem_2_pred = central_mems[2][:, :128, :, :]
    central_mem_2_mask = central_mems[2][:, 128:, :, :]

    # We apply the first decoder, guided by the upsampling mask
    pred = self.decoder1(pred, central_mem_2_mask)

    # We concatenate the prediction from the third central memory, and apply the convolution to go
    # from 256 to 128 channels
    pred = torch.concat((pred, central_mem_2_pred), dim=1)
    pred = self.conv_concat1(pred)

    # We decompose the second central memory in two parts: a "prediction" part and an "upsampling
    # mask" part
    central_mem_1_pred = central_mems[1][:, :64, :, :]
    central_mem_1_mask = central_mems[1][:, 64:, :, :]

    # We apply the second decoder, guided by the upsampling mask
    pred = self.decoder2(pred, central_mem_1_mask)

    # We concatenate the prediction from the second  central memory, and apply the convolution to go
    # from 128 to 64 channels
    pred = torch.concat((pred, central_mem_1_pred), dim=1)
    pred = self.conv_concat2(pred)

    # We decompose the first central memory in two parts: a "prediction" part and an "upsampling
    # mask" part
    central_mem_0_pred = central_mems[0][:, :32, :, :]
    central_mem_0_mask = central_mems[0][:, 32:, :, :]

    # We apply the last decoder, guided by the upsampling mask
    pred = self.decoder3(pred, central_mem_0_mask)

    # We concatenate the prediction from the first central memory, and apply the convolution to go
    # from 64 to 32 channels
    pred = torch.concat((pred, central_mem_0_pred), dim=1)
    pred = self.conv_concat3(pred)

    # We finish by applying the prediction layer
    pred = self.prediction_layer(pred)

    # We return the prediction and the updated central memories
    return pred, central_mems


import time

# =========================================================================
# 测试脚本 (Main)
# =========================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running ALED on {device}...")

    # 1. 配置模型参数
    # ALED 的参数非常简单，只需要输入输出通道数
    model = ALED(
        lidar_channels=1,
        event_channels=4,
        out_channels=1
    ).to(device)

    # 2. 模拟数据
    # ⚠️ 注意: ALED 包含 3 层下采样 (stride=2)，长宽必须是 2^3 = 8 的倍数。
    # 260x346 无法被 8 整除，解码时 concat 会报错。这里采用最接近的 256x352
    h, w = 720, 1280
    x_lidar = torch.randn(1, 1, h, w).to(device)
    x_event = torch.randn(1, 4, h, w).to(device)

    # 初始状态下没有记忆，设为 None
    central_mems = None

    print(f"📦 Input: LiDAR {x_lidar.shape}, Event {x_event.shape}")

    # ==========================================
    # A. 计算参数量 (Parameters)
    # ==========================================
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数量 (Trainable Params): {total_params / 1e6:.2f} M")

    # ==========================================
    # B. 计算计算量 (FLOPs / MACs)
    # ==========================================
    try:
        from thop import profile
        print("🧮 正在计算 FLOPs (可能会花费几秒钟)...")
        # 传入 x_lidar, x_event 以及 central_mems (初始为None)
        macs, params = profile(model, inputs=(x_lidar, x_event, central_mems), verbose=False)
        print(f"🧮 计算量 (MACs): {macs / 1e9:.2f} G")
        print(f"🧮 预估 FLOPs: {(macs * 2) / 1e9:.2f} G")
    except ImportError:
        print("【提示】未检测到 thop 库，跳过 FLOPs 计算。建议运行 `pip install thop` 后重试。")
    except Exception as e:
        print(f"⚠️ thop 计算失败: {e} (可能是 thop 对 list 类型的 memory 解析有问题，跳过)")

    # ==========================================
    # C. 计算单次运行时间 (Inference Time)
    # ==========================================
    try:
        print("⏱️ 开始测速 (包含 Warm-up)...")
        model.eval()
        with torch.no_grad():
            # 预热 (Warm-up): 循环传递记忆以模拟连续时间步
            mems = None
            for _ in range(5):
                _, mems = model(x_lidar, x_event, mems)

            # 严格同步 GPU 开始时间
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # 实际测速运行 (继续使用上一帧的记忆)
            pred, mems = model(x_lidar, x_event, mems)

            # 严格同步 GPU 结束时间
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

        infer_time_ms = (end_time - start_time) * 1000
        print(f"⚡ 单次前向传播时间: {infer_time_ms:.2f} ms (约 {1000 / infer_time_ms:.1f} FPS)")

        # 验证输出
        print(f"✅ Success! Depth Output Shape: {pred.shape}")

        if pred.shape[-2:] == (h, w):
            print("🎉 Resolution Match!")
        else:
            print(f"⚠️ Resolution Mismatch: Expected {(h, w)}, Got {pred.shape[-2:]}")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

    def measure_memory_cost(model, dummy_inputs):
      # 1. 清空之前的显存缓存
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()

      # 2. 记录基础显存（模型权重 + CUDA上下文）
      base_memory = torch.cuda.memory_allocated() / (1024 ** 3)

      # 3. 运行一次前向传播
      model.eval()
      with torch.no_grad():
        _ = model(*dummy_inputs)

      # 4. 获取峰值显存 (Peak Memory)
      peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

      print(f"📦 基础显存 (模型权重): {base_memory:.3f} GB")
      print(f"📈 峰值显存 (GPU Memory Cost): {peak_memory:.3f} GB")

    # 使用示例 (假设使用你之前的 ALED 模型变量):
    measure_memory_cost(model, (x_lidar, x_event, central_mems))

if __name__ == "__main__":
    main()