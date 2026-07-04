import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class ConvEncodingHead(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size

        # --- 原始代码 (导致报错) ---
        # self.conv1 = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        # self.act = nn.GELU()
        # stride_2 = patch_size // 2
        # self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=stride_2, padding=1)
        # self.bn2 = nn.BatchNorm2d(embed_dim)  <-- 罪魁祸首 (Size 96)

        # --- 修改后 (使用 GroupNorm) ---
        self.conv1 = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1)
        # GroupNorm(组数, 通道数)。通常组数设为 8 或 16。
        # 注意 GroupNorm 输入顺序是 (num_groups, num_channels)
        self.gn1 = nn.GroupNorm(8, embed_dim // 2)

        self.act = nn.GELU()

        stride_2 = patch_size // 2
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=stride_2, padding=1)
        # 替换 bn2
        self.gn2 = nn.GroupNorm(8, embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (B, C, H, W)

        # 修改 bn1 -> gn1
        x_skip = self.act(self.gn1(self.conv1(x)))

        # 修改 bn2 -> gn2
        x_enc = self.gn2(self.conv2(x_skip))

        return x_enc, x_skip


class ConvDecodingHead(nn.Module):
    """
    将 Patch 特征上采样回全分辨率深度图。
    """

    def __init__(self, embed_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        # PixelShuffle 上采样效率高且无棋盘格效应
        self.up_conv1 = nn.Conv2d(embed_dim, 64 * patch_size * patch_size, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, out_channels, kernel_size=1)  # 最终深度回归
        )

    def forward(self, x: Tensor, skip_evt: Tensor, h_p: int, w_p: int) -> Tensor:
        # x: (B, L, D) -> 需要 Reshape 回 2D
        B, L, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, h_p, w_p)

        # 上采样: (B, 64, H, W)
        x = self.pixel_shuffle(self.up_conv1(x))

        # 这里为了代码简洁未融合 skip_evt，实际项目中可以将 skip_evt concat 进来
        out = self.out_conv(x)
        return out


class PytorchSSM(nn.Module):
    """
    [核心创新] PyTorch 原生实现的简化版选择性状态空间模型 (Selective SSM)。
    数学原理近似 Mamba/S4，支持时序记忆 h_t 的传递。
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state  # 记忆状态维度

        # 投影层：同时生成内容(u)、门控(g)和遗忘因子(dt)
        self.in_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        # A 参数：状态衰减率 (模拟连续系统的离散化)
        # self.A_log = nn.Parameter(torch.log(torch.randn(dim, d_state).abs() + 1e-3))
        self.D = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, prev_state: Tensor | None) -> tuple[Tensor, Tensor]:
        """
        x: (B, L, D) - 输入序列
        prev_state: (B, L, D) - 上一时刻的记忆状态 (简化版，假设 Patch 间独立传播)
        """
        B, L, D = x.shape

        if prev_state is None:
            prev_state = torch.zeros(B, L, D, device=x.device)

        # 1. 投影与切分
        proj = self.in_proj(x)
        u, g, dt = proj.chunk(3, dim=-1)

        # 2. 激活函数
        u = F.silu(u)  # 输入内容
        g = F.silu(g)  # 输出门控
        dt = torch.sigmoid(dt)  # 遗忘门/时间步长 (0~1)

        # 3. 状态更新 (Recurrence)
        # h_t = (1 - dt) * h_{t-1} + dt * u
        # dt 越大，越倾向于遗忘过去，接受新输入
        current_state = (1 - dt) * prev_state + dt * u

        # 4. 输出计算
        # y = state * gate
        y = current_state * g
        out = self.out_proj(y)

        return out, current_state


class VSSBlock(nn.Module):
    """
    Visual State Space Block.
    类似于 Transformer Block，但核心是 SSM 而不是 Attention。
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = PytorchSSM(dim, d_state)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x: Tensor, prev_state: Tensor | None) -> tuple[Tensor, Tensor]:
        # x: (B, L, D)
        residual = x
        x_norm = self.norm(x)

        # SSM 运算
        x_ssm, new_state = self.ssm(x_norm, prev_state)

        # 第一次残差
        x_mid = residual + x_ssm

        # FFN + 第二次残差
        x_out = x_mid + self.ffn(x_mid)

        return x_out, new_state

#----------------------------------------------------------------------
class ParallaxHead(nn.Module):
    """
    [角度二: Teacher/Student]
    这是一个轻量级的头，用于强制 Event 分支独立学习 "运动-深度" 映射。
    """
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, patch_size * patch_size) # 直接回归像素值
        )
        self.patch_size = patch_size

    def forward(self, x: Tensor, h_p: int, w_p: int) -> Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        out = self.head(x) # (B, L, P*P)
        # Reshape back to image
        out = out.permute(0, 2, 1).reshape(B, self.patch_size*self.patch_size, h_p, w_p)
        out = nn.functional.pixel_shuffle(out, self.patch_size) # (B, 1, H, W)
        return out


# =========================================================================
# 🔥 [核心创新: Angle 1 - Master/Slave Densification]
# =========================================================================
class StructureGuidedSSM(nn.Module):
    """
    结构引导致密化 SSM。

    Roles:
    - Master (Content): LiDAR features. 它是要被传播的内容。
    - Slave (Structure): Event features. 它定义传播规则 (A, Delta)。

    Logic:
    LiDAR 提供了稀疏的 u (Input)。
    Events 提供了稠密的 Delta (遗忘率/步长) 和 A (状态转移)。

    Effect:
    Events 充当了“导管”。在没有 Events 纹理的地方，Delta 小，LiDAR 状态保持并扩散（致密化）。
    在有 Events 边缘的地方，Delta 大，状态快速更新（保持边界）。
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state

        # 1. Master Channel (LiDAR): 只提供内容 u
        self.proj_content = nn.Linear(dim, dim)

        # 2. Slave Channel (Events): 提供结构参数 dt, g
        # 注意：这里我们让 Event 显式预测 dt (Time-scale/Gate)
        self.proj_structure = nn.Linear(dim, dim * 2)

        self.out_proj = nn.Linear(dim, dim)

        # A 也是可以被 Structure 调节的，但为稳定性这里保持为可学习参数
        # self.A_log = nn.Parameter(torch.log(torch.randn(dim, d_state).abs() + 1e-3))

    def forward(self, content_feat: Tensor, structure_feat: Tensor, prev_state: Tensor | None) -> tuple[Tensor, Tensor]:
        B, L, D = content_feat.shape
        if prev_state is None:
            prev_state = torch.zeros(B, L, D, device=content_feat.device)

        # Master: 获取 LiDAR 的值
        u = self.proj_content(content_feat)
        u = F.silu(u)

        # Slave: 获取 Events 的结构信息
        # dt (delta): 决定了当前像素是应该 "保持上一刻记忆(致密化)" 还是 "接受新输入(边缘)"
        # g (gate): 输出门控
        structure_params = self.proj_structure(structure_feat)
        g, dt = structure_params.chunk(2, dim=-1)

        dt = torch.sigmoid(dt)
        g = F.silu(g)

        # [Densification Logic]
        # h_t = (1 - dt) * h_{t-1} + dt * u
        # 如果 dt -> 0 (Event 认为这里是平滑区域), h_t ≈ h_{t-1} (LiDAR 深度值被保留并传递下去)
        # 如果 dt -> 1 (Event 认为这里是边缘), h_t ≈ u (更新为当前特征)
        current_state = (1 - dt) * prev_state + dt * u

        # Output
        y = current_state * g
        out = self.out_proj(y)

        return out, current_state


class StructureGuidedBlock(nn.Module):
    """
    封装了 StructureGuidedSSM 的残差块。
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm_content = nn.LayerNorm(dim)
        self.norm_structure = nn.LayerNorm(dim)

        self.guided_ssm = StructureGuidedSSM(dim, d_state)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, content: Tensor, structure: Tensor, prev_state: Tensor | None) -> tuple[Tensor, Tensor]:
        # Master (LiDAR) 是残差流的主体
        residual = content

        # Slave (Structure) 引导 Master (Content) 的状态更新
        x_densified, new_state = self.guided_ssm(
            self.norm_content(content),
            self.norm_structure(structure),
            prev_state
        )

        x_mid = residual + x_densified
        return x_mid + self.ffn(x_mid), new_state


class MultiHeadStructureGuidedSSM(nn.Module):
    """
    [升级版] 多头结构引导致密化 SSM (MH-SG-SSM)

    模仿 Transformer 的多头机制，将特征空间划分为 H 个子空间。
    不同的头可以学习不同的致密化速率（有的头关注高频边缘，有的头关注低频平面）。

    Args:
        dim (int): 特征总维度 (D)
        num_heads (int): 头数 (H)
        d_state (int): SSM 状态维度 (N)
    """

    def __init__(self, dim: int, num_heads: int = 4, d_state: int = 16):
        super().__init__()
        assert dim % num_heads == 0, f"Dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.d_state = d_state

        # --- 1. Master Projection (Content) ---
        # 投影后将 reshape 为 (B, L, H, Head_Dim)
        self.proj_content = nn.Linear(dim, dim)

        # --- 2. Slave Projection (Structure/Guide) ---
        # Events 同样投影为多头，每个头生成独立的 dt 和 g
        # 输出维度是 dim * 2 (dt + g)
        self.proj_structure = nn.Linear(dim, dim * 2)

        # --- 3. Output Projection (Mixing) ---
        # 这一步至关重要：将所有头的信息融合
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, content_feat: Tensor, structure_feat: Tensor, prev_state: Tensor | None) -> tuple[Tensor, Tensor]:
        """
        content_feat: (B, L, D) - LiDAR
        structure_feat: (B, L, D) - Event
        prev_state: (B, L, D) - 实际上我们内部会把它视为 (B, L, H, Head_Dim)
        """
        B, L, D = content_feat.shape
        H = self.num_heads
        D_head = self.head_dim

        # 初始化状态 (如果第一帧)
        if prev_state is None:
            prev_state = torch.zeros(B, L, D, device=content_feat.device)

        # ==========================================================
        # Step 1: 多头投影与重塑 (Multi-Head Projection & Reshape)
        # ==========================================================

        # Master (LiDAR): (B, L, D) -> (B, L, H, D_head)
        u = self.proj_content(content_feat)
        u = u.view(B, L, H, D_head)
        u = F.silu(u)

        # Slave (Event): (B, L, D) -> (B, L, H, 2*D_head)
        structure_params = self.proj_structure(structure_feat)
        structure_params = structure_params.view(B, L, H, 2 * D_head)

        # 切分 g (gate) 和 dt (delta)
        # 现在的 dt 是针对每个头的！这意味着不同头有不同的“遗忘率”
        g, dt = structure_params.chunk(2, dim=-1)  # (B, L, H, D_head)

        dt = torch.sigmoid(dt)  # (0, 1)
        g = F.silu(g)

        # 重塑 prev_state 以匹配多头计算
        prev_state_view = prev_state.view(B, L, H, D_head)

        # ==========================================================
        # Step 2: 独立的多头致密化 (Independent Densification)
        # ==========================================================

        # h_t = (1 - dt) * h_{t-1} + dt * u
        # 这里的运算是在 (B, L, H, D_head) 维度上进行的
        # 物理意义：
        # Head 1 可能 dt -> 1 (快速更新，捕捉 LiDAR 跳变)
        # Head 2 可能 dt -> 0 (保持历史，填补 LiDAR 空洞)
        current_state_view = (1 - dt) * prev_state_view + dt * u

        # ==========================================================
        # Step 3: 门控与融合 (Gating & Fusion)
        # ==========================================================

        # Apply output gate per head
        y = current_state_view * g

        # 还原形状: (B, L, H, D_head) -> (B, L, D)
        # 这一步相当于 Concat Heads
        y = y.view(B, L, D)
        current_state = current_state_view.view(B, L, D)

        # Final Mixing: 让不同头的信息交流
        out = self.out_proj(y)

        return out, current_state


class MultiHeadStructureGuidedBlock(nn.Module):
    """
    封装了 MultiHeadStructureGuidedSSM 的残差块。
    """

    def __init__(self, dim: int, num_heads: int = 4, d_state: int = 16):
        super().__init__()
        self.norm_content = nn.LayerNorm(dim)
        self.norm_structure = nn.LayerNorm(dim)

        # 使用多头版 SSM
        self.guided_ssm = MultiHeadStructureGuidedSSM(dim, num_heads, d_state)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, content: Tensor, structure: Tensor, prev_state: Tensor | None) -> tuple[Tensor, Tensor]:
        residual = content

        x_densified, new_state = self.guided_ssm(
            self.norm_content(content),
            self.norm_structure(structure),
            prev_state
        )

        x_mid = residual + x_densified
        return x_mid + self.ffn(x_mid), new_state



