import torch
import torch.nn.functional as F
from typing import Literal, Tuple, Optional
import os, sys
import math
from einops import rearrange

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(os.path.join(ROOT, "RVT"))

from s5.jax_func import associative_scan
from s5.s5_init import *

# Runtime functions


@torch.jit.script
def binary_operator(
    q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]
):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    # return A_j * A_i, A_j * b_i + b_j
    return A_j * A_i, torch.addcmul(b_j, A_j, b_i)


def apply_ssm(
    Lambda_bars: torch.Tensor,
    B_bars,
    C_tilde,
    D,
    input_sequence,
    prev_state,
    bidir: bool = False,
):
    B_bars = as_complex(B_bars)
    C_tilde = as_complex(C_tilde)
    Lambda_bars = as_complex(Lambda_bars)

    cinput_sequence = input_sequence.type(
        Lambda_bars.dtype
    )  # Cast to correct complex type

    if B_bars.ndim == 3:
        # Dynamic timesteps (significantly more expensive)
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
    else:
        # Static timesteps
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

    if Lambda_bars.ndim == 1:  # Repeat for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

    Lambda_bars[0] = Lambda_bars[0] * prev_state

    _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))

    if bidir:
        _, xs2 = associative_scan(
            binary_operator, (Lambda_bars, Bu_elements), reverse=True
        )
        xs = torch.cat((xs, xs2), axis=-1)

    Du = torch.vmap(lambda u: D * u)(input_sequence)
    # TODO: the last element of xs (non-bidir) is the hidden state, allow returning it
    return torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du, xs[-1]


def apply_ssm_liquid(
    Lambda_bars, B_bars, C_tilde, D, input_sequence, bidir: bool = False
):
    """Liquid time constant SSM \u00e1 la dynamical systems given in Eq. 8 of
    https://arxiv.org/abs/2209.12951"""
    cinput_sequence = input_sequence.type(
        Lambda_bars.dtype
    )  # Cast to correct complex type

    if B_bars.ndim == 3:
        # Dynamic timesteps (significantly more expensive)
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
    else:
        # Static timesteps
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

    if Lambda_bars.ndim == 1:  # Repeat for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

    _, xs = associative_scan(binary_operator, (Lambda_bars + Bu_elements, Bu_elements))

    if bidir:
        _, xs2 = associative_scan(
            binary_operator, (Lambda_bars, Bu_elements), reverse=True
        )
        xs = torch.cat((xs, xs2), axis=-1)

    Du = torch.vmap(lambda u: D * u)(input_sequence)
    return torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Lambda = torch.view_as_complex(Lambda)

    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde

    Lambda_bar = torch.view_as_real(Lambda_bar)
    B_bar = torch.view_as_real(B_bar)

    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    # Identity = torch.ones(Lambda.shape[0], device=Lambda.device) # (replaced by -1)
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - 1))[..., None] * B_tilde
    return Lambda_bar, B_bar


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt


Initialization = Literal["dense_columns", "dense", "factorized"]


class S5SSM(torch.nn.Module):
    def __init__(
        self,
        lambdaInit: torch.Tensor,
        V: torch.Tensor,
        Vinv: torch.Tensor,
        h: int,
        p: int,
        dt_min: float,
        dt_max: float,
        liquid: bool = False,
        factor_rank: Optional[int] = None,
        discretization: Literal["zoh", "bilinear"] = "bilinear",
        bcInit: Initialization = "factorized",
        degree: int = 1,
        bidir: bool = False,
        step_scale: float = 1.0,
        bandlimit: Optional[float] = None,
    ):
        """The S5 SSM
        Args:
            lambdaInit  (complex64): Initial diagonal state matrix       (P,)
            V           (complex64): Eigenvectors used for init          (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init  (P,P)
            h           (int32):     Number of features of input seq
            p           (int32):     state size
            k           (int32):     rank of low-rank factorization (if used)
            bcInit      (string):    Specifies How B and C are initialized
                        Options: [factorized: low-rank factorization,
                                dense: dense matrix drawn from Lecun_normal]
                                dense_columns: dense matrix where the columns
                                of B and the rows of C are each drawn from Lecun_normal
                                separately (i.e. different fan-in then the dense option).
                                We found this initialization to be helpful for Pathx.
            discretization: (string) Specifies discretization method
                            options: [zoh: zero-order hold method,
                                    bilinear: bilinear transform]
            liquid:         (bool): use liquid_ssm from LiquidS4
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_scale:  (float32): allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
        """
        super().__init__()
        self.Lambda = torch.nn.Parameter(torch.view_as_real(lambdaInit))
        self.degree = degree
        self.liquid = liquid
        self.bcInit = bcInit
        self.bidir = bidir
        self.bandlimit = bandlimit

        cp = p
        if self.bidir:
            cp = cp*2

        match bcInit:
            case "complex_normal":
                self.C = torch.nn.Parameter(
                    torch.normal(0, 0.5**0.5, (h, cp), dtype=torch.complex64)
                )
                self.B = torch.nn.Parameter(
                    init_VinvB(lecun_normal(), Vinv)((p, h), torch.float)
                )
            case "dense_columns" | "dense":
                if bcInit == "dense_columns":
                    B_eigen_init = init_columnwise_VinvB
                    B_init = init_columnwise_B
                    C_init = init_rowwise_C
                elif bcInit == "dense":
                    B_eigen_init = init_VinvB
                    B_init = C_init = lecun_normal()
                # TODO: make init_*VinvB all a the same interface
                self.B = torch.nn.Parameter(
                    B_eigen_init(B_init, Vinv)((p, h), torch.float)
                )
                if self.bidir:
                    C = torch.cat(
                        [init_CV(C_init, (h, p), V), init_CV(C_init, (h, p), V)],
                        axis=-1,
                    )
                else:
                    C = init_CV(C_init, (h, p), V)
                self.C = torch.nn.Parameter(torch.view_as_real(C))
            case _:
                raise NotImplementedError(f"BC_init method {bcInit} not implemented")

        # Initialize feedthrough (D) matrix
        self.D = torch.nn.Parameter(
            torch.rand(
                h,
            )
        )
        self.log_step = torch.nn.Parameter(init_log_steps(p, dt_min, dt_max))
        match discretization:
            case "zoh":
                self.discretize = discretize_zoh
            case "bilinear":
                self.discretize = discretize_bilinear
            case _:
                raise ValueError(f"Unknown discretization {discretization}")

        if self.bandlimit is not None:
            step = step_scale * torch.exp(self.log_step)

            freqs = step / step_scale * self.Lambda[:, 1].abs() / (2 * math.pi)
            mask = torch.where(freqs < bandlimit * 0.5, 1, 0)  # (64, )
            self.C = torch.nn.Parameter(
                torch.view_as_real(torch.view_as_complex(self.C) * mask)
            )

    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        _, C_tilde = self.get_BC_tilde()

        return torch.zeros((*batch_shape, C_tilde.shape[-2]))

    def get_BC_tilde(self):
        match self.bcInit:
            case "dense_columns" | "dense" | "complex_normal":
                B_tilde = as_complex(self.B)
                C_tilde = self.C
            case "factorized":
                B_tilde = self.BP @ self.BH.T
                C_tilde = self.CH.T @ self.CP
        return B_tilde, C_tilde

    def forward_rnn(self, signal, prev_state, step_scale: float | torch.Tensor = 1.0):
        assert not self.bidir, "Can't use bidirectional when manually stepping"
        B_tilde, C_tilde = self.get_BC_tilde()
        step = step_scale * torch.exp(self.log_step)
        Lambda_bar, B_bar = self.discretize(self.Lambda, B_tilde, step)
        if self.degree != 1:
            assert (
                B_bar.shape[-2] == B_bar.shape[-1]
            ), "higher-order input operators must be full-rank"
            B_bar = B_bar**self.degree

        if not torch.is_tensor(step_scale) or step_scale.ndim == 0:
            step_scale = torch.ones(signal.shape[-2], device=signal.device) * step_scale
        step = step_scale[:, None] * torch.exp(self.log_step)
        # https://arxiv.org/abs/2209.12951v1, Eq. 9
        Bu = B_bar @ signal
        if self.liquid:
            Lambda_bar = Lambda_bar + Bu
        # https://arxiv.org/abs/2208.04933v2, Eq. 2
        x = Lambda_bar * prev_state + Bu
        y = (C_tilde @ x + self.D * signal).real
        return y, x

    # NOTE: can only be used as RNN OR S5(MIMO) (no mixing)
    def forward(self, signal, prev_state, step_scale: float | torch.Tensor = 1.0):
        B_tilde, C_tilde = self.get_BC_tilde()
        step = step_scale * torch.exp(self.log_step)
        Lambda_bar, B_bar = self.discretize(self.Lambda, B_tilde, step)
        if self.degree != 1:
            assert (
                B_bar.shape[-2] == B_bar.shape[-1]
            ), "higher-order input operators must be full-rank"
            B_bar =  B_bar ** self.degree

        if not torch.is_tensor(step_scale) or step_scale.ndim == 0:
            # step_scale = torch.ones(signal.shape[-2], device=signal.device) * step_scale
            step = step_scale * torch.exp(self.log_step)
        else:
            # TODO: This is very expensive due to individual steps being multiplied by B_tilde in self.discretize
            step = step_scale[:, None] * torch.exp(self.log_step)

        Lambda_bars, B_bars = self.discretize(self.Lambda, B_tilde, step)
        # Lambda_bars, B_bars = torch.vmap(self.discretize, (None, None, 0))(self.Lambda, B_tilde, step)
        forward = apply_ssm_liquid if self.liquid else apply_ssm
        return forward(
            Lambda_bars, B_bars, C_tilde, self.D, signal, prev_state, bidir=self.bidir
        )


class S5(torch.nn.Module):
    def __init__(
        self,
        width: int,
        state_width: Optional[int] = None,
        factor_rank: Optional[int] = None,
        block_count: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        liquid: bool = False,
        degree: int = 1,
        bidir: bool = False,
        bcInit: Optional[Initialization] = None,
        bandlimit: Optional[float] = None,
    ):
        super().__init__()
        state_width = state_width or width
        assert (
            state_width % block_count == 0
        ), "block_count should be a factor of state_width"

        block_size = state_width // block_count
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        Vinv = V.conj().T
        Lambda, B, V, B_orig, Vinv = map(
            lambda v: torch.tensor(v, dtype=torch.complex64),
            (Lambda, B, V, B_orig, Vinv),
        )
        if block_count > 1:
            Lambda = Lambda[:block_size]
            V = V[:, :block_size]
            Lambda = (Lambda * torch.ones((block_count, block_size))).ravel()
            V = torch.block_diag(*([V] * block_count))
            Vinv = torch.block_diag(*([Vinv] * block_count))

        assert bool(factor_rank) != bool(
            bcInit != "factorized"
        ), "Can't have `bcInit != factorized` and `factor_rank` defined"
        bc_init = "factorized" if factor_rank is not None else (bcInit or "dense")
        self.width = width
        self.seq = S5SSM(
            Lambda,
            V,
            Vinv,
            width,
            state_width,
            dt_min,
            dt_max,
            factor_rank=factor_rank,
            bcInit=bc_init,
            liquid=liquid,
            degree=degree,
            bidir=bidir,
            bandlimit=bandlimit,
        )

    def initial_state(self, batch_size: Optional[int] = None):
        return self.seq.initial_state(batch_size)

    def forward(self, signal, prev_state, step_scale: float | torch.Tensor = 1.0):
        # NOTE: step_scale can be float | Tensor[batch] | Tensor[batch, seq]
        if not torch.is_tensor(step_scale):
            # Duplicate across batchdim
            step_scale = torch.ones(signal.shape[0], device=signal.device) * step_scale

        return torch.vmap(lambda s, ps, ss: self.seq(s, prev_state=ps, step_scale=ss))(
            signal, prev_state, step_scale
        )


class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class S5Block(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        state_dim: int,
        bidir: bool,
        block_count: int = 1,
        liquid: bool = False,
        degree: int = 1,
        factor_rank: int | None = None,
        bcInit: Optional[Initialization] = None,
        ff_mult: float = 1.0,
        glu: bool = True,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        bandlimit: Optional[float] = None,
    ):
        super().__init__()
        self.s5 = S5(
            dim,
            state_width=state_dim,
            bidir=bidir,
            block_count=block_count,
            liquid=liquid,
            degree=degree,
            factor_rank=factor_rank,
            bcInit=bcInit,
            bandlimit=bandlimit,
        )
        self.attn_norm = torch.nn.LayerNorm(dim)
        self.attn_dropout = torch.nn.Dropout(p=attn_dropout)
        self.geglu = GEGLU() if glu else None
        self.ff_enc = torch.nn.Linear(dim, int(dim * ff_mult) * (1 + glu), bias=False)
        self.ff_dec = torch.nn.Linear(int(dim * ff_mult), dim, bias=False)
        self.ff_norm = torch.nn.LayerNorm(dim)
        self.ff_dropout = torch.nn.Dropout(p=ff_dropout)
    #
    # def forward(self, x, states):
    #     fx = x
    #     res = fx.clone()
    #     x, new_state = self.s5(fx, states)
    #
    #     x = F.gelu(x) + res
    #     x = self.attn_dropout(x)
    #     fx = x
    #     res = fx.clone()
    #     x = self.ff_enc(fx)
    #     if self.geglu is not None:
    #         x = self.geglu(x)
    #     x = self.ff_dec(x) + res
    #     x = self.ff_dropout(x)
    #     return x, new_state

    def forward(self, x, states):
        # Standard transfomer-style block with GEGLU/Pre-LayerNorm
        fx = self.attn_norm(x)
        res = fx.clone()
        x, new_state = self.s5(fx, states)
        x = F.gelu(x) + res
        x = self.attn_dropout(x)
        fx = self.ff_norm(x)
        res = fx.clone()
        x = self.ff_enc(fx)
        if self.geglu is not None:
            x = self.geglu(x)
        x = self.ff_dec(x) + res
        x = self.ff_dropout(
            x
        )  # TODO: test if should be placed inbetween ff or after ff
        return x, new_state

if __name__ == "__main__":
    # import lovely_tensors as lt
    #
    # lt.monkey_patch()
    #
    # def tensor_stats(t: torch.Tensor):  # Clone of lovely_tensors for complex support
    #     return f"tensor[{t.shape}] n={t.shape.numel()}, u={t.mean()}, s={round(t.std().item(), 3)} var={round(t.var().item(), 3)}\n"

    # x = torch.rand([2, 256, 32]).cuda()
    # model = S5(32, 32, factor_rank=None).cuda()
    # print("B", tensor_stats(model.seq.B.data))
    # print("C", tensor_stats(model.seq.C.data))
    # # print('B', tensor_stats(model.seq.BH.data), tensor_stats(model.seq.BP.data))
    # # print('C', tensor_stats(model.seq.CH.data), tensor_stats(model.seq.CP.data))
    # # FIXME: unstable initialization
    # state = model.initial_state(256)
    # res = model(x, prev_state=state)
    # print(res.shape, res.dtype, res)

    # Example : (B, C, H, W) inputs

    # x = torch.randn([2, 5, 224, 224]).cuda()
    # states = torch.zeros([2, 1, 224, 224]).cuda()
    # x = rearrange(x, 'B C H W -> C B 1 H W')  # C=L

    x = torch.randn([5, 2, 4, 224, 224]).cuda()
    states = torch.zeros([2, 4, 224, 224]).cuda()

    batch_size = x.shape[1]
    sequence_length = x.shape[0]
    h, w = x.shape[3], x.shape[4]
    x = rearrange(x, "L B C H W -> (B H W) L C", L=sequence_length)
    states = rearrange(states, "B C H W -> (B H W) C")
    print(states.shape)
    print(x.shape)
    model = S5Block(4, 4, False).cuda()
    x, states = model(x, states)
    x = rearrange(
        x, "(B H W) L C -> L B C H W", B=batch_size, H=int(h), W=int(w)
    )
    states = rearrange(states, "(B H W) C -> B C H W", H=h, W=w)

# if __name__ == "__main__":
#     import torch
#
#     # 设置随机种子以便复现
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(42)
#
#     print("=== Testing S5Block Forward Pass ===")
#
#     # 1. 定义超参数配置
#     # 注意：根据代码约束，state_dim 必须能被 block_count 整除
#     config = {
#         "dim": 96,  # 输入特征维度 (Feature Dimension)
#         "state_dim": 96,  # S5 内部状态维度 (Hidden State Dimension)
#         "bidir": False,  # 是否双向 (Bidirectional)
#         "block_count": 4,  # 并行块数 (Block Count for DPLR)
#         "batch_size": 2,  # 批次大小
#         "seq_len": 128,  # 序列长度 (Sequence Length)
#     }
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Running on device: {device}")
#
#     try:
#         # 2. 实例化 S5Block 模型
#         model = S5Block(
#             dim=config["dim"],
#             state_dim=config["state_dim"],
#             bidir=config["bidir"],
#             block_count=config["block_count"],
#             # 其他参数使用默认值
#             liquid=False,
#             degree=1
#         ).to(device)
#
#         print("\nModel instantiated successfully.")
#         print(
#             f"Model Structure: S5Block(dim={config['dim']}, state_dim={config['state_dim']}, blocks={config['block_count']})")
#
#         # 3. 创建随机输入数据
#         # 输入形状应为 (Batch, Sequence Length, Dimension)
#         x = torch.randn(
#             config["batch_size"],
#             config["seq_len"],
#             config["dim"]
#         ).to(device)
#
#         # 4. 初始化状态 (prev_state)
#         # S5Block 内部封装了 S5，S5 提供了 initial_state 方法
#         # 我们需要一个初始的 hidden state 来开始递归
#         prev_state = model.s5.initial_state(batch_size=config["batch_size"]).to(device)
#
#         print(f"\n--- Inputs ---")
#         print(f"Input Data (x):      {x.shape}  [Batch, Seq, Dim]")
#         print(f"Initial State:       {prev_state.shape}  [Batch, State_Dim_Complex]")
#
#         # 5. 执行前向传播
#         # forward 接收 (input, prev_state)
#         output, new_state = model(x, prev_state)
#
#         # 6. 验证输出结果
#         print(f"\n--- Outputs ---")
#         print(f"Output Data (y):     {output.shape}  [Batch, Seq, Dim]")
#         print(f"New State:           {new_state.shape}  [Batch, State_Dim_Complex]")
#
#         # 维度检查
#         assert output.shape == (config["batch_size"], config["seq_len"], config["dim"]), \
#             f"Output shape mismatch! Expected {(config['batch_size'], config['seq_len'], config['dim'])}, got {output.shape}"
#
#         # 检查是否包含 NaN (梯度爆炸常见问题)
#         if torch.isnan(output).any():
#             print("\n⚠️ Warning: Output contains NaN values!")
#         else:
#             print("\n✅ Test Passed: Forward pass successful with valid shapes.")
#
#     except NameError as e:
#         print(f"\n❌ Error: 缺少必要的依赖函数。")
#         print(f"详细错误: {e}")
#         print("提示: 请确保 'make_DPLR_HiPPO', 'init_VinvB' 等函数已在当前上下文中定义或导入。")
#     except Exception as e:
#         print(f"\n❌ Runtime Error: {e}")
#         import traceback
#
#         traceback.print_exc()