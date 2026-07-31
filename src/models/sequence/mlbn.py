"""Flipped-GARI sequence encoder and its reversal-aligned building blocks.

Some internal class names are retained for compatibility with existing
checkpoint state dictionaries. Public configurations and documentation use
the Flipped-GARI name.
"""

import math

import mamba_ssm as mamba
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm.modules.mamba_simple import Mamba as Mamba1
except ImportError:
    Mamba1 = None


def _make_mamba_mixer(input_dim, d_state, d_conv, expand, head_dim):
    """Create the available Mamba mixer while preserving checkpoint compatibility."""
    if hasattr(mamba, "Mamba2"):
        return mamba.Mamba2(input_dim, d_state, d_conv, expand, head_dim)
    if Mamba1 is None:
        raise ImportError("Flipped-GARI requires mamba_ssm.Mamba2 or mamba_ssm.modules.mamba_simple.Mamba.")
    return Mamba1(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand)


def _encoder_drop_path_rates(num_blocks: int, max_rate: float):
    """Linear stochastic-depth schedule from zero to ``max_rate``."""
    if num_blocks <= 0:
        return []
    if max_rate <= 0.0:
        return [0.0] * num_blocks
    if num_blocks == 1:
        return [max_rate]
    return [float(x) for x in torch.linspace(0, max_rate, num_blocks)]


def generator_reindex(tensor, anchor_idx, n=2, inverse: bool = False):
    """Apply the reversal generator while preserving an optional anchor token."""
    if n != 2:
        raise ValueError(f"Flipped-GARI supports the reversal-induced Z2 action; got n={n}")
    if anchor_idx is None:
        return torch.flip(tensor, dims=[1])
    seq_len = tensor.size(1)
    anchor_idx = anchor_idx if anchor_idx >= 0 else seq_len + anchor_idx
    if anchor_idx < 0 or anchor_idx >= seq_len:
        raise ValueError(f"anchor_idx {anchor_idx} invalid for sequence length {seq_len}")
    if anchor_idx == 0:
        anchor = tensor[:, :1, :]
        rest = torch.flip(tensor[:, 1:, :], dims=[1])
        return torch.cat([anchor, rest], dim=1)
    if anchor_idx == seq_len - 1:
        rest = torch.flip(tensor[:, :-1, :], dims=[1])
        anchor = tensor[:, -1:, :]
        return torch.cat([rest, anchor], dim=1)
    left = torch.flip(tensor[:, :anchor_idx, :], dims=[1])
    anchor = tensor[:, anchor_idx : anchor_idx + 1, :]
    right = torch.flip(tensor[:, anchor_idx + 1 :, :], dims=[1])
    return torch.cat([left, anchor, right], dim=1)


def maybe_compile(fn, compile_mode=None):
    """Apply ``torch.compile`` when a compile mode is requested and available."""
    if compile_mode is None or not hasattr(torch, "compile"):
        return fn
    return torch.compile(fn, mode=compile_mode)


class ODBC(nn.Module):
    """
    ODBC is One-dimensional backward convolution block.
    Args:
        input_dim: int
        kernel_size: int
    """

    def __init__(self, input_dim, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.pad = nn.ZeroPad1d((0, self.kernel_size - 1))
        self.conv = nn.Conv1d(
            self.input_dim,
            self.input_dim,
            self.kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
        )

    def forward(self, x):
        """
        input:
            x: (batch_size, input_dim, seq_len)
        output:
            out: (batch_size, input_dim, seq_len)
        """
        return self.conv(self.pad(x))


class MambaBlock(nn.Module):
    """
    MambaBlock
    """

    def __init__(
        self,
        input_dim,
        d_state,
        d_conv,
        expand,
        head_dim,
        dropout_rate,
    ):
        """
        input_dim: input dimension
        d_state: state dimension
        d_conv: convolution dimension
        expand: expand factor
        head_dim: head dimension
        Attention:input_dim*expand/head_dim is 8's multiple
        dropout_rate: dropout rate
        """
        super().__init__()
        self.Mamba2 = _make_mamba_mixer(input_dim, d_state, d_conv, expand, head_dim)
        self.forward_pipeline = nn.Sequential(
            self.Mamba2,
            nn.Dropout(dropout_rate),
        )
        self.backward_pipeline = nn.Sequential(
            self.Mamba2,
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """
        input:
            x: (2,batch_size,seq_len,input_dim)
        output:
            x: (2,batch_size,seq_len,input_dim)
        """
        # Direct indexing avoids unnecessary tensor copies.
        x_f = self.forward_pipeline(x[0])
        x_b = self.backward_pipeline(x[1])
        return torch.stack([x_f, x_b], dim=0)


class GradientEquilibrium(nn.Module):
    """
    Depth-aware gradient equilibrium between forward/backward chains; learnable bandwidth and min-norm.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        init_min_norm: float = 0.1,
        init_beta: float = 6.907755278982137,  # ln(1000) to match previous [1e-3, 1e3]
        learnable: bool = True,
    ):
        super().__init__()
        self.eps = eps
        # softplus param (>0), initialize close to init_min_norm
        init_min_param = torch.log(torch.expm1(torch.tensor(init_min_norm, dtype=torch.float32)))
        self._min_norm_param = nn.Parameter(init_min_param, requires_grad=learnable)
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32), requires_grad=learnable)

    def min_norm_threshold(self):
        return F.softplus(self._min_norm_param)

    @torch.no_grad()
    def set_depth(
        self,
        layer_idx: int,
        num_layers: int,
        tighten: float = 0.0,
        beta_floor: float = 2.302585092994046,
    ):
        """Optionally tighten by depth; only affects init, still learnable."""
        if num_layers <= 1:
            return
        depth_frac = max(0.0, min(1.0, float(layer_idx) / float(num_layers - 1)))
        current_beta = float(self.beta.detach().cpu())
        new_beta = max(current_beta * (1.0 - tighten * depth_frac), beta_floor)
        self.beta.copy_(torch.tensor(new_beta, device=self.beta.device, dtype=self.beta.dtype))

    def forward(self, x):
        """
        Balance gradients between forward and backward chains
        Args:
            x: (2, batch_size, seq_len, input_dim)
        Returns:
            x: (2, batch_size, seq_len, input_dim)
        """
        x_f, x_b = x[0], x[1]

        # vector_norm is faster and more stable than a manual reduction.
        norm_f = torch.linalg.vector_norm(x_f) + self.eps
        norm_b = torch.linalg.vector_norm(x_b) + self.eps

        # Min-norm guard
        mn = self.min_norm_threshold()
        if norm_f < mn or norm_b < mn:
            x_f = F.layer_norm(x_f, x_f.shape[-1:])
            x_b = F.layer_norm(x_b, x_b.shape[-1:])
            norm_f = torch.sqrt(torch.sum(x_f**2) + self.eps)
            norm_b = torch.sqrt(torch.sum(x_b**2) + self.eps)
            if norm_f < mn:
                x_f = x_f * (mn / (norm_f + self.eps))
            if norm_b < mn:
                x_b = x_b * (mn / (norm_b + self.eps))

        # Clamp ratio to [exp(-beta), exp(beta)]
        ratio = norm_f / (norm_b + self.eps)
        beta = torch.clamp(self.beta, min=0.0)
        lower, upper = torch.exp(-beta), torch.exp(beta)
        if ratio < lower:
            boost = torch.sqrt(lower / (ratio + self.eps))
            x_f = x_f * boost
        elif ratio > upper:
            boost = torch.sqrt(ratio / (upper + self.eps))
            x_b = x_b * boost

        # The learnable guard scalars are otherwise data-conditionally absent
        # from the autograd graph.  Keep a zero-valued dependency so DDP sees
        # the same parameter set on every rank and on every accumulation step.
        guard_dependency = (self._min_norm_param + self.beta) * 0.0
        x_f = x_f + guard_dependency

        return torch.stack([x_f, x_b], dim=0)


class Attention(nn.Module):
    """
    Hybrid Attention: Q/K from one input, V from another input, with optional 1D RoPE positional encoding for Q/K.
    """

    def __init__(
        self,
        input_dim,
        num_heads,
        dropout_rate,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        cls_index: int | str | None = None,
        rope_interleaved: bool = False,
    ):
        """
        input_dim: input dimension
        num_heads: head number
        dropout_rate: dropout rate
        use_rope: whether to apply 1D RoPE to Q/K
        rope_base: base for RoPE frequency computation
        cls_index: optional index for CLS token that should not be rotated
        Attention: input_dim must be divisible by num_heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.input_dim = input_dim
        # Q/K share one source representation; V uses the other source.
        self.qk = nn.Linear(input_dim, input_dim * 2)
        self.v = nn.Linear(input_dim, input_dim)  # V
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(input_dim)
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.cls_index = cls_index
        # False preserves the positional convention used by existing
        # Flipped-GARI checkpoints.
        # New checkpoints can opt into the mathematically
        # consistent adjacent-pair convention used by _rotate_every_two.
        self.rope_interleaved = rope_interleaved

    def _build_rope_cache(self, seq_len: int, device, dtype):
        # RoPE expects even head_dim to pair dimensions
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {self.head_dim}")
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, half_dim)
        if self.rope_interleaved:
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
        else:
            emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]  # (1,1,seq_len,head_dim)
        sin = emb.sin()[None, None, :, :]  # (1,1,seq_len,head_dim)
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)

        # Handle CLS token that should not be rotated
        if self.cls_index is not None:
            idx = None
            if isinstance(self.cls_index, str):
                mode = self.cls_index.lower()
                if mode in {"none", "off"}:
                    idx = None
                elif mode in {"first", "cls", "head"}:
                    idx = 0
                elif mode in {"last", "tail", "-1"}:
                    idx = seq_len - 1
                elif mode in {"middle", "medium", "mid"}:
                    idx = seq_len // 2
                else:
                    raise ValueError(f"Unsupported cls_index mode: {self.cls_index}")
            elif isinstance(self.cls_index, int):
                idx = self.cls_index if self.cls_index >= 0 else seq_len + self.cls_index
            else:
                raise TypeError("cls_index must be int, str, or None")

            if idx is not None:
                if idx < 0 or idx >= seq_len:
                    raise ValueError(
                        f"cls_index {self.cls_index} resolved to invalid position {idx} for seq_len={seq_len}"
                    )
                cos[..., idx, :] = 1.0
                sin[..., idx, :] = 0.0
        return cos, sin

    @staticmethod
    def _rotate_every_two(x):
        # Pairwise rotate last dimension: [-x_odd, x_even]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(-2)

    def _apply_rope(self, q, k):
        # q, k: (batch, heads, seq_len, head_dim)
        b, h, s, d = q.shape
        cos, sin = self._build_rope_cache(s, q.device, q.dtype)
        q_out = (q * cos) + (self._rotate_every_two(q) * sin)
        k_out = (k * cos) + (self._rotate_every_two(k) * sin)
        return q_out, k_out

    def forward(self, x_qk, x_v):
        """
        input:
            x_qk: (batch_size, seq_len, input_dim), source for Q and K
            x_v: (batch_size, seq_len, input_dim), source for V
        output:
            attention: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x_qk.shape

        # Generate Q and K from x_qk.
        qk = self.qk(x_qk).reshape(batch_size, seq_len, 2, self.num_heads, self.head_dim)
        q, k = qk.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # (batch, heads, seq_len, head_dim)

        # Generate V from x_v.
        v = self.v(x_v).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # (batch, heads, seq_len, head_dim)

        # Apply 1D RoPE to queries and keys if enabled
        if self.use_rope:
            q, k = self._apply_rope(q, k)

        # PyTorch selects an available scaled-dot-product attention kernel.
        # This may be Flash Attention on supported hardware.
        # - Memory-Efficient Attention
        # - Standard mathematical implementation
        dropout_p = self.dropout.p if self.training else 0.0

        # Hybrid attention: Q/K from x_qk, V from x_v
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=False,
        )

        # Restore the original tensor layout.
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.input_dim)
        attn = self.norm(attn)
        out = self.dropout(attn)
        return out


class CrossAttention_pro(nn.Module):
    """
    Cross Attention with optional 1D RoPE positional encoding for Q/K.
    """

    def __init__(
        self,
        input_dim,
        num_heads,
        dropout_rate,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        cls_index: int | str | None = None,
        rope_interleaved: bool = False,
    ):
        """
        input_dim: input dimension
        num_heads: head number
        dropout_rate: dropout rate
        use_rope: whether to apply 1D RoPE to Q/K
        rope_base: base for RoPE frequency computation
        Attention: input_dim must be divisible by num_heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.input_dim = input_dim
        # A fused Q/K/V projection reduces kernel launches.
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(input_dim)
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.cls_index = cls_index
        self.rope_interleaved = rope_interleaved

    def _build_rope_cache(self, seq_len: int, device, dtype):
        # RoPE expects even head_dim to pair dimensions
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {self.head_dim}")
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, half_dim)
        if self.rope_interleaved:
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
        else:
            emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]  # (1,1,seq_len,head_dim)
        sin = emb.sin()[None, None, :, :]  # (1,1,seq_len,head_dim)
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
        if self.cls_index is not None:
            idx = None
            if isinstance(self.cls_index, str):
                mode = self.cls_index.lower()
                if mode in {"none", "off"}:
                    idx = None
                elif mode in {"first", "cls", "head"}:
                    idx = 0
                elif mode in {"last", "tail", "-1"}:
                    idx = seq_len - 1
                elif mode in {"middle", "medium", "mid"}:
                    idx = seq_len // 2
                else:
                    raise ValueError(f"Unsupported cls_index mode: {self.cls_index}")
            elif isinstance(self.cls_index, int):
                idx = self.cls_index if self.cls_index >= 0 else seq_len + self.cls_index
            else:
                raise TypeError("cls_index must be int, str, or None")

            if idx is not None:
                if idx < 0 or idx >= seq_len:
                    raise ValueError(
                        f"cls_index {self.cls_index} resolved to invalid position {idx} for seq_len={seq_len}"
                    )
                cos[..., idx, :] = 1.0
                sin[..., idx, :] = 0.0
        return cos, sin

    @staticmethod
    def _rotate_every_two(x):
        # Pairwise rotate last dimension: [-x_odd, x_even]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(-2)

    def _apply_rope(self, q, k):
        # q, k: (batch, heads, seq_len, head_dim)
        b, h, s, d = q.shape
        cos, sin = self._build_rope_cache(s, q.device, q.dtype)
        q_out = (q * cos) + (self._rotate_every_two(q) * sin)
        k_out = (k * cos) + (self._rotate_every_two(k) * sin)
        return q_out, k_out

    def forward(self, x):
        """
        input:
        x: (2,batch_size,seq_len,input_dim)
        output:
        attention: (2,batch_size,seq_len,input_dim)
        """
        # utilize the metric characteristic of cross attention
        x_f, x_b = x[0], x[1]
        batch_size, seq_len, _ = x_f.shape

        # Project Q/K/V together, then split the result.
        qkv_f = self.qkv(x_f).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv_b = self.qkv(x_b).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Split Q, K, V and move heads before the sequence dimension.
        q_f, k_f, v_f = qkv_f.unbind(dim=2)
        q_b, k_b, v_b = qkv_b.unbind(dim=2)
        q_f, k_f, v_f = q_f.transpose(1, 2), k_f.transpose(1, 2), v_f.transpose(1, 2)
        q_b, k_b, v_b = q_b.transpose(1, 2), k_b.transpose(1, 2), v_b.transpose(1, 2)

        # Apply 1D RoPE to queries and keys if enabled
        if self.use_rope:
            q_f, k_f = self._apply_rope(q_f, k_f)
            q_b, k_b = self._apply_rope(q_b, k_b)

        # PyTorch selects an available scaled-dot-product attention kernel.
        # This may be Flash Attention on supported hardware.
        # - Memory-Efficient Attention
        # - Standard mathematical implementation
        dropout_p = self.dropout.p if self.training else 0.0

        # Cross attention: q_b attends to k_f, v_f
        attn_f = F.scaled_dot_product_attention(
            q_b,
            k_f,
            v_f,
            dropout_p=dropout_p,
            is_causal=False,
        )
        # Reverse update uses the symmetric cross-stream pattern.
        attn_b = F.scaled_dot_product_attention(
            q_f,
            k_b,
            v_b,
            dropout_p=dropout_p,
            is_causal=False,
        )
        attn_f = attn_f.transpose(1, 2).reshape(batch_size, seq_len, self.input_dim)
        attn_b = attn_b.transpose(1, 2).reshape(batch_size, seq_len, self.input_dim)
        attn = torch.stack([attn_f, attn_b], dim=0)
        attn = self.norm(attn)
        out = self.dropout(attn)
        return out


class fusion_block(nn.Module):
    def __init__(
        self,
        input_dim,
        kernel_size,
        d_state,
        d_conv,
        expand,
        head_dim,
        dropout_rate1,
        num_heads,
        dropout_rate2,
        cls_rope_mode=None,
        drop_path_rate: float = 0.0,
        enable_cross_attention: bool = True,
        discrepancy_mode: str = "squared_difference",
        rope_interleaved: bool = False,
    ):
        """
        input_dim: input dimension
        kernel_size: kernel size
        d_state: state dimension
        d_conv: convolution dimension
        expand: expand factor
        head_dim:head dimension
        dropout_rate1: MambaBlock dropout rate
        num_heads: head number
        dropout_rate2: CrossAttention dropout rate
        drop_path_rate: stochastic depth on residual branch (timm / ViT style)
        """
        super().__init__()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.ODBC = ODBC(input_dim, kernel_size)
        self.MambaBlock = MambaBlock(
            input_dim,
            d_state,
            d_conv,
            expand,
            head_dim,
            dropout_rate1,
        )
        self.GradientEquilibrium = GradientEquilibrium()
        self.cls_rope_mode_cfg = cls_rope_mode
        self.enable_cross_attention = bool(enable_cross_attention)
        if discrepancy_mode not in {"common_duplicate", "squared_difference"}:
            raise ValueError(
                "discrepancy_mode must be 'common_duplicate' or 'squared_difference'; "
                f"got {discrepancy_mode!r}"
            )
        self.discrepancy_mode = discrepancy_mode
        self.CrossAttention = CrossAttention_pro(
            input_dim,
            num_heads,
            dropout_rate2,
            cls_index=cls_rope_mode,
            rope_interleaved=rope_interleaved,
        )
        self.Attention = Attention(
            input_dim,
            num_heads,
            dropout_rate2,
            cls_index=cls_rope_mode,
            rope_interleaved=rope_interleaved,
        )
        self.Linear1 = nn.Linear(input_dim, input_dim)
        self.Linear2 = nn.Linear(input_dim, input_dim)
        # Separate normalization sites stabilize the residual paths.
        self.norm = nn.LayerNorm(input_dim)
        self.norm_add = nn.LayerNorm(input_dim)
        self.norm_sub = nn.LayerNorm(input_dim)
        self.norm_cross_attention = nn.LayerNorm(input_dim)
        self.norm_attention = nn.LayerNorm(input_dim)
        # Normalize the canonical and flipped streams independently.
        self.norm_after_odbc_f = nn.LayerNorm(input_dim)
        self.norm_after_odbc_b = nn.LayerNorm(input_dim)

        if not self.enable_cross_attention:
            # Keep state_dict/total-parameter compatibility while making the
            # inactive A0 branch explicit to the optimizer and DDP reducer.
            self.CrossAttention.requires_grad_(False)
            self.norm_cross_attention.requires_grad_(False)

        self.GELU = nn.GELU()
        # Initialize parameters
        self.apply_initialization()

    def _resolve_anchor_index(self, seq_len):
        mode = self.cls_rope_mode_cfg
        if mode is None:
            return None
        if isinstance(mode, str):
            m = mode.lower()
            if m in {"first", "cls", "head"}:
                return 0
            if m in {"last", "tail"}:
                return seq_len - 1
            if m in {"medium", "middle", "mid"}:
                return seq_len // 2
            if m in {"none", "off"}:
                return None
            raise ValueError(f"Unsupported cls_rope_mode: {mode}")
        idx = mode if mode >= 0 else seq_len + mode
        if idx < 0 or idx >= seq_len:
            raise ValueError(f"cls_rope_mode index {mode} invalid for sequence length {seq_len}")
        return idx

    @staticmethod
    def GeneratorReindex(tensor, anchor_idx, n=2, inverse: bool = False):
        return generator_reindex(tensor, anchor_idx, n=n, inverse=inverse)

    def _init_weights(self):
        """Initialize weights"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "qkv" in name:
                    # Use a smaller gain for the fused QKV projection.
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    # Other linear layers using He initialization (suitable for GELU activation function)
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                # LayerNorm standard initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv1d):
                # 1D convolution layer using He initialization
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Special initialization: ODBC conv layers using smaller initialization to prevent gradient explosion
        nn.init.xavier_uniform_(self.ODBC.conv.weight, gain=0.1)

        # Linear1 layer: use Xavier initialization (now with GELU)
        nn.init.xavier_uniform_(self.Linear1.weight, gain=1.0)
        if self.Linear1.bias is not None:
            nn.init.zeros_(self.Linear1.bias)

        # Contract layer: use Xavier initialization (more suitable for GELU)
        nn.init.xavier_uniform_(self.Linear2.weight, gain=1.0)
        if self.Linear2.bias is not None:
            nn.init.zeros_(self.Linear2.bias)

    def _init_mamba_parameters(self):
        """Initialize Mamba related parameters"""
        # Initialize parameters in each shared Mamba pipeline.
        for pipeline_name in ["forward_pipeline", "backward_pipeline"]:
            if hasattr(self.MambaBlock, pipeline_name):
                pipeline = getattr(self.MambaBlock, pipeline_name)

                # Iterate through all Mamba2 modules in the pipeline
                for module in pipeline.modules():
                    if hasattr(module, "__class__") and "Mamba2" in str(module.__class__):
                        self._init_single_mamba2(module, pipeline_name)

    def _init_single_mamba2(self, mamba_module, pipeline_name):
        """Initialize special parameters for a single Mamba2 module"""
        # A matrix: state transition matrix
        if hasattr(mamba_module, "A_log"):
            with torch.no_grad():
                # Use uniform initialization range
                mamba_module.A_log.data.uniform_(-4.0, -1.5)

    def apply_initialization(self):
        """Apply custom initialization"""
        self._init_weights()
        self._init_mamba_parameters()

    def forward(self, Tl_1):
        """
        input:
        Tl_1: (2,batch_size,seq_len,input_dim)
        output:
        x: (batch_size,seq_len,input_dim)
        """
        Tl_1p = self.norm(Tl_1)
        x = self.GELU(self.Linear1(Tl_1p))
        # Direct indexing avoids unnecessary tensor copies.
        x_f, x_b = x[0], x[1].flip(1)

        # Apply ODBC in channel-first layout.
        x_f = self.GELU(self.ODBC(x_f.transpose(1, 2)).transpose(1, 2))
        x_b = self.GELU(self.ODBC(x_b.transpose(1, 2)).transpose(1, 2))

        # Normalize forward and backward chains separately
        x_f = self.norm_after_odbc_f(x_f)
        x_b = self.norm_after_odbc_b(x_b)
        x = torch.stack([x_f, x_b], dim=0)

        # MambaBlock with normalization
        x = self.MambaBlock(x)
        x = self.GradientEquilibrium(x)

        # Pre-normalized cross-stream residual update.
        x = torch.stack([x[0], x[1].flip(1)], dim=0)
        if self.enable_cross_attention:
            x = x + self.CrossAttention(self.norm_cross_attention(x))
        x_add = self.norm_add(x[0] + x[1])
        if self.discrepancy_mode == "common_duplicate":
            # Conservative B0 control: keep the query path and its parameters
            # active, but remove access to x_f - x_b directional discrepancy.
            x_query = self.norm_sub(x_add)
        else:
            x_query = self.norm_sub(-((x[0] - x[1]) ** 2))
        x = self.norm_attention(self.Attention(x_query, x_add)) + x_add
        delta = self.GELU(self.Linear2(x))
        x = Tl_1[0] + self.drop_path(delta)
        return x


class MLBN(nn.Module):
    """One intermediate Flipped-GARI block.

    The legacy class name is retained for checkpoint compatibility.
    """

    def __init__(
        self,
        input_dim,
        kernel_size,
        d_state,
        d_conv,
        expand,
        head_dim,
        dropout_rate1,
        num_heads,
        dropout_rate2,
        cls_rope_mode=None,
        drop_path_rate: float = 0.0,
        enable_cross_attention: bool = True,
        rope_interleaved: bool = False,
    ):
        """
        input_dim: input dimension
        kernel_size: kernel size
        d_state: state dimension
        d_conv: convolution dimension
        expand: expand factor
        head_dim:head dimension
        dropout_rate1: MambaBlock dropout rate
        num_heads: head number
        dropout_rate2: CrossAttention dropout rate
        drop_path_rate: stochastic depth on residual branch (timm / ViT style)
        """
        super().__init__()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.ODBC = ODBC(input_dim, kernel_size)
        self.MambaBlock = MambaBlock(
            input_dim,
            d_state,
            d_conv,
            expand,
            head_dim,
            dropout_rate1,
        )
        self.GradientEquilibrium = GradientEquilibrium()
        self.cls_rope_mode_cfg = cls_rope_mode
        self.enable_cross_attention = bool(enable_cross_attention)
        self.CrossAttention = CrossAttention_pro(
            input_dim,
            num_heads,
            dropout_rate2,
            cls_index=cls_rope_mode,
            rope_interleaved=rope_interleaved,
        )
        self.Linear1 = nn.Linear(input_dim, input_dim)
        self.Linear2 = nn.Linear(input_dim, input_dim)
        # Separate normalization sites stabilize the residual paths.
        self.norm = nn.LayerNorm(input_dim)
        self.norm_cross_attention = nn.LayerNorm(input_dim)
        self.norm_after_cross_attention = nn.LayerNorm(input_dim)
        # Normalize the canonical and flipped streams independently.
        self.norm_after_odbc_f = nn.LayerNorm(input_dim)
        self.norm_after_odbc_b = nn.LayerNorm(input_dim)

        if not self.enable_cross_attention:
            self.CrossAttention.requires_grad_(False)
            self.norm_cross_attention.requires_grad_(False)

        self.GELU = nn.GELU()
        # Initialize parameters
        self.apply_initialization()

    def _resolve_anchor_index(self, seq_len):
        mode = self.cls_rope_mode_cfg
        if mode is None:
            return None
        if isinstance(mode, str):
            m = mode.lower()
            if m in {"first", "cls", "head"}:
                return 0
            if m in {"last", "tail"}:
                return seq_len - 1
            if m in {"medium", "middle", "mid"}:
                return seq_len // 2
            if m in {"none", "off"}:
                return None
            raise ValueError(f"Unsupported cls_rope_mode: {mode}")
        idx = mode if mode >= 0 else seq_len + mode
        if idx < 0 or idx >= seq_len:
            raise ValueError(f"cls_rope_mode index {mode} invalid for sequence length {seq_len}")
        return idx

    @staticmethod
    def GeneratorReindex(tensor, anchor_idx, n=2, inverse: bool = False):
        return generator_reindex(tensor, anchor_idx, n=n, inverse=inverse)

    def _init_weights(self):
        """Initialize weights"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "qkv" in name:
                    # Use a smaller gain for the fused QKV projection.
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    # Other linear layers using He initialization (suitable for GELU activation function)
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                # LayerNorm standard initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv1d):
                # 1D convolution layer using He initialization
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Special initialization: ODBC conv layers using smaller initialization to prevent gradient explosion
        nn.init.xavier_uniform_(self.ODBC.conv.weight, gain=0.1)

        # Linear1 layer: use Xavier initialization (now with GELU)
        nn.init.xavier_uniform_(self.Linear1.weight, gain=1.0)
        if self.Linear1.bias is not None:
            nn.init.zeros_(self.Linear1.bias)

        # Contract layer: use Xavier initialization (more suitable for GELU)
        if hasattr(self, "contract"):
            nn.init.xavier_uniform_(self.contract.weight, gain=1.0)
            if self.contract.bias is not None:
                nn.init.zeros_(self.contract.bias)

    def _init_mamba_parameters(self):
        """Initialize Mamba related parameters"""
        # Initialize parameters in each shared Mamba pipeline.
        for pipeline_name in ["forward_pipeline", "backward_pipeline"]:
            if hasattr(self.MambaBlock, pipeline_name):
                pipeline = getattr(self.MambaBlock, pipeline_name)

                # Iterate through all Mamba2 modules in the pipeline
                for module in pipeline.modules():
                    if hasattr(module, "__class__") and "Mamba2" in str(module.__class__):
                        self._init_single_mamba2(module, pipeline_name)

    def _init_single_mamba2(self, mamba_module, pipeline_name):
        """Initialize special parameters for a single Mamba2 module"""
        # A matrix: state transition matrix
        if hasattr(mamba_module, "A_log"):
            with torch.no_grad():
                # Use uniform initialization range
                mamba_module.A_log.data.uniform_(-4.0, -1.5)

    def apply_initialization(self):
        """Apply custom initialization"""
        self._init_weights()
        self._init_mamba_parameters()

    def forward(self, Tl_1):
        """
        input:
        Tl_1: (2,batch_size,seq_len,input_dim)
        output:
        Tl: (2,batch_size,seq_len,input_dim)
        """
        Tl_1p = self.norm(Tl_1)
        x = self.GELU(self.Linear1(Tl_1p))
        # Direct indexing avoids unnecessary tensor copies.
        x_f, x_b = x[0], x[1].flip(1)

        # Apply ODBC in channel-first layout.
        x_f = self.GELU(self.ODBC(x_f.transpose(1, 2)).transpose(1, 2))
        x_b = self.GELU(self.ODBC(x_b.transpose(1, 2)).transpose(1, 2))

        # Normalize forward and backward chains separately
        x_f = self.norm_after_odbc_f(x_f)
        x_b = self.norm_after_odbc_b(x_b)
        x = torch.stack([x_f, x_b], dim=0)

        # MambaBlock with normalization
        x = self.MambaBlock(x)
        x = self.GradientEquilibrium(x)

        # Pre-normalized cross-stream residual update.
        x = torch.stack([x[0], x[1].flip(1)], dim=0)
        if self.enable_cross_attention:
            x = x + self.CrossAttention(self.norm_cross_attention(x))
        delta = self.Linear2(self.norm_after_cross_attention(x))
        Tl = Tl_1 + self.drop_path(delta)
        return Tl

    def get_parameter_count(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameter_count(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_size_mb(self):
        """Get parameter size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_size / (1024 * 1024)

    def get_trainable_parameter_size_mb(self):
        """Get trainable parameter size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters() if p.requires_grad)
        return param_size / (1024 * 1024)

    def get_parameter_details(self):
        """Get detailed parameter breakdown by layer"""
        details = {}
        seen_params = set()

        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                # Count shared parameters only once.
                module_params = 0
                for param in module.parameters():
                    param_id = id(param)
                    if param_id not in seen_params:
                        seen_params.add(param_id)
                        module_params += param.numel()

                if module_params > 0:
                    details[name] = module_params

        return details

    def summary(self):
        """Print detailed model summary"""
        print(f"\n{'=' * 60}")
        print("Flipped-GARI Block Summary")
        print(f"{'=' * 60}")

        # Basic parameter counts
        total_params = self.get_parameter_count()
        trainable_params = self.get_trainable_parameter_count()
        total_size_mb = self.get_parameter_size_mb()
        trainable_size_mb = self.get_trainable_parameter_size_mb()

        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        print(f"Parameter Size: {total_size_mb:.2f} MB")
        print(f"Trainable Size: {trainable_size_mb:.2f} MB")

        # Component details
        print("\nComponents:")
        print(f"  - ODBC: {self.ODBC.input_dim} -> {self.ODBC.input_dim}")
        print("  - MambaBlock: bidirectional")
        print(f"    - d_state: {self.MambaBlock.forward_pipeline[0].d_state}")
        print(f"    - d_conv: {self.MambaBlock.forward_pipeline[0].d_conv}")
        print(f"    - expand: {self.MambaBlock.forward_pipeline[0].expand}")
        print("  - GradientEquilibrium: enabled")
        print(f"  - CrossAttention: {self.CrossAttention.num_heads} heads")
        print(f"    - head_dim: {self.CrossAttention.head_dim}")

        # Parameter breakdown
        print("\nParameter Breakdown:")
        details = self.get_parameter_details()
        for name, params in sorted(details.items(), key=lambda x: abs(x[1]), reverse=True):
            if params != 0:
                print(f"  - {name}: {params:,}")

        # Model configuration
        print("\nModel Configuration:")
        print(f"  - Input dim: {self.ODBC.input_dim}")
        print(f"  - Kernel size: {self.ODBC.kernel_size}")
        print(f"  - Dropout rates: {self.MambaBlock.forward_pipeline[1].p}")

        print(f"{'=' * 60}")


class MLBN_encoder(nn.Module):
    """Flipped-GARI sequence encoder.

    The legacy class name is retained for checkpoint compatibility.
    """

    def __init__(
        self,
        L,
        input_dim,
        kernel_size,
        d_state,
        d_conv,
        expand,
        head_dim,
        dropout_rate1,
        num_heads,
        dropout_rate2,
        cls_rope_mode=None,
        drop_path_rate: float = 0.0,
        checkpoint_blocks: bool = False,
        enable_cross_attention: bool = True,
        discrepancy_mode: str = "squared_difference",
        rope_interleaved: bool = False,
        zero_init_residual: bool = False,
    ):
        """
        L: number of intermediate Flipped-GARI blocks to stack
        Other parameters: shared block and terminal-fusion settings
        input_dim: hidden-state dimension
        kernel_size: ODBC kernel width
        d_state: Mamba state dimension
        d_conv: Mamba convolution width
        expand: Mamba expansion factor
        Attention:input_dim*expand/head_dim is 8's multiple,input_dim must be divisible by num_heads
        cls_rope_mode: None,first,last
        drop_path_rate: max stochastic depth; linear 0 to max over L blocks plus terminal fusion
        """
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.enable_cross_attention = bool(enable_cross_attention)
        self.discrepancy_mode = discrepancy_mode
        self.rope_interleaved = bool(rope_interleaved)
        self.zero_init_residual = bool(zero_init_residual)
        num_blocks = L + 1
        dpr = _encoder_drop_path_rates(num_blocks, drop_path_rate)
        self.blocks = nn.ModuleList(
            [
                MLBN(
                    input_dim,
                    kernel_size,
                    d_state,
                    d_conv,
                    expand,
                    head_dim,
                    dropout_rate1,
                    num_heads,
                    dropout_rate2,
                    cls_rope_mode=cls_rope_mode,
                    drop_path_rate=dpr[i],
                    enable_cross_attention=enable_cross_attention,
                    rope_interleaved=rope_interleaved,
                )
                for i in range(L)
            ]
            + [
                fusion_block(
                    input_dim,
                    kernel_size,
                    d_state,
                    d_conv,
                    expand,
                    head_dim,
                    dropout_rate1,
                    num_heads,
                    dropout_rate2,
                    cls_rope_mode=cls_rope_mode,
                    drop_path_rate=dpr[L],
                    enable_cross_attention=enable_cross_attention,
                    discrepancy_mode=discrepancy_mode,
                    rope_interleaved=rope_interleaved,
                )
            ]
        )
        # depth-aware init for GradientEquilibrium
        for i, block in enumerate(self.blocks):
            if hasattr(block, "GradientEquilibrium") and hasattr(block.GradientEquilibrium, "set_depth"):
                block.GradientEquilibrium.set_depth(
                    layer_idx=i,
                    num_layers=len(self.blocks),
                    tighten=0,
                    beta_floor=math.log(10.0),
                )

        # self.multi_scale_enhancer = MultiScaleFeatureEnhancer(
        #    input_dim, enable_class_attention=True
        # )

        # Initialize parameters for all blocks
        self.apply_encoder_initialization()
        if self.zero_init_residual:
            # Start every residual branch as an exact identity.  This is
            # opt-in so existing checkpoints/configs retain legacy behavior,
            # while deep randomly initialized ablations avoid a product of
            # large normalization Jacobians on the first backward pass.
            for block in self.blocks:
                nn.init.zeros_(block.Linear2.weight)
                if block.Linear2.bias is not None:
                    nn.init.zeros_(block.Linear2.bias)

    def _init_encoder_weights(self):
        """Custom encoder weights initialization"""
        for i, block in enumerate(self.blocks):
            # Apply residual scaling for deeper layers to prevent gradient vanishing
            residual_scale = 0.1 ** (i / len(self.blocks))  # Gradually smaller initialization

            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    if "qkv" in name:
                        # Use a smaller gain for the fused QKV projection.
                        nn.init.xavier_uniform_(module.weight, gain=0.5 * residual_scale)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    else:
                        # Other linear layers using He initialization
                        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

                elif isinstance(module, nn.LayerNorm):
                    # LayerNorm standard initialization
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

                elif isinstance(module, nn.Conv1d):
                    # 1D convolution layer using He initialization
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        print("Flipped-GARI encoder weight initialization completed")

    def _init_encoder_mamba_parameters(self):
        """Initialize Mamba related parameters in the encoder"""
        for i, block in enumerate(self.blocks):
            # Process MambaBlock parameters in each block
            for pipeline_name in ["forward_pipeline", "backward_pipeline"]:
                if hasattr(block.MambaBlock, pipeline_name):
                    pipeline = getattr(block.MambaBlock, pipeline_name)

                    # Iterate through all Mamba2 modules in the pipeline
                    for module in pipeline.modules():
                        if hasattr(module, "__class__") and "Mamba2" in str(module.__class__):
                            self._init_single_mamba2_encoder(module, pipeline_name, i)

    def _init_single_mamba2_encoder(self, mamba_module, pipeline_name, block_idx):
        """Initialize special parameters for a single Mamba2 module in the encoder"""
        # A matrix: state transition matrix, adjust initialization based on block depth
        if hasattr(mamba_module, "A_log"):
            with torch.no_grad():
                # Deep blocks use more stable initialization
                depth_factor = 0.1 ** (block_idx / max(1, len(self.blocks)))
                mamba_module.A_log.data.uniform_(-4.0 * depth_factor, -1.5 * depth_factor)

    def apply_encoder_initialization(self):
        """Apply encoder custom initialization"""
        self._init_encoder_weights()
        self._init_encoder_mamba_parameters()

    def get_parameter_count(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameter_count(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_size_mb(self):
        """Get parameter size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_size / (1024 * 1024)

    def get_trainable_parameter_size_mb(self):
        """Get trainable parameter size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters() if p.requires_grad)
        return param_size / (1024 * 1024)

    def get_parameter_details(self):
        """Get detailed parameter breakdown by layer"""
        details = {}
        seen_params = set()

        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                # Count shared parameters only once.
                module_params = 0
                for param in module.parameters():
                    param_id = id(param)
                    if param_id not in seen_params:
                        seen_params.add(param_id)
                        module_params += param.numel()

                if module_params > 0:
                    details[name] = module_params

        # Calculate parameters per block
        block_params = []
        for i, block in enumerate(self.blocks):
            block_param_count = sum(p.numel() for p in block.parameters())
            block_params.append(block_param_count)
            details[f"flipped_gari_block_{i}"] = block_param_count
        return details

    def summary(self):
        """Print detailed model summary"""
        print(f"\n{'=' * 60}")
        print(f"Flipped-GARI Encoder Summary (L={len(self.blocks)})")
        print(f"{'=' * 60}")

        # Basic parameter counts
        total_params = self.get_parameter_count()
        trainable_params = self.get_trainable_parameter_count()
        total_size_mb = self.get_parameter_size_mb()
        trainable_size_mb = self.get_trainable_parameter_size_mb()

        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        print(f"Parameter Size: {total_size_mb:.2f} MB")
        print(f"Trainable Size: {trainable_size_mb:.2f} MB")

        # Component details
        print("\nComponents:")
        print(f"  - Number of Flipped-GARI blocks: {len(self.blocks)}")
        for i, block in enumerate(self.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            print(f"    - Block {i}: {block_params:,} parameters")
        print("  - Discrepancy-aware terminal fusion: enabled")

        # Parameter breakdown
        print("\nParameter Breakdown:")
        details = self.get_parameter_details()
        for name, params in sorted(details.items(), key=lambda x: x[1], reverse=True):
            if params > 0:
                print(f"  - {name}: {params:,}")

        # Model configuration
        if len(self.blocks) > 0:
            first_block = self.blocks[0]
            print("\nModel Configuration:")
            print(f"  - Input dim: {first_block.ODBC.input_dim}")
            print(f"  - Kernel size: {first_block.ODBC.kernel_size}")
            print(f"  - d_state1: {first_block.MambaBlock.forward_pipeline[0].d_state}")
            print(f"  - d_state2: {first_block.MambaBlock.backward_pipeline[0].d_state}")
            print(f"  - d_conv1: {first_block.MambaBlock.forward_pipeline[0].d_conv}")
            print(f"  - d_conv2: {first_block.MambaBlock.backward_pipeline[0].d_conv}")
            print(f"  - expand: {first_block.MambaBlock.forward_pipeline[0].expand}")
            print(f"  - head_dim: {first_block.MambaBlock.forward_pipeline[0].head_dim}")

        print(f"{'=' * 60}")

    def forward(self, x):
        """
        input:
            x: (batch_size, seq_len, input_dim)
        output:
            x: (batch_size, seq_len, input_dim)
        """
        x = torch.stack([x, x], dim=0)

        for block in self.blocks:
            if self.checkpoint_blocks and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


# Public names used by the paper and release configs. The aliases preserve the
# module structure and state-dict keys of existing experimental checkpoints.
FlippedGARIBlock = MLBN
FlippedGARITerminalFusion = fusion_block
FlippedGARIEncoder = MLBN_encoder
