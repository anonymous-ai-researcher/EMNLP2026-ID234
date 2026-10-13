"""The three from-scratch architectures (Appendix C.1; Appendix F.7 for the size sweep).

All three map token ids to a binary grammaticality decision read at one target
position, and expose the same *staged* forward pass: ``encode(ids, upto=site)``
returns the residual stream at a named site, and ``decode(h, site, ...)``
continues from it.  DAS, SAEs and probes act on these sites.

Sites (Appendix C.3): ``block1`` (output of the first block; atomic variables),
``block2_attn`` (Transformer: the residual stream entering the last MLP, after the
second attention sublayer; compositional variables) and ``block2`` (output of
the last block).  The recurrent models have no sublayers, so their
compositional site is ``block2``.

Parameter counts, including embeddings, normalization and the two-class head,
match the formulas of Appendix F.7 exactly (see ``count_parameters`` and the tests):
Transformer 24w^2 + 229w + 2, LSTM 16w^2 + 153w + 2 and Mamba
12w^2 + 8w*ceil(w/16) + 3240w + 2 for a 135-token vocabulary, context 64 and Mamba
state size 256.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Hook = Callable[[torch.Tensor], torch.Tensor]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================================ Transformer
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention; a position may attend to itself, never to the
    future or to padding.  Exposes per-head outputs for head ablation (Appendix F.8)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ablate: Optional[Tuple[int, torch.Tensor]] = None   # (head, positions) to zero
        self.last_attention: Optional[torch.Tensor] = None
        self.current_targets: Optional[torch.Tensor] = None
        self.record = False

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        mask = causal[None, None]
        if pad_mask is not None:
            mask = mask & pad_mask[:, None, None, :]
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        if self.record:
            self.last_attention = att.detach()
        y = att @ v                                            # (b, h, n, dh)
        if self.ablate is not None:
            head, pos = self.ablate
            pos = self.current_targets if pos is None else pos
            y = y.clone()
            y[torch.arange(b, device=x.device), head, pos] = 0.0
        return self.out(y.transpose(1, 2).reshape(b, n, self.d))


class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))


class Transformer(nn.Module):
    """Causal pre-LN Transformer: h <- h + Attn(LN(h)); h <- h + FFN(LN(h))."""

    kind = "transformer"
    SITES = ("embed", "block1", "block2_attn", "block2")

    def __init__(self, vocab_size: int, d: int = 128, n_layers: int = 2, n_heads: int = 4,
                 ffn: Optional[int] = None, max_len: int = 30):
        super().__init__()
        self.d, self.n_layers = d, n_layers
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads, ffn or 4 * d) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d, eps=1e-5)
        self.head = nn.Linear(d, 2)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    @property
    def sites(self) -> List[str]:
        out = ["embed"]
        for l in range(1, self.n_layers + 1):
            out += [f"block{l}_attn", f"block{l}"]
        return out

    def _stages(self):
        """Ordered (site_after_stage, function) pairs."""
        st = []
        for l, blk in enumerate(self.blocks, 1):
            st.append((f"block{l}_attn", lambda h, m, b=blk: h + b.attn(b.ln1(h), m)))
            st.append((f"block{l}", lambda h, m, b=blk: h + b.ffn(b.ln2(h))))
        return st

    def encode(self, ids: torch.Tensor, upto: str = "embed", hooks: Optional[Dict[str, Hook]] = None
               ) -> torch.Tensor:
        pad = ids != 0
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        h = self.embed(ids) + self.pos(pos)
        if hooks and "embed" in hooks:
            h = hooks["embed"](h)
        if upto == "embed":
            return h
        for name, fn in self._stages():
            h = fn(h, pad)
            if hooks and name in hooks:
                h = hooks[name](h)
            if name == upto:
                return h
        raise KeyError(upto)

    def decode(self, h: torch.Tensor, site: str, ids: torch.Tensor, target: torch.Tensor,
               hooks: Optional[Dict[str, Hook]] = None) -> torch.Tensor:
        pad = ids != 0
        started = site == "embed"
        for name, fn in self._stages():
            if started:
                h = fn(h, pad)
                if hooks and name in hooks:
                    h = hooks[name](h)
            elif name == site:
                started = True
        return self.read(h, target)

    def read(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_f(h[torch.arange(h.shape[0], device=h.device), target]))

    def forward(self, ids, target, hooks: Optional[Dict[str, Hook]] = None):
        return self.decode(self.encode(ids, "embed", hooks), "embed", ids, target, hooks)


# ============================================================================ LSTM
class LSTMClassifier(nn.Module):
    """Unidirectional LSTM, zero initial states, default PyTorch gate initialization."""

    kind = "lstm"

    def __init__(self, vocab_size: int, d: int = 128, n_layers: int = 2, **_):
        super().__init__()
        self.d, self.n_layers = d, n_layers
        self.embed = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList([nn.LSTM(d, d, num_layers=1, batch_first=True) for _ in range(n_layers)])
        self.head = nn.Linear(d, 2)

    @property
    def sites(self) -> List[str]:
        return ["embed"] + [f"block{l}" for l in range(1, self.n_layers + 1)]

    def encode(self, ids, upto="embed", hooks=None):
        h = self.embed(ids)
        if hooks and "embed" in hooks:
            h = hooks["embed"](h)
        if upto == "embed":
            return h
        for l, rnn in enumerate(self.layers, 1):
            h, _ = rnn(h)
            if hooks and f"block{l}" in hooks:
                h = hooks[f"block{l}"](h)
            if upto == f"block{l}":
                return h
        raise KeyError(upto)

    def decode(self, h, site, ids, target, hooks=None):
        start = 0 if site == "embed" else int(site.replace("block", ""))
        for l in range(start + 1, self.n_layers + 1):
            h, _ = self.layers[l - 1](h)
            if hooks and f"block{l}" in hooks:
                h = hooks[f"block{l}"](h)
        return self.read(h, target)

    def read(self, h, target):
        return self.head(h[torch.arange(h.shape[0], device=h.device), target])

    def forward(self, ids, target, hooks=None):
        return self.decode(self.encode(ids, "embed", hooks), "embed", ids, target, hooks)


# ============================================================================ Mamba-1
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaMixer(nn.Module):
    """Mamba-1 selective state-space mixer in plain PyTorch (sequential scan).

    in_proj (no bias) -> causal depthwise conv (width 4, with bias) -> SiLU -> x_proj
    (no bias) -> dt_proj (with bias, softplus) -> selective scan with A = -exp(A_log)
    and skip D -> gate SiLU(z) -> out_proj (no bias).  Initialization follows the
    reference module: dt in [0.001, 0.1] log-uniform (floor 1e-4), A_log = log(1..N),
    D = 1.  ``last_states`` keeps the per-step states when ``record`` is set (state probe)."""

    def __init__(self, d: int, d_state: int = 16, expand: int = 2, d_conv: int = 4,
                 dt_rank: Optional[int] = None, dt_min: float = 1e-3, dt_max: float = 0.1,
                 dt_init_floor: float = 1e-4, dt_scale: float = 1.0):
        super().__init__()
        self.d_inner = expand * d
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d / 16)
        self.in_proj = nn.Linear(d, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)
        self.record = False
        self.last_states: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :n].transpose(1, 2))
        dbc = self.x_proj(u)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                         # (b, n, d_inner)
        A = -torch.exp(self.A_log)                                # (d_inner, N)
        state = x.new_zeros(b, self.d_inner, self.d_state)
        ys, states = [], []
        for t in range(n):
            dA = torch.exp(dt[:, t, :, None] * A)                  # (b, d_inner, N)
            dBu = dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            state = dA * state + dBu
            if self.record:
                states.append(state)
            ys.append((state * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, 1) + u * self.D
        if self.record:
            self.last_states = torch.stack(states, 1).detach()   # (b, n, d_inner, N)
        return self.out_proj(y * F.silu(z))


def _fused_mixer(d: int, d_state: int, expand: int, d_conv: int, dt_rank: Optional[int]):
    """The CUDA ``mamba-ssm`` Mamba-1 module (same parameterization and initialization),
    used when SPEC2VERIFY_MAMBA_SSM=1 and the package is installed; the plain PyTorch
    mixer is used otherwise and for the state probe."""
    import os
    if os.environ.get("SPEC2VERIFY_MAMBA_SSM") != "1":
        return None
    try:
        from mamba_ssm.modules.mamba_simple import Mamba as FusedMamba
    except ImportError:
        return None
    return FusedMamba(d_model=d, d_state=d_state, d_conv=d_conv, expand=expand,
                      dt_rank=dt_rank or math.ceil(d / 16), bias=False, conv_bias=True)


class MambaBlock(nn.Module):
    def __init__(self, d: int, **kw):
        super().__init__()
        self.norm = RMSNorm(d)
        self.mixer = _fused_mixer(d, kw.get("d_state", 16), kw.get("expand", 2), kw.get("d_conv", 4),
                                  kw.get("dt_rank")) or MambaMixer(d, **kw)

    def forward(self, h):
        return h + self.mixer(self.norm(h))


class Mamba(nn.Module):
    """Pre-RMSNorm residual Mamba-1 blocks with a final RMSNorm."""

    kind = "mamba"

    def __init__(self, vocab_size: int, d: int = 128, n_layers: int = 2, d_state: int = 16,
                 expand: int = 2, d_conv: int = 4, dt_rank: Optional[int] = None, **_):
        super().__init__()
        self.d, self.n_layers = d, n_layers
        self.embed = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList([MambaBlock(d, d_state=d_state, expand=expand, d_conv=d_conv,
                                                dt_rank=dt_rank) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d)
        self.head = nn.Linear(d, 2)

    @property
    def sites(self) -> List[str]:
        return ["embed"] + [f"block{l}" for l in range(1, self.n_layers + 1)]

    def encode(self, ids, upto="embed", hooks=None):
        h = self.embed(ids)
        if hooks and "embed" in hooks:
            h = hooks["embed"](h)
        if upto == "embed":
            return h
        for l, blk in enumerate(self.blocks, 1):
            h = blk(h)
            if hooks and f"block{l}" in hooks:
                h = hooks[f"block{l}"](h)
            if upto == f"block{l}":
                return h
        raise KeyError(upto)

    def decode(self, h, site, ids, target, hooks=None):
        start = 0 if site == "embed" else int(site.replace("block", ""))
        for l in range(start + 1, self.n_layers + 1):
            h = self.blocks[l - 1](h)
            if hooks and f"block{l}" in hooks:
                h = hooks[f"block{l}"](h)
        return self.read(h, target)

    def read(self, h, target):
        return self.head(self.norm_f(h[torch.arange(h.shape[0], device=h.device), target]))

    def forward(self, ids, target, hooks=None):
        return self.decode(self.encode(ids, "embed", hooks), "embed", ids, target, hooks)


# ============================================================================ factory
def build_model(arch: str, vocab_size: int, d: int = 128, max_len: int = 30, d_state: int = 16,
                n_layers: int = 2) -> nn.Module:
    if arch == "transformer":
        return Transformer(vocab_size, d, n_layers, n_heads=4, ffn=4 * d, max_len=max_len)
    if arch == "lstm":
        return LSTMClassifier(vocab_size, d, n_layers)
    if arch == "mamba":
        return Mamba(vocab_size, d, n_layers, d_state=d_state, dt_rank=math.ceil(d / 16))
    raise ValueError(arch)


def site_for(model: nn.Module, kind: str) -> str:
    """The fixed alignment sites of the 2-layer models (Appendix C.3)."""
    if kind == "atomic":
        return "block1"
    return "block2_attn" if getattr(model, "kind", "") == "transformer" else "block2"


def parameter_formula(arch: str, w: int) -> int:
    """Appendix F.7 (135-token vocabulary, context 64, Mamba state 256)."""
    if arch == "transformer":
        return 24 * w * w + 229 * w + 2
    if arch == "lstm":
        return 16 * w * w + 153 * w + 2
    if arch == "mamba":
        return 12 * w * w + 8 * w * math.ceil(w / 16) + 3240 * w + 2
    raise ValueError(arch)
