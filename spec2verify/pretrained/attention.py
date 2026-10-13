"""Attention analysis (Appendix F.8, "Attention").

Head selection: each candidate head's output at the query position is zeroed
(its aggregated value vector, before the output projection; under grouped-query
attention only that query head) and the head whose ablation most reduces atomic
IIA on development pairs is kept (ties: shallowest layer, then smallest head
index); test pairs only evaluate this head.

Attention mass: the softmax weight from the last subword at the consumption
position to all subwords of the target word (subject, antecedent, determiner or
licensor), summed over those subwords, on 1,000 test items per phenomenon in
each of five repeats (seeds 1103 ... 5531), averaged within each subset and then
over phenomena and repeats.  "Largest weight" compares whole-word sums.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPEAT_SEEDS = (1103, 2207, 3319, 4421, 5531)


@contextmanager
def ablate_scratch_head(model, layer: int, head: int, positions: torch.Tensor):
    attn = model.blocks[layer - 1].attn
    attn.ablate = (head, positions)
    try:
        yield
    finally:
        attn.ablate = None


def output_projection(model, layer: int):
    from .hf import blocks
    blk = blocks(model)[layer - 1]
    if hasattr(blk, "attention"):            # GPT-NeoX (Pythia)
        return blk.attention.dense, blk.attention.head_size
    return blk.self_attn.o_proj, blk.self_attn.head_dim


@contextmanager
def ablate_hf_head(model, layer: int, head: int, positions):
    """``positions``: a tensor of query positions, one per row, or a callable returning the
    current batch's positions (``lambda: backbone.current_positions``)."""
    proj, dh = output_projection(model, layer)

    def pre(module, args):
        x = args[0].clone()
        pos = positions() if callable(positions) else positions
        x[torch.arange(x.shape[0], device=x.device), pos, head * dh:(head + 1) * dh] = 0.0
        return (x,) + tuple(args[1:])
    h = proj.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        h.remove()


def select_head(candidates: Sequence[Tuple[int, int]], atomic_iia: Callable[[Tuple[int, int] or None], float]
                ) -> Tuple[int, int]:
    """The (layer, head) whose ablation lowers development atomic IIA most."""
    base = atomic_iia(None)
    drops = [(-(base - atomic_iia(c)), c[0], c[1]) for c in candidates]
    drops.sort()
    return drops[0][1], drops[0][2]


def attention_mass(row: torch.Tensor, query: int, key_positions: Sequence[int], all_words: List[List[int]]):
    """(mass on the target word, whether it is the largest whole-word mass) for one item.

    row: the softmax weights of one head from the query position; all_words: subword
    positions of every word before the critical word."""
    mass = row[list(key_positions)].sum().item()
    best = max(row[list(w)].sum().item() for w in all_words if w and max(w) <= query)
    return mass, mass >= best - 1e-12
