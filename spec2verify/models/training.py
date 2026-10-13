"""Training of the from-scratch models (Appendix C.1, Table 8; Appendices F.3 and F.7).

AdamW (betas 0.9/0.999, eps 1e-8, weight decay on every parameter), 500 linear
warm-up steps then cosine decay to zero, batch 64, gradient-norm clip 1.0, no
dropout and no early stopping.  A checkpoint is written every 500 steps; the
converged results use the final checkpoint.  Each model seed sets Python,
NumPy and PyTorch (all CUDA devices).
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.templates import Sentence
from ..data.vocab import LABEL_TOKENS, Vocab
from .architectures import build_model, site_for

MODEL_SEEDS = (1103, 2207, 3319, 4421, 5531)


@dataclass
class TrainConfig:
    arch: str = "transformer"
    d: int = 128
    d_state: int = 16
    max_len: int = 30
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    batch_size: int = 64
    steps: int = 50_000
    warmup: int = 500
    grad_clip: float = 1.0
    checkpoint_every: int = 500
    seed: int = 1103
    aux_weight: float = 0.0          # Appendix F.3: 0.3 for the auxiliary atomic loss
    label_as_input: bool = False     # Appendix F.3: sentence-initial label token
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_every: int = 500


#: Per-architecture settings of the size sweep (Appendix F.7, selected on dev accuracy at d = 0..4).
TUNED = {
    "transformer": dict(lr=1e-3, weight_decay=0.0, steps=200_000),
    "lstm": dict(lr=1e-3, weight_decay=0.1, steps=200_000),
    "mamba": dict(lr=1e-3, weight_decay=0.0, steps=200_000, d_state=256),
}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------- batching
def encode_batch(sents: Sequence[Sentence], vocab: Vocab, device: str = "cpu", label_as_input: bool = False,
                 max_len: Optional[int] = None):
    """Right-padded token ids, target positions and labels."""
    rows, targets = [], []
    for s in sents:
        w = list(s.words)
        t = s.target
        if label_as_input:
            w = [LABEL_TOKENS[s.label]] + w; t += 1
        rows.append(vocab.encode(w)); targets.append(t)
    n = max(len(r) for r in rows)
    if max_len:
        assert n <= max_len, f"sequence of length {n} exceeds context {max_len}"
    ids = torch.zeros(len(rows), n, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
    return (ids.to(device), torch.tensor(targets, device=device),
            torch.tensor([s.label for s in sents], device=device))


def lr_lambda(cfg: TrainConfig) -> Callable[[int], float]:
    def f(step: int) -> float:
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


@torch.no_grad()
def accuracy(model: nn.Module, sents: Sequence[Sentence], vocab: Vocab, device: str, bs: int = 512,
             label_as_input: bool = False) -> float:
    model.eval(); correct = 0
    for i in range(0, len(sents), bs):
        ids, tgt, y = encode_batch(sents[i:i + bs], vocab, device, label_as_input)
        correct += (model(ids, tgt).argmax(-1) == y).sum().item()
    return correct / max(1, len(sents))


class AuxHead(nn.Module):
    """Appendix F.3: an independent 128 -> 2 head at the first-block site that predicts the
    atomic SCM variable; used only in the training loss."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 2)


def train(train_set: List[Sentence], vocab: Vocab, cfg: TrainConfig, out_dir: Optional[str] = None,
          eval_sets: Optional[Dict[str, List[Sentence]]] = None,
          atomic_labels: Optional[Callable[[Sentence], int]] = None,
          on_checkpoint: Optional[Callable[[int, nn.Module], None]] = None) -> nn.Module:
    """Train one model; returns the final model.  ``on_checkpoint(step, model)`` is called
    at step 0 and every ``checkpoint_every`` steps (e.g. to compute per-variable IIA)."""
    deterministic(); set_seed(cfg.seed)
    vocab_size = len(vocab)
    model = build_model(cfg.arch, vocab_size, cfg.d, cfg.max_len, cfg.d_state).to(cfg.device)
    aux = AuxHead(cfg.d).to(cfg.device) if cfg.aux_weight > 0 else None
    params = list(model.parameters()) + (list(aux.parameters()) if aux else [])
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    order_rng = np.random.default_rng(cfg.seed)
    perm = order_rng.permutation(len(train_set)); cursor = 0
    log: List[Dict] = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def checkpoint(step: int):
        rec = {"step": step}
        for name, data in (eval_sets or {}).items():
            rec[f"{name}_accuracy"] = accuracy(model, data, vocab, cfg.device, label_as_input=cfg.label_as_input)
        log.append(rec)
        if out_dir:
            torch.save(model.state_dict(), os.path.join(out_dir, f"step{step:06d}.pt"))
        if on_checkpoint:
            on_checkpoint(step, model)
        model.train()

    checkpoint(0)
    model.train()
    for step in range(1, cfg.steps + 1):
        if cursor + cfg.batch_size > len(perm):
            perm = order_rng.permutation(len(train_set)); cursor = 0
        batch = [train_set[i] for i in perm[cursor:cursor + cfg.batch_size]]; cursor += cfg.batch_size
        ids, tgt, y = encode_batch(batch, vocab, cfg.device, cfg.label_as_input, cfg.max_len)
        if aux is not None:
            site = site_for(model, "atomic")
            h1 = model.encode(ids, site)
            logits = model.decode(h1, site, ids, tgt)
            a_y = torch.tensor([int(atomic_labels(s)) for s in batch], device=cfg.device)
            a_logits = aux.lin(h1[torch.arange(len(batch), device=cfg.device), tgt])
            loss = F.cross_entropy(logits, y) + cfg.aux_weight * F.cross_entropy(a_logits, a_y)
        else:
            loss = F.cross_entropy(model(ids, tgt), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step(); sched.step()
        if step % cfg.checkpoint_every == 0:
            checkpoint(step)
    model.training_log = log
    if out_dir:
        torch.save({"config": cfg.__dict__, "log": log}, os.path.join(out_dir, "run.pt"))
    return model.eval()


def load_model(path: str, arch: str, vocab_size: int, cfg: Optional[TrainConfig] = None) -> nn.Module:
    cfg = cfg or TrainConfig(arch=arch)
    m = build_model(arch, vocab_size, cfg.d, cfg.max_len, cfg.d_state)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


# ----------------------------------------------------------------------------- milestones
def first_step_above(log: List[Dict], key: str, threshold: float) -> Optional[int]:
    """First saved checkpoint whose value is strictly above ``threshold`` (no interpolation)."""
    for rec in log:
        if rec.get(key) is not None and rec[key] > threshold:
            return rec["step"]
    return None


def grokking_point(log: List[Dict], key: str = "test_accuracy") -> Optional[int]:
    """Appendix C.1: first checkpoint with test accuracy above 95%."""
    return first_step_above(log, key, 0.95)
