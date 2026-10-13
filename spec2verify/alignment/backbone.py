"""A common interface over the networks that DAS, SAEs and probes intervene on.

``vectors(sents, site)`` returns the residual-stream vector of each sentence at
its intervention position, and ``patched_logits(sents, site, new)`` replaces
that vector by ``new`` (differentiably) and returns two-class logits whose
argmax is kappa, the network's discrete output.  Frozen activations are cached
per (site, sentence set) because the network never changes during alignment.

* ``ScratchBackbone`` wraps the from-scratch classifiers: the intervention
  position is the target (consumption) position and the logits are the
  classifier's.
* ``spec2verify.pretrained.hf.HFBackbone`` implements the same interface for
  pre-trained language models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..data.templates import Sentence
from ..data.vocab import Vocab
from ..models.training import encode_batch


class Backbone:
    sites: List[str]
    d: int

    def vectors(self, sents: Sequence[Sentence], site: str) -> torch.Tensor:
        raise NotImplementedError

    def patched_logits(self, sents: Sequence[Sentence], site: str, new: torch.Tensor,
                       second: Optional[Tuple[str, torch.Tensor]] = None) -> torch.Tensor:
        raise NotImplementedError

    def logits(self, sents: Sequence[Sentence]) -> torch.Tensor:
        raise NotImplementedError


class ScratchBackbone(Backbone):
    def __init__(self, model: nn.Module, vocab: Vocab, device: Optional[str] = None,
                 label_as_input: bool = False):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.vocab = vocab
        self.device = device or next(model.parameters()).device.type
        self.model.to(self.device)
        self.d = model.d
        self.sites = [s for s in model.sites if s != "embed"]
        self.label_as_input = label_as_input
        self._cache: Dict[Tuple, Tuple] = {}

    def _encode(self, sents: Sequence[Sentence]):
        ids, tgt, y = encode_batch(sents, self.vocab, self.device, self.label_as_input)
        for blk in getattr(self.model, "blocks", []):
            if hasattr(blk, "attn"):
                blk.attn.current_targets = tgt          # read by head ablation (Appendix F.8)
        return ids, tgt, y

    @staticmethod
    def _key(s: Sentence):
        return (tuple(s.words), s.target, s.label if s.label in (0, 1) else -1)

    @torch.no_grad()
    def vectors(self, sents, site, bs: int = 1024) -> torch.Tensor:
        """Target-position vectors at ``site``, cached by sentence content."""
        cache = self._cache.setdefault(site, {})
        todo = [s for s in dict.fromkeys(self._key(s) for s in sents) if s not in cache]
        if todo:
            proxies = [Sentence(list(w), lab, t, "", "") for (w, t, lab) in todo]
            for i in range(0, len(proxies), bs):
                ids, tgt, _ = self._encode(proxies[i:i + bs])
                h = self.model.encode(ids, site)
                v = h[torch.arange(h.shape[0], device=h.device), tgt]
                for k, row in zip(todo[i:i + bs], v):
                    cache[k] = row
        return torch.stack([cache[self._key(s)] for s in sents])

    def patched_logits(self, sents, site, new, second=None) -> torch.Tensor:
        """``new``: (B, d) vectors for the intervention position at ``site``; ``second``:
        an optional (site2, fn) where fn(h_pos) returns the patched vector at a later site
        (used by joint alignment, whose atomic patch propagates to the compositional site)."""
        ids, tgt, _ = self._encode(sents)
        with torch.no_grad():
            h = self.model.encode(ids, site)
        b = torch.arange(len(sents), device=h.device)
        h = h.clone(); h[b, tgt] = new.to(h.dtype)
        hooks = None
        if second is not None:
            site2, fn = second

            def hook(x, fn=fn):
                x = x.clone(); x[b, tgt] = fn(x[b, tgt]); return x
            hooks = {site2: hook}
        return self.model.decode(h, site, ids, tgt, hooks)

    @torch.no_grad()
    def logits(self, sents, bs: int = 1024) -> torch.Tensor:
        out = []
        for i in range(0, len(sents), bs):
            ids, tgt, _ = self._encode(sents[i:i + bs])
            out.append(self.model(ids, tgt))
        return torch.cat(out)
