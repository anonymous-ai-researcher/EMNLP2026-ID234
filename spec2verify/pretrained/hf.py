"""Pre-trained language models (Appendices C.2 and C.4).

Models run frozen in float32, in evaluation mode, with eager attention, without
quantization or fine-tuning; the tokenizer and weights share one identifier.
Llama-3.1-8B-Instruct receives the same raw sentences as the base models (no
chat template).

Sentence scoring: sums of natural-log token probabilities, no length
normalization.  Llama and Gemma receive exactly one BOS token and every word is
scored; Pythia and Qwen receive no BOS, so the first token is unscored.  Exact
ties are broken by the parity of the pair ID.

Interventions act on the residual stream after a full block (``layer{l}``,
1-based).  The language model is read where it predicts the critical word: the
intervention position is the last token of the prefix before that word, and
kappa compares the total log-probability of the two candidate forms of the
critical word (all of their subwords) given the (intervened) prefix.  Items
whose critical word has more than three subwords are filtered.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..alignment.backbone import Backbone
from ..data.pretrained_templates import critical_alternative
from ..data.templates import Sentence

#: Exact checkpoint identifiers (Table 9).
MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m", "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b", "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B", "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B", "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B", "qwen2.5-3b": "Qwen/Qwen2.5-3B", "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "gemma-2-2b": "google/gemma-2-2b",
}
ENGLISH_CORE = ["pythia-160m", "pythia-410m", "pythia-1.4b", "llama-3.2-1b", "llama-3.2-3b"]
GERMAN = ["llama-3.2-1b", "llama-3.2-3b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "gemma-2-2b"]
LARGER = ["pythia-2.8b", "pythia-6.9b", "llama-3.1-8b", "llama-3.1-8b-instruct"]
PYTHIA_REVISION = "step143000"


def uses_bos(name: str) -> bool:
    return name.startswith("llama") or name.startswith("gemma")


def detokenize(words: Sequence[str]) -> str:
    out = ""
    for w in words:
        out += w if (not out or w in (".", ",")) else " " + w
    return out


def load(name: str, random_init: bool = False, seed: Optional[int] = None, device: str = "cuda"):
    """Load a model by short name.  ``random_init`` rebuilds the architecture from its
    configuration with the model class's own initialization (seeded), keeping tokenizer,
    depth, vocabulary and output protocol."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    mid = MODELS[name]
    kw = {"revision": PYTHIA_REVISION} if name.startswith("pythia") else {}
    tok = AutoTokenizer.from_pretrained(mid, **kw)
    if random_init:
        torch.manual_seed(seed or 0)
        cfg = AutoConfig.from_pretrained(mid, **kw)
        cfg._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32,
                                                     attn_implementation="eager", **kw)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def blocks(model) -> torch.nn.ModuleList:
    for path in ("gpt_neox.layers", "model.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise ValueError("unknown architecture")


class HFBackbone(Backbone):
    def __init__(self, model, tok, name: str, device: str = "cuda", max_subwords: int = 3):
        self.model, self.tok, self.name, self.device = model, tok, name, device
        self.layers = blocks(model)
        self.d = model.config.hidden_size
        self.sites = [f"layer{l}" for l in range(1, len(self.layers) + 1)]
        self.max_subwords = max_subwords
        self._cache: Dict = {}

    # ---------------------------------------------------------------- tokenization
    def _ids(self, text: str) -> List[int]:
        memo = self.__dict__.setdefault("_memo", {})
        if text not in memo:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            memo[text] = ([self.tok.bos_token_id] if uses_bos(self.name) else []) + ids
        return memo[text]

    def _raw(self, text: str) -> List[int]:
        memo = self.__dict__.setdefault("_memo_raw", {})
        if text not in memo:
            memo[text] = self.tok(text, add_special_tokens=False)["input_ids"]
        return memo[text]

    def encode_item(self, s: Sentence, word: Optional[str] = None) -> Tuple[List[int], List[int]]:
        """(prefix ids, candidate ids) for the critical word ``word`` (default: the base's)."""
        prefix = detokenize(s.words[:s.target])
        cand = word if word is not None else s.words[s.target]
        pid = self._ids(prefix)
        full = self._raw(prefix + " " + cand)
        n_prefix = len(pid) - (1 if uses_bos(self.name) else 0)
        return pid, full[n_prefix:]

    def position(self, s: Sentence) -> int:
        """Intervention position: the last prefix token before the critical word, or, when
        ``s.meta['intervene_at']`` names a word index (e.g. the German noun for can_nom),
        the last subword of that word."""
        k = s.meta.get("intervene_at")
        if k is None:
            return len(self.encode_item(s)[0]) - 1
        return len(self._ids(detokenize(s.words[:k + 1]))) - 1

    def word_span(self, s: Sentence, k: int) -> List[int]:
        """Token positions of all subwords of word k."""
        a = len(self._ids(detokenize(s.words[:k]))) if k else (1 if uses_bos(self.name) else 0)
        b = len(self._ids(detokenize(s.words[:k + 1])))
        return list(range(a, b))

    @torch.no_grad()
    def attention_rows(self, sents, layer: int, head: int) -> List[torch.Tensor]:
        """The attention row of ``head`` at each sentence's intervention position (prefix only)."""
        out = []
        for s in sents:
            ids, att = self._batch([self.encode_item(s)[0]])
            a = self.model(input_ids=ids, attention_mask=att, output_attentions=True).attentions[layer - 1]
            out.append(a[0, head, self.position(s)].cpu())
        return out

    def keep(self, s: Sentence) -> bool:
        """Filter: the critical word and its alternative have at most three subwords."""
        return all(len(self.encode_item(s, w)[1]) <= self.max_subwords
                   for w in (s.words[s.target], critical_alternative(s)))

    # ---------------------------------------------------------------- activations
    def _hook(self, layer: int, b_idx: torch.Tensor, pos: torch.Tensor, new: torch.Tensor):
        def fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone(); h[b_idx, pos] = new.to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return self.layers[layer - 1].register_forward_hook(fn)

    def _batch(self, seqs: List[List[int]]):
        n = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        return ids.to(self.device), att.to(self.device)

    @torch.no_grad()
    def vectors(self, sents, site, bs: int = 16) -> torch.Tensor:
        layer = int(site.replace("layer", ""))
        out = []
        for i in range(0, len(sents), bs):
            pref = [self.encode_item(s)[0] for s in sents[i:i + bs]]
            ids, att = self._batch(pref)
            pos = torch.tensor([self.position(s) for s in sents[i:i + bs]], device=self.device)
            self.current_positions = pos
            hs = self.model(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[layer]
            out.append(hs[torch.arange(len(pref), device=self.device), pos])
        return torch.cat(out)

    def _candidate_logprob(self, sents, words: List[str], hook=None) -> torch.Tensor:
        seqs, spans = [], []
        for s, w in zip(sents, words):
            p, c = self.encode_item(s, w)
            seqs.append(p + c); spans.append((len(p), len(p) + len(c)))
        ids, att = self._batch(seqs)
        self.current_positions = torch.tensor([self.position(s) for s in sents], device=self.device)
        handle = hook() if hook else None
        try:
            logits = self.model(input_ids=ids, attention_mask=att).logits.float()
        finally:
            if handle:
                handle.remove()
        lp = logits.log_softmax(-1)
        out = []
        for i, (a, b) in enumerate(spans):
            tgt = ids[i, a:b]
            out.append(lp[i, a - 1:b - 1].gather(-1, tgt[:, None]).sum())
        return torch.stack(out)

    def patched_logits(self, sents, site, new, second=None) -> torch.Tensor:
        """Two-class logits [log p(alternative), log p(base word)] under the intervention."""
        layer = int(site.replace("layer", ""))
        pos = torch.tensor([self.position(s) for s in sents], device=self.device)
        b = torch.arange(len(sents), device=self.device)
        hook = lambda: self._hook(layer, b, pos, new)
        base_w = [s.words[s.target] for s in sents]
        alt_w = [critical_alternative(s) for s in sents]
        lp_base = self._candidate_logprob(sents, base_w, hook)
        lp_alt = self._candidate_logprob(sents, alt_w, hook)
        return torch.stack([lp_alt, lp_base], -1)

    @torch.no_grad()
    def logits(self, sents, bs: int = 16) -> torch.Tensor:
        out = []
        for i in range(0, len(sents), bs):
            ch = sents[i:i + bs]
            out.append(torch.stack([self._candidate_logprob(ch, [critical_alternative(s) for s in ch]),
                                    self._candidate_logprob(ch, [s.words[s.target] for s in ch])], -1))
        return torch.cat(out)

    # ---------------------------------------------------------------- sentence scoring
    @torch.no_grad()
    def sentence_logprob(self, texts: Sequence[str], bs: int = 16) -> List[float]:
        out = []
        for i in range(0, len(texts), bs):
            seqs = [self._ids(t) for t in texts[i:i + bs]]
            ids, att = self._batch(seqs)
            lp = self.model(input_ids=ids, attention_mask=att).logits.float().log_softmax(-1)
            for j, s in enumerate(seqs):
                t = ids[j, 1:len(s)]
                out.append(lp[j, :len(s) - 1].gather(-1, t[:, None]).sum().item())
        return out


def minimal_pair_accuracy(bb: HFBackbone, good: Sequence[str], bad: Sequence[str], pair_ids: Sequence[int]) -> float:
    """Fraction of pairs whose grammatical sentence receives the higher log-probability;
    exact ties go to the grammatical sentence for even pair IDs and to the other for odd."""
    g = bb.sentence_logprob(good); b = bb.sentence_logprob(bad)
    wins = [(x > y) or (x == y and pid % 2 == 0) for x, y, pid in zip(g, b, pair_ids)]
    return sum(wins) / len(wins)
