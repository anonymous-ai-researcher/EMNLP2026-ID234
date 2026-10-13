"""Exercise the pre-trained interface on tiny randomly initialized models with a local
word-level tokenizer (no download)."""
import random

import pytest
import torch

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from spec2verify.alignment import das as DAS
from spec2verify.data import pairs as PR
from spec2verify.data import pretrained_templates as PT
from spec2verify.pretrained import hf as HF


def tiny(kind: str):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    words = set()
    for ph in PT.GENERATORS:
        for f in PT.pretrained_pool(ph, 1, 0):
            for s in f.sentences:
                words |= set(s.words) | {PT.critical_alternative(s)}
    vocab = {w: i for i, w in enumerate(["<pad>", "<s>"] + sorted(words))}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", bos_token="<s>")
    if kind == "pythia":
        cfg = transformers.GPTNeoXConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                                         num_attention_heads=4, intermediate_size=64, max_position_embeddings=64)
    else:
        cfg = transformers.LlamaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
                                       max_position_embeddings=64, bos_token_id=1)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval()
    return model, tok


@pytest.mark.parametrize("kind,name", [("pythia", "pythia-160m"), ("llama", "llama-3.2-1b")])
def test_backbone_interface(kind, name):
    model, tok = tiny(kind)
    bb = HF.HFBackbone(model, tok, name, device="cpu")
    fams = PT.pretrained_pool("agreement", 2, 0)
    sents = [s for f in fams for s in f.sentences if bb.keep(s)]
    assert sents
    v = bb.vectors(sents[:6], "layer1")
    assert v.shape == (6, 32)
    new = v.clone().requires_grad_(True)
    out = bb.patched_logits(sents[:6], "layer1", new)
    out.sum().backward()
    assert new.grad is not None and out.shape == (6, 2)
    # the unpatched forward equals the clean scores
    assert torch.allclose(bb.patched_logits(sents[:6], "layer1", v), bb.logits(sents[:6]), atol=1e-4)
    rng = random.Random(0)
    pairs = PR.split_by_family(PR.make_pairs(sents, "subj_num", rng, PR.scm_for("agreement")), rng)
    res = DAS.refit_fixed(bb, pairs, "layer2", 1, 701, epochs=1)
    assert 0.0 <= res.test_iia <= 1.0
    acc = HF.minimal_pair_accuracy(bb, [HF.detokenize(f.good.words) for f in fams],
                                   [HF.detokenize(f.bad.words) for f in fams], list(range(len(fams))))
    assert 0.0 <= acc <= 1.0


@pytest.mark.parametrize("kind,name", [("pythia", "pythia-160m"), ("llama", "llama-3.2-1b")])
def test_head_ablation_changes_output(kind, name):
    from spec2verify.pretrained import attention as ATT
    model, tok = tiny(kind)
    bb = HF.HFBackbone(model, tok, name, device="cpu")
    sents = [s for f in PT.pretrained_pool("binding", 1, 0) for s in f.sentences if bb.keep(s)][:4]
    before = bb.logits(sents)
    pos = torch.tensor([bb.position(s) for s in sents])
    with ATT.ablate_hf_head(model, 2, 1, pos):
        after = bb.logits(sents)
    assert not torch.allclose(before, after)
    assert torch.allclose(bb.logits(sents), before)
