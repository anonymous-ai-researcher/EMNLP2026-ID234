"""Behavioral evaluation: minimal-pair accuracy (Section 4)."""
import torch

def minimal_pair_accuracy(model, grammatical, ungrammatical, target_pos):
    """Fraction of pairs where model assigns higher probability to grammatical."""
    model.eval()
    correct = 0
    total = len(grammatical)
    with torch.no_grad():
        for gram, ungram in zip(grammatical, ungrammatical):
            logits_g = model(gram.unsqueeze(0), target_pos)
            logits_u = model(ungram.unsqueeze(0), target_pos)
            if logits_g[0, 1] > logits_u[0, 1]:
                correct += 1
    return correct / total
