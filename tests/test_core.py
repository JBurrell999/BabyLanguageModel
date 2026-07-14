"""Integration tests: mixer batch geometry, loss parity of the per-token
forward against the vendored GPT-BERT forward, adaptive masking, Muon
orthogonalization, and exposure accounting."""

import json
import math

import pytest
import torch

from src.objectives.mixer import ObjectiveMixer, _round_composition
from src.objectives.adaptive_mask import DifficultyTable, AdaptiveSpanMasking
from src.optim.muon import (Muon, newton_schulz_orthogonalize,
                            split_params_for_muon, HybridOptimizer)
from src.pretrain.exposure import ExposureCounter, EpochCapExceeded
from src.pretrain.hybrid_forward import hybrid_forward
from src.model.model import Bert
from src.model.dataset import SpanMaskingStrategy


# ---------------------------------------------------------------------------
# helpers: tiny synthetic datasets shaped like GPT-BERT's
# ---------------------------------------------------------------------------

L = 32          # trainer sees seq_length-1 sized items in the real datasets;
                # here we just fix a length
VOCAB = 128
PAD = 3


class ToyDataset(torch.utils.data.Dataset):
    """Emits (input_ids [L], target_ids [L], attention_mask [L, L], mask_p)
    with a marker so tests can tell which stream an example came from."""

    def __init__(self, marker: int, causal: bool, n: int = 512):
        self.marker = marker
        self.causal = causal
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i + self.marker * 100000)
        input_ids = torch.randint(4, VOCAB, (L,), generator=g)
        input_ids[0] = self.marker
        target_ids = torch.where(torch.rand(L, generator=g) < 0.3,
                                 input_ids, torch.full((L,), -100))
        att = torch.zeros(L, L, dtype=torch.bool)
        if self.causal:
            att = ~torch.ones(L, L, dtype=torch.bool).tril()
        return input_ids, target_ids, att, torch.tensor(0.3)


MASKED_MARK, CAUSAL_MARK = 1, 2


def make_mixer(batch_size=16, ratio_mode="composition"):
    return ObjectiveMixer(ToyDataset(MASKED_MARK, causal=False),
                          ToyDataset(CAUSAL_MARK, causal=True),
                          batch_size=batch_size, ratio_mode=ratio_mode,
                          num_workers=0, seed=0)


# ---------------------------------------------------------------------------
# mixer
# ---------------------------------------------------------------------------

def test_round_composition_bounds():
    assert _round_composition(16, 0.0) == 0
    assert _round_composition(16, 1.0) == 16
    assert _round_composition(16, 0.5) == 8
    assert _round_composition(16, 2.0) == 16


@pytest.mark.parametrize("p,expected", [(0.0, 0), (0.25, 4), (0.5, 8),
                                        (0.9375, 15), (1.0, 16)])
def test_mixer_composition_and_order(p, expected):
    mx = make_mixer(16)
    b = mx.next_batch(p, global_step=0)
    assert b["num_masked"] == expected
    # seq-first layout: markers live at input_ids[0, :]
    markers = b["input_ids"][0]
    assert (markers[:expected] == MASKED_MARK).all()
    assert (markers[expected:] == CAUSAL_MARK).all()
    # causal columns have lower-triangular visibility (mask True = blocked)
    if expected < 16:
        att = b["attention_mask"][expected]          # first causal example
        assert att[0, 1]                             # future blocked
        assert not att[1, 0]                         # past visible


def test_mixer_shapes_seq_first():
    mx = make_mixer(8)
    b = mx.next_batch(0.5, global_step=0)
    assert b["input_ids"].shape == (L, 8)
    assert b["target_ids"].shape == (L, 8)
    assert b["attention_mask"].shape == (8, L, L)


def test_ratio_modes():
    mx = make_mixer(16, ratio_mode="composition")
    assert mx.next_batch(0.9375, 0)["ratio"] == pytest.approx(15 / 16)
    mx = make_mixer(16, ratio_mode="schedule")
    assert mx.next_batch(0.9, 0)["ratio"] == pytest.approx(0.9)
    mx = make_mixer(16, ratio_mode="fixed:0.5")
    assert mx.next_batch(0.9, 0)["ratio"] == pytest.approx(0.5)


def test_mixer_streams_are_endless():
    mx = make_mixer(16)
    for step in range(200):  # 3200 examples > 2x dataset size: must not stop
        mx.next_batch(0.5, step)
    assert mx.masked.epoch >= 1


# ---------------------------------------------------------------------------
# hybrid forward: parity with vendored Bert.forward
# ---------------------------------------------------------------------------

class Cfg:
    attention_probs_dropout_prob = 0.0
    hidden_dropout_prob = 0.0
    hidden_size = 48
    intermediate_size = 96
    max_position_embeddings = 64
    position_bucket_size = 8
    num_attention_heads = 4
    num_hidden_layers = 2
    vocab_size = VOCAB
    layer_norm_eps = 1e-5


def _mixed_batch(num_masked=5, batch=8):
    mx = make_mixer(batch)
    return mx.next_batch(num_masked / batch, 0)


def test_hybrid_forward_matches_reference_losses():
    torch.manual_seed(0)
    model = Bert(Cfg()).eval()
    b = _mixed_batch()
    with torch.no_grad():
        ref = model(b["input_ids"], b["attention_mask"], b["target_ids"],
                    num_masked=b["num_masked"], ratio=b["ratio"])
        ref_loss, ref_mlm, ref_clm = ref[0], ref[1], ref[2]
        ref_z = ref[6]
        out = hybrid_forward(model, b["input_ids"], b["attention_mask"],
                             b["target_ids"], b["num_masked"], b["ratio"])
    assert float(out.loss) == pytest.approx(float(ref_loss), rel=1e-5)
    assert out.masked_loss == pytest.approx(float(ref_mlm), rel=1e-5)
    assert out.causal_loss == pytest.approx(float(ref_clm), rel=1e-5)
    assert float(out.z_loss) == pytest.approx(float(ref_z), rel=1e-5)


def test_hybrid_forward_all_masked_and_all_causal():
    torch.manual_seed(0)
    model = Bert(Cfg()).eval()
    for p in (0.0, 1.0):
        b = _mixed_batch(num_masked=int(8 * p), batch=8)
        with torch.no_grad():
            out = hybrid_forward(model, b["input_ids"], b["attention_mask"],
                                 b["target_ids"], b["num_masked"], b["ratio"])
        assert math.isfinite(float(out.loss))
        if p == 1.0:
            assert out.num_causal_tokens == 0 and out.num_masked_tokens > 0
        else:
            assert out.num_masked_tokens == 0 and out.num_causal_tokens > 0


def test_per_token_losses_average_to_scalar():
    torch.manual_seed(0)
    model = Bert(Cfg()).eval()
    b = _mixed_batch()
    with torch.no_grad():
        out = hybrid_forward(model, b["input_ids"], b["attention_mask"],
                             b["target_ids"], b["num_masked"], b["ratio"])
    assert out.masked_token_losses.numel() == out.num_masked_tokens
    assert float(out.masked_token_losses.mean()) == \
        pytest.approx(out.masked_loss, rel=1e-5)


# ---------------------------------------------------------------------------
# adaptive masking
# ---------------------------------------------------------------------------

def test_difficulty_table_update_and_normalize():
    t = DifficultyTable(vocab_size=50, beta=0.5)
    ids = torch.tensor([10, 10, 20])
    losses = torch.tensor([4.0, 2.0, 1.0])
    t.update(ids, losses)                       # id 10 mean = 3.0, id 20 = 1.0
    assert t.ema[10] == pytest.approx(3.0)
    assert t.ema[20] == pytest.approx(1.0)
    norm = t.normalized(torch.tensor([10, 20, 30]))
    assert norm[0] == pytest.approx(1.0)        # hardest
    assert norm[1] == pytest.approx(0.0)        # easiest
    assert norm[2] == pytest.approx(0.5)        # unseen -> neutral


def test_adaptive_masking_prefers_hard_tokens():
    torch.manual_seed(0)
    base = SpanMaskingStrategy(n_special_tokens=4, random_p=0.0, keep_p=0.0,
                               vocab_size=VOCAB, mask_token_id=0)
    table = DifficultyTable(VOCAB)
    hard, easy = 40, 41
    table.update(torch.tensor([hard, easy]), torch.tensor([10.0, 0.1]))
    strat = AdaptiveSpanMasking(base, table, strength=0.9)

    tokens = torch.tensor([hard, easy] * 32)
    hard_first = 0
    trials = 300
    for _ in range(trials):
        ratios, _ = strat(tokens)
        finite = torch.isfinite(ratios)
        order = torch.argsort(ratios[finite])
        first = tokens[finite][order[0]].item()
        hard_first += (first == hard)
    # hard token should be first-to-mask far more than half the time
    assert hard_first / trials > 0.7


def test_adaptive_masking_zero_strength_is_identity():
    torch.manual_seed(1)
    base = SpanMaskingStrategy(4, 0.1, 0.1, VOCAB, 0)
    strat = AdaptiveSpanMasking(base, DifficultyTable(VOCAB), strength=0.0)
    tokens = torch.randint(4, VOCAB, (64,))
    torch.manual_seed(7)
    r1, t1 = base(tokens)
    torch.manual_seed(7)
    r2, t2 = strat(tokens)
    assert torch.equal(r1, r2) and torch.equal(t1, t2)


# ---------------------------------------------------------------------------
# muon
# ---------------------------------------------------------------------------

def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(32, 64)
    X = newton_schulz_orthogonalize(G, steps=5).float()
    # Muon's quintic NS converges singular values into a loose band around 1
    # (not exact orthogonality — that's by design). Check the band, and that
    # the spread collapses relative to the input.
    sv_in = torch.linalg.svdvals(G)
    sv_out = torch.linalg.svdvals(X)
    assert sv_out.min() > 0.3 and sv_out.max() < 1.6
    spread_in = float(sv_in.max() / sv_in.min())
    spread_out = float(sv_out.max() / sv_out.min())
    assert spread_out < spread_in / 3


def test_muon_rejects_non_2d():
    p = torch.nn.Parameter(torch.randn(10))
    with pytest.raises(ValueError):
        Muon([p])


def test_param_split_keeps_embeddings_and_tied_head_off_muon():
    model = Bert(Cfg())
    muon_params, adamw_params = split_params_for_muon(model)
    emb = model.embedding.word_embedding.weight
    assert all(id(p) != id(emb) for p in muon_params)
    assert all(p.ndim == 2 for p in muon_params)
    n_total = sum(1 for _ in model.parameters())
    assert len(muon_params) + len(adamw_params) == n_total
    assert len(muon_params) > 0


def test_hybrid_optimizer_decreases_loss():
    torch.manual_seed(0)
    model = Bert(Cfg())
    opt = HybridOptimizer(model, muon_lr=0.01, adamw_lr=1e-3)
    b = _mixed_batch()
    losses = []
    for _ in range(8):
        opt.zero_grad()
        out = hybrid_forward(model, b["input_ids"], b["attention_mask"],
                             b["target_ids"], b["num_masked"], b["ratio"])
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss))
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------

def test_exposure_epochs_and_cap():
    c = ExposureCounter(corpus_words=1000, words_per_token=0.5, max_epochs=2)
    c.add_tokens(4000)          # = 2000 words = 2.0 epochs: at cap, ok
    c.check()
    c.add_tokens(10)
    with pytest.raises(EpochCapExceeded):
        c.check()


def test_exposure_milestones_fire_once_each():
    c = ExposureCounter(corpus_words=100_000_000, words_per_token=1.0)
    c.add_tokens(2_500_000)
    assert c.due_milestones() == [1_000_000, 2_000_000]
    assert c.due_milestones() == []
    c.add_tokens(500_000)
    assert c.due_milestones() == [3_000_000]


def test_exposure_state_roundtrip():
    c = ExposureCounter(corpus_words=10_000_000, words_per_token=0.8)
    c.add_tokens(3_000_000)
    c.due_milestones()
    s = c.state_dict()
    c2 = ExposureCounter(corpus_words=10_000_000, words_per_token=0.8)
    c2.load_state_dict(s)
    assert c2.tokens_seen == c.tokens_seen
    assert c2.due_milestones() == []
