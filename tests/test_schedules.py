import math

import pytest

from src.objectives.schedules import (
    FixedSchedule, LinearSchedule, CosineSchedule, StepSchedule,
    LossAdaptiveSchedule, SlopeAdaptiveSchedule, LossFeedback, build_schedule,
)


def test_fixed_matches_official_baselines():
    for p in (15 / 16, 1 / 2, 1 / 16):
        s = FixedSchedule(p, p_min=0.0, p_max=1.0)
        assert s.step(0, 100) == pytest.approx(p)
        assert s.step(99, 100) == pytest.approx(p)


def test_linear_endpoints_and_midpoint():
    s = LinearSchedule(0.9, 0.1, p_min=0.0, p_max=1.0)
    assert s.step(0, 100) == pytest.approx(0.9)
    assert s.step(50, 100) == pytest.approx(0.5)
    assert s.step(100, 100) == pytest.approx(0.1)


def test_cosine_endpoints_and_monotone():
    s = CosineSchedule(0.9, 0.1, p_min=0.0, p_max=1.0)
    vals = [s.step(t, 100) for t in range(101)]
    assert vals[0] == pytest.approx(0.9)
    assert vals[-1] == pytest.approx(0.1)
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))  # monotone dec.


def test_step_schedule_switches():
    s = StepSchedule(boundaries=[0.5], values=[0.9, 0.1],
                     p_min=0.0, p_max=1.0)
    assert s.step(10, 100) == pytest.approx(0.9)
    assert s.step(60, 100) == pytest.approx(0.1)


def test_clamping():
    s = LinearSchedule(1.0, 0.0, p_min=0.05, p_max=0.95)
    assert s.step(0, 100) == pytest.approx(0.95)
    assert s.step(100, 100) == pytest.approx(0.05)


def test_loss_adaptive_moves_toward_needier_objective():
    s = LossAdaptiveSchedule(warmup_steps=0, update_every=1,
                             temperature=0.05, p_min=0.05, p_max=0.95)
    # equal history first so slow EMAs are populated
    for t in range(50):
        s.step(t, 1000, LossFeedback(3.0, 3.0, 100, 100))
    p_eq = s.step(50, 1000, LossFeedback(3.0, 3.0, 100, 100))
    # masked loss spikes -> masked is "needier" -> p should rise
    for t in range(51, 120):
        s.step(t, 1000, LossFeedback(4.0, 3.0, 100, 100))
    p_hi = s.step(120, 1000, LossFeedback(4.0, 3.0, 100, 100))
    assert p_hi > p_eq
    # causal spikes instead -> p should fall below the equal point
    s2 = LossAdaptiveSchedule(warmup_steps=0, update_every=1,
                              temperature=0.05, p_min=0.05, p_max=0.95)
    for t in range(50):
        s2.step(t, 1000, LossFeedback(3.0, 3.0, 100, 100))
    for t in range(50, 120):
        s2.step(t, 1000, LossFeedback(3.0, 4.0, 100, 100))
    assert s2.step(120, 1000, LossFeedback(3.0, 4.0, 100, 100)) < p_eq


def test_slope_adaptive_moves_toward_faster_improver():
    s = SlopeAdaptiveSchedule(warmup_steps=0, update_every=1,
                              temperature=0.02, p_min=0.05, p_max=0.95)
    # masked loss falls quickly, causal flat -> masked improving faster -> p up
    lm = 5.0
    for t in range(300):
        lm = max(1.0, lm - 0.02)
        s.step(t, 1000, LossFeedback(lm, 4.0, 100, 100))
    assert s.step(300, 1000, LossFeedback(lm, 4.0, 100, 100)) > 0.5


def test_adaptive_warmup_holds_p_warmup():
    s = LossAdaptiveSchedule(warmup_steps=100, p_warmup=0.7,
                             p_min=0.05, p_max=0.95)
    assert s.step(0, 1000, LossFeedback(9.0, 1.0, 10, 10)) == pytest.approx(0.7)
    assert s.step(99, 1000, LossFeedback(9.0, 1.0, 10, 10)) == pytest.approx(0.7)


def test_state_dict_roundtrip_resumes_exactly():
    s = LossAdaptiveSchedule(warmup_steps=0, update_every=1)
    for t in range(40):
        s.step(t, 100, LossFeedback(3.0 - 0.01 * t, 3.5, 10, 10))
    state = s.state_dict()
    p_next = s.step(40, 100, LossFeedback(2.6, 3.5, 10, 10))

    s2 = LossAdaptiveSchedule(warmup_steps=0, update_every=1)
    s2.load_state_dict(state)
    assert s2.step(40, 100, LossFeedback(2.6, 3.5, 10, 10)) == \
        pytest.approx(p_next)


def test_build_schedule_from_config():
    s = build_schedule({"name": "linear", "p_start": 0.9375,
                        "p_end": 0.0625, "p_min": 0.05, "p_max": 0.95})
    assert isinstance(s, LinearSchedule)
    with pytest.raises(KeyError):
        build_schedule({"name": "nope"})
