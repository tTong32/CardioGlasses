"""Processing, decision, explainer, and end-to-end pipeline tests (no network)."""

from __future__ import annotations

import functools
import time

import numpy as np
import pytest

from ai.contracts import Config, PatientContext, Reading, Sample
from ai.decision import Context, DecisionEngine
from ai.explainer import Explainer, Wording, template
from ai.pipeline import Pipeline
from ai.processing import HR_MIN_QUALITY, analyze_ppg, classify_activity
from ai.replay import SCENARIOS, generate_scenario

CONTEXT = PatientContext(
    patient_id="patient-demo-polypharmacy",
    age=78,
    conditions=["Atrial fibrillation", "Heart failure", "Hypothyroidism"],
    medications=["24 HR metoprolol succinate 50 MG", "metformin"],
    risk_tier="high",
    config=Config(min_quality=0.6, persist_s=30, deviation_trigger=2.0, cooldown_s=120),
)


@functools.lru_cache(maxsize=None)
def scenario(name: str) -> tuple[Sample, ...]:
    return tuple(generate_scenario(SCENARIOS[name], add_noise_artifacts=(name == "noisy")))


def windows(name: str, step: int = 100, size: int = 500):
    samples = scenario(name)
    sc, t0 = SCENARIOS[name], samples[0].t
    for end in range(size, len(samples) + 1, step):
        win = list(samples[end - size:end])
        times = [(s.t - t0) / 1000 for s in win]
        moving = np.mean([sc.motion_fn(t) for t in times]) > 0
        truth = float(np.mean([sc.hr_fn(t) for t in times]))
        yield win, truth, moving


SINUS = CONTEXT.model_copy(update={"conditions": ["Heart failure", "Hypothyroidism"]})  # no AF on record


def run(name: str, context: PatientContext = CONTEXT) -> list:
    return list(Pipeline(context).run(scenario(name)))


def alert_levels(name: str, context: PatientContext = CONTEXT) -> list[str]:
    return [step.alert.level for step in run(name, context) if step.alert is not None]


def af_rhythm(seed: int = 12, jitter: float = 18.0) -> list[Sample]:
    from ai.replay import Scenario, _constant_hr

    return generate_scenario(Scenario("af", 60, _constant_hr(80), seed=seed, jitter_percent=jitter))


def reading(**overrides) -> Reading:
    base = dict(
        t=0, hr=100.0, ibi_ms=None, activity="resting", quality=0.95, baseline_hr=68.0, deviation=8.0,
        persist_s=0.0, recovery_tau_s=None, hr_drop_60s=None, recovery_ratio=None,
        recovery_percentile=None, recovery_verdict=None, signal_status="ok",
    )
    base.update(overrides)
    return Reading(**base)


# ---------- Processing ----------

@pytest.mark.parametrize("name", ["rest", "normal_recovery", "slow_recovery", "elevated_rest"])
def test_hr_within_5_bpm_at_rest(name):
    errors = [abs(analyze_ppg(win).hr - truth) for win, truth, moving in windows(name) if not moving]
    assert errors and max(errors) < 5


def test_clean_signal_scores_high_quality():
    qualities = [analyze_ppg(win).quality for win, _, moving in windows("rest")]
    assert min(qualities) >= 0.9


def test_noisy_signal_scores_low_and_withholds_hr():
    results = [analyze_ppg(win) for win, _, _ in windows("noisy")]
    assert all(r.quality < 0.6 for r in results)
    assert all(r.hr is None for r in results if r.quality < HR_MIN_QUALITY)


def test_flat_signal_has_zero_quality():
    flat = [s.model_copy(update={"ppg": 50000}) for s in scenario("rest")[:500]]
    result = analyze_ppg(flat)
    assert result.hr is None and result.quality == 0 and "flat" in result.reason


def test_missing_ppg_is_reported():
    holes = [s.model_copy(update={"ppg": None if i % 3 == 0 else s.ppg}) for i, s in enumerate(scenario("rest")[:500])]
    assert "missing" in analyze_ppg(holes).reason


def test_ibis_match_heart_rate():
    win, truth, _ = next(windows("rest"))
    result = analyze_ppg(win)
    assert 60000 / np.median(result.ibi_ms) == pytest.approx(truth, abs=3)


def test_irregular_mode_trusts_an_af_rhythm():
    samples = af_rhythm()
    wins = [samples[e - 500:e] for e in range(500, len(samples) + 1, 100)]
    normal = np.mean([analyze_ppg(w).quality >= 0.6 for w in wins])
    irregular = np.mean([analyze_ppg(w, irregular_rhythm=True).quality >= 0.6 for w in wins])
    assert irregular >= 0.9 and irregular > normal


def test_irregular_mode_still_rejects_noise():
    assert all(analyze_ppg(win, irregular_rhythm=True).quality < 0.6 for win, _, _ in windows("noisy"))
    rng = np.random.default_rng(3)
    noisy_af = [s.model_copy(update={"ppg": int(s.ppg + rng.normal(0, 600))}) for s in af_rhythm(seed=13)]
    wins = [noisy_af[e - 500:e] for e in range(500, len(noisy_af) + 1, 100)]
    assert np.mean([analyze_ppg(w, irregular_rhythm=True).quality >= 0.6 for w in wins]) < 0.1


def test_stateless_activity():
    win, _, _ = next(windows("rest"))
    assert classify_activity(win) == "resting"
    moving = next(w for w, _, m in windows("moving") if m)
    assert classify_activity(moving) == "moving"


# ---------- Decision rules ----------

def test_unexplained_elevation_escalates_in_steps():
    engine = DecisionEngine(CONTEXT)
    seen = []
    for i, persist in enumerate([0, 10, 30, 60, 90]):
        alert = engine.update(reading(t=i * 1000, persist_s=persist))
        seen.append(alert.level if alert else None)
    assert seen == [None, "monitor", "notify", None, "escalate"]


def test_escalate_needs_a_large_deviation():
    engine = DecisionEngine(CONTEXT)
    engine.update(reading(persist_s=30, deviation=3.0))
    assert engine.update(reading(t=1, persist_s=120, deviation=3.0)) is None
    assert engine.level == "notify"


@pytest.mark.parametrize(
    "overrides",
    [
        {"quality": 0.4},
        {"signal_status": "poor"},
        {"hr": None},
        {"activity": "moving"},
        {"deviation": None},
    ],
)
def test_untrusted_or_moving_never_alerts(overrides):
    engine = DecisionEngine(CONTEXT)
    assert engine.update(reading(persist_s=200, **overrides)) is None
    assert engine.level == "normal"


def test_recovery_after_exertion_is_explained():
    engine = DecisionEngine(CONTEXT)
    assert engine.update(reading(persist_s=60), Context(recovering=True, live_verdict="normal")) is None
    assert engine.update(reading(t=1, persist_s=60), Context(recovering=True, live_verdict=None)) is None
    assert engine.update(reading(t=2, persist_s=60), Context(seconds_since_moving=30)) is None


@pytest.mark.parametrize(("verdict", "level"), [("slow", "monitor"), ("very_slow", "notify")])
def test_slow_recovery_alerts(verdict, level):
    engine = DecisionEngine(CONTEXT)
    alert = engine.update(reading(persist_s=5), Context(recovering=True, live_verdict=verdict))
    assert alert.level == level
    assert alert.headline in ("Slower recovery than usual", "Heart rate slow to settle")


def test_untrusted_reading_does_not_restart_the_normal_timer():
    engine = DecisionEngine(CONTEXT)
    engine.update(reading(t=0, persist_s=30))
    engine.update(reading(t=2_000, deviation=0.5))
    engine.update(reading(t=10_000, deviation=0.5, quality=0.3, signal_status="poor"))  # no evidence
    assert engine.update(reading(t=22_000, deviation=0.5)).level == "normal"


def test_return_to_normal_needs_20s_and_then_cools_down():
    engine = DecisionEngine(CONTEXT)
    engine.update(reading(t=0, persist_s=30))
    assert engine.level == "notify"
    assert engine.update(reading(t=2_000, deviation=0.5, persist_s=0)) is None
    assert engine.update(reading(t=12_000, deviation=0.5, persist_s=0)) is None
    alert = engine.update(reading(t=22_000, deviation=0.5, persist_s=0))
    assert alert.level == "normal"
    # Within cooldown_s: the same level can't be raised again...
    assert engine.update(reading(t=40_000, persist_s=30)) is None
    # ...but a higher one can.
    assert engine.update(reading(t=41_000, persist_s=90)).level == "escalate"


def test_brief_dip_resets_the_normal_timer():
    engine = DecisionEngine(CONTEXT)
    engine.update(reading(t=0, persist_s=30))
    engine.update(reading(t=2_000, deviation=0.5))
    engine.update(reading(t=10_000, deviation=1.5))  # in between: hold, timer resets
    assert engine.update(reading(t=24_000, deviation=0.5)) is None


def test_low_confidence_caps_at_monitor():
    loose = CONTEXT.model_copy(update={"config": CONTEXT.config.model_copy(update={"min_quality": 0.3})})
    engine = DecisionEngine(loose)
    alert = engine.update(reading(persist_s=30, quality=0.4))
    assert alert.level == "monitor"


def test_alert_carries_reading_and_fixed_next_step():
    alert = DecisionEngine(CONTEXT).update(reading(persist_s=30))
    assert alert.reading.hr == 100
    assert alert.next_step == "Sit down and check your phone"
    assert "atrial fibrillation" in alert.reason


# ---------- Explainer ----------

class FakeModels:
    def __init__(self, result=None, error=None, delay=0.0):
        self.result, self.error, self.delay, self.calls = result, error, delay, 0

    def generate_content(self, **kwargs):
        self.calls += 1
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return type("Response", (), {"parsed": self.result})()


def fake_explainer(models: FakeModels, timeout_s: float = 15.0) -> Explainer:
    explainer = Explainer(use_llm=False, timeout_s=timeout_s)
    explainer._client = type("Client", (), {"models": models})()
    return explainer


def test_gemini_wording_gets_fixed_action_and_is_cached():
    models = FakeModels(Wording(reason="Your heart rate is 100, above your usual 68.", voice_text="Your resting heart rate is up."))
    explainer = fake_explainer(models)
    first = explainer.explain("notify", "unexplained_elevation", reading(persist_s=30), CONTEXT)
    assert first.source == "gemini"
    assert first.voice_text == "Your resting heart rate is up. Please sit down and check your phone."
    assert first.next_step == "Sit down and check your phone"
    assert explainer.explain("notify", "unexplained_elevation", reading(persist_s=31), CONTEXT).source == "cache"
    assert models.calls == 1


@pytest.mark.parametrize(
    "wording",
    [
        Wording(reason="This could be a heart attack.", voice_text="Something changed."),
        Wording(reason="Fine.", voice_text="Please call your doctor right now."),
        Wording(reason="Fine.", voice_text=" ".join(["word"] * 30)),
    ],
)
def test_unsafe_or_long_wording_falls_back(wording):
    result = fake_explainer(FakeModels(wording)).explain("notify", "unexplained_elevation", reading(), CONTEXT)
    assert result.source == "template"


def test_errors_and_slow_responses_fall_back():
    failing = fake_explainer(FakeModels(error=RuntimeError("503")))
    assert failing.explain("notify", "unexplained_elevation", reading(), CONTEXT).source == "template"
    slow = fake_explainer(FakeModels(Wording(reason="ok", voice_text="ok"), delay=1.5), timeout_s=0.1)
    slow.explain("normal", "back_to_normal", reading(deviation=0.2), CONTEXT)  # warm the SDK import
    started = time.time()
    assert slow.explain("notify", "unexplained_elevation", reading(), CONTEXT).source == "template"
    assert time.time() - started < 1.0


def test_templates_cover_every_level_and_kind():
    for level, kind in [("monitor", "unexplained_elevation"), ("notify", "unexplained_elevation"),
                        ("escalate", "unexplained_elevation"), ("monitor", "slow_recovery"),
                        ("notify", "slow_recovery"), ("escalate", "slow_recovery"), ("normal", "back_to_normal")]:
        words = template(level, kind, reading(persist_s=30), CONTEXT)
        assert words.headline and words.reason and words.voice_text and words.next_step


# ---------- End to end on synthetic scenarios ----------

@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("rest", []),
        ("moving", []),
        ("noisy", []),
        ("normal_recovery", []),
        ("elevated_rest", ["monitor", "notify", "escalate", "normal"]),
        ("af_elevated_rest", ["monitor", "notify", "escalate", "normal"]),
        # AF on record widens the baseline floor to 6 bpm, so these stop short of escalate.
        ("slow_recovery", ["notify"]),
        ("calibration_then_slow", ["notify"]),
    ],
)
def test_scenario_alerts_af_patient(name, expected):
    assert alert_levels(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("rest", []),
        ("normal_recovery", []),
        ("elevated_rest", ["monitor", "notify", "escalate", "normal"]),
        ("slow_recovery", ["notify", "escalate"]),
        ("calibration_then_slow", ["notify", "escalate"]),
    ],
)
def test_scenario_alerts_sinus_patient(name, expected):
    assert alert_levels(name, SINUS) == expected


def test_af_record_keeps_an_af_rhythm_trusted():
    def trusted(steps):
        return sum(s.reading.signal_status == "ok" for s in steps) / len(steps)

    assert trusted(run("af_elevated_rest", CONTEXT)) > 0.85
    assert trusted(run("af_elevated_rest", SINUS)) < 0.6
    assert alert_levels("af_elevated_rest", SINUS) != ["monitor", "notify", "escalate", "normal"]


def test_slow_recovery_is_named_and_fit_is_close():
    steps = run("slow_recovery", SINUS)
    assert next(s.alert for s in steps if s.alert).headline == "Heart rate slow to settle"
    taus = [s.reading.recovery_tau_s for s in steps if s.reading.recovery_tau_s]
    assert 100 < taus[-1] < 220  # true tau is 150 s


def test_normal_recovery_fit_is_close():
    taus = [s.reading.recovery_tau_s for s in run("normal_recovery", SINUS) if s.reading.recovery_tau_s]
    assert taus and 20 < taus[-1] < 50  # true tau is 30 s


def test_every_reading_is_a_valid_contract_b():
    for step in run("calibration_then_slow"):
        Reading.model_validate(step.reading.model_dump())
        assert step.reading.signal_status in ("ok", "poor", "offline")


def test_baseline_is_not_absorbed_by_an_episode():
    baselines = [s.reading.baseline_hr for s in run("elevated_rest") if s.reading.baseline_hr is not None]
    assert max(baselines) < 72


def test_noisy_reports_poor_signal():
    steps = run("noisy")
    assert all(s.reading.signal_status == "poor" for s in steps)
    assert all(s.reading.hr is None or s.reading.quality < 0.6 for s in steps)


def test_data_gap_is_not_analysed_across():
    samples = list(scenario("rest"))
    gapped = samples[:3000] + [s.model_copy(update={"t": s.t + 10_000}) for s in samples[3000:]]
    steps = list(Pipeline(CONTEXT).run(gapped))
    times = [s.reading.t for s in steps]
    gap_start = samples[2999].t
    # No reading until a fresh 6 s window has built up after the gap.
    assert not any(gap_start < t < gap_start + 10_000 + 6_000 for t in times)


def test_out_of_order_samples_are_ignored():
    samples = list(scenario("rest")[:1000])
    shuffled = samples[:500] + [samples[100]] + samples[500:]
    assert len(list(Pipeline(CONTEXT).run(shuffled))) == len(list(Pipeline(CONTEXT).run(samples)))
