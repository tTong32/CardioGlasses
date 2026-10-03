"""How the FinchNode record changes monitoring."""

from __future__ import annotations

import math

import pytest

from ai import clinical
from ai.contracts import Config, PatientContext
from ai.recovery import RecoveryModel

BASE = Config(min_quality=0.6, persist_s=30, deviation_trigger=2.0, cooldown_s=120)


def patient(conditions=(), medications=(), tier="high") -> PatientContext:
    return PatientContext(
        patient_id="p", age=78, conditions=list(conditions), medications=list(medications),
        risk_tier=tier, config=BASE,
    )


@pytest.mark.parametrize(("tier", "trigger", "persist"), [("high", 2.0, 30), ("medium", 2.5, 45), ("low", 3.0, 60)])
def test_tier_sets_sensitivity(tier, trigger, persist):
    tuned = clinical.tuned_config(BASE, tier)
    assert (tuned.deviation_trigger, tuned.persist_s) == (trigger, persist)
    assert (tuned.min_quality, tuned.cooldown_s) == (BASE.min_quality, BASE.cooldown_s)


def test_unknown_tier_keeps_base():
    assert clinical.tuned_config(BASE, "unknown") == BASE


def test_irregular_rhythm_from_conditions():
    assert clinical.irregular_rhythm_expected(patient(["Atrial fibrillation"]))
    assert clinical.irregular_rhythm_expected(patient(["Paroxysmal atrial flutter"]))
    assert not clinical.irregular_rhythm_expected(patient(["Heart failure"]))
    assert clinical.baseline_min_std(patient(["Atrial fibrillation"])) == 6.0
    assert clinical.baseline_min_std(patient(["Heart failure"])) == 4.0


def test_effects_describe_the_record():
    lines = clinical.effects(patient(
        ["Atrial fibrillation", "Heart failure"],
        ["24 HR metoprolol succinate 50 MG Extended Release Oral Tablet"],
    ))
    assert lines[0] == "High risk: alerts if your resting heart rate stays 12+ bpm above your usual for 30 seconds."
    assert lines[1].startswith("Atrial fibrillation: an irregular beat is expected")
    assert lines[2].startswith("On metoprolol:")
    assert clinical.effects(patient(["Asthma"]))[0].startswith("High risk: alerts if your resting heart rate stays 8+ bpm")


def test_effects_mention_the_clinic_starting_point():
    with_hr = patient(["Heart failure"]).model_copy(update={"clinic_resting_hr": 73.5})
    assert any(line.startswith("Clinic heart rate 74:") for line in clinical.effects(with_hr))
    assert not any("Clinic" in line for line in clinical.effects(patient(["Heart failure"])))


def test_recovery_start_uses_first_clean_hr_when_stop_is_unreadable():
    model = RecoveryModel(medications=[])
    t = 0
    for _ in range(20):  # 40 s moving, HR readable, peak 100
        model.update(t, 100.0, "moving", 68.0, 2.0)
        t += 2000
    model.update(t, None, "resting", 68.0, 2.0)  # the stop: motion corrupts the reading
    t += 2000
    for i in range(30):  # recovery from 110 with tau 30 s
        model.update(t, 68 + 42 * math.exp(-(i * 2 + 2) / 30), "resting", 68.0, 2.0)
        t += 2000
    assert model._hr0 is not None and model._hr0 > 100


def test_one_medication_list():
    from ai import explainer, recovery

    assert recovery.MEDS_KEYWORDS == list(clinical.RATE_CONTROL_TERMS)
    record = patient(["Heart failure"], ["amiodarone 200 MG", "metformin"])
    assert explainer.relevant_history(record)[1] == ["amiodarone 200 MG"]  # was missing from the wording list


def test_recovery_group_follows_medication_updates():
    model = RecoveryModel(medications=["metformin"])
    before = model._group
    model.set_medications(["24 HR metoprolol succinate 50 MG"])
    assert model._group == "meds" and before != "meds"
