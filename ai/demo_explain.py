"""Demo explanation mode - shows clinical reasoning for judges."""

from ai.contracts import PatientContext, Alert
from ai.clinical import (
    effects,
    irregular_rhythm_expected,
    rate_control_medications,
    baseline_min_std,
    TIER_SENSITIVITY
)


def print_patient_emr(context: PatientContext) -> None:
    """Display patient's electronic medical record."""
    print("=" * 80)
    print("📋 PATIENT ELECTRONIC MEDICAL RECORD (EMR)")
    print("=" * 80)
    print(f"Patient ID: {context.patient_id}")
    print(f"Age: {context.age} years")
    print(f"\nMedical Conditions:")
    for condition in context.conditions:
        print(f"  • {condition}")
    print(f"\nCurrent Medications:")
    for med in context.medications:
        print(f"  • {med}")
    if context.clinic_resting_hr:
        print(f"\nMost Recent Clinic Visit:")
        print(f"  • Resting Heart Rate: {context.clinic_resting_hr:.0f} bpm")
    print()


def print_clinical_reasoning(context: PatientContext) -> None:
    """Explain how EMR determines monitoring thresholds."""
    print("=" * 80)
    print("🧠 CLINICAL DECISION-MAKING PROCESS")
    print("=" * 80)

    # Risk tier analysis
    tier = context.risk_tier
    sensitivity = TIER_SENSITIVITY[tier]
    print(f"\n1. RISK STRATIFICATION")
    print(f"   Risk Tier: {tier.upper()}")
    print(f"   Rationale:")

    # Explain why high risk
    if tier == "high":
        if "hypertension" in [c.lower() for c in context.conditions]:
            print(f"      • Hypertension present → increased cardiovascular risk")
        if "diabetes" in str(context.conditions).lower():
            print(f"      • Diabetes present → comorbidity increases risk")
        if context.age > 65:
            print(f"      • Age {context.age} → elderly population at higher risk")

    print(f"\n   Alert Thresholds (based on risk tier):")
    print(f"      • Deviation Trigger: {sensitivity['deviation_trigger']}σ above baseline")
    print(f"      • Persistence Required: {sensitivity['persist_s']:.0f} seconds")
    print(f"      Translation: Alert if HR stays {sensitivity['deviation_trigger']}σ above normal")
    print(f"                   for {sensitivity['persist_s']:.0f}+ seconds while resting")

    # Irregular rhythm handling
    print(f"\n2. SIGNAL PROCESSING ADJUSTMENTS")
    irregular = irregular_rhythm_expected(context)
    if irregular:
        rhythm_conditions = [c for c in context.conditions
                            if any(term in c.lower() for term in
                                  ["atrial fibrillation", "atrial flutter", "arrhythmia"])]
        print(f"   Irregular Rhythm Detected: YES")
        print(f"   Condition: {rhythm_conditions[0]}")
        print(f"   Adjustment:")
        print(f"      • Baseline variability floor: {baseline_min_std(context):.0f} bpm")
        print(f"        (vs {4.0} bpm for regular rhythm)")
        print(f"      • Rationale: AF causes natural 4-6 bpm variation")
        print(f"      • Beat detection uses irregular-rhythm algorithm")
    else:
        print(f"   Irregular Rhythm: NO")
        print(f"   Using standard sinus rhythm processing")

    # Medication effects
    print(f"\n3. MEDICATION CONSIDERATIONS")
    rate_meds = rate_control_medications(context)
    if rate_meds:
        print(f"   Rate-Control Medication: {rate_meds[0]}")
        print(f"   Effects on Monitoring:")
        print(f"      • Lowers resting heart rate baseline")
        print(f"      • Blunts heart rate response to activity")
        print(f"      • Recovery compared to 'meds' cohort (not general population)")
        print(f"      • Exercise zone: Resting + 20-30 bpm (not age formula)")
    else:
        print(f"   No rate-control medications")
        print(f"   Using standard age-based formulas")

    # Baseline strategy
    print(f"\n4. BASELINE CALIBRATION STRATEGY")
    if context.clinic_resting_hr:
        print(f"   Provisional Baseline: {context.clinic_resting_hr:.0f} bpm (from clinic)")
        print(f"   Variability floor: {8.0} bpm (wider margin for clinic reading)")
        print(f"   Will be replaced after 60s of measured rest")
    else:
        print(f"   No clinic baseline available")
        print(f"   Requires 60s continuous rest to establish baseline")

    print(f"\n5. ALERT DECISION TREE")
    min_rise = round(sensitivity['deviation_trigger'] * baseline_min_std(context))
    print(f"   For this patient to trigger an alert:")
    print(f"      ✓ Must be resting (not moving)")
    print(f"      ✓ HR must be ≥{min_rise} bpm above baseline")
    print(f"      ✓ Must persist for ≥{sensitivity['persist_s']:.0f} seconds")
    print(f"      ✓ Signal quality must be ≥{context.config.min_quality:.0%}")
    print(f"      ✓ Must be ≥{context.config.cooldown_s}s since last alert")

    print("\n" + "=" * 80)
    print()


def print_alert_reasoning(alert: Alert, context: PatientContext) -> None:
    """Explain why a specific alert was triggered."""
    reading = alert.reading
    if not reading:
        return

    print("\n" + "🚨" * 40)
    print(f"ALERT TRIGGERED: {alert.level.upper()}")
    print("🚨" * 40)

    print(f"\nClinical Justification:")
    print(f"  Current HR: {reading.hr:.0f} bpm")
    print(f"  Baseline HR: {reading.baseline_hr:.0f} bpm")
    print(f"  Deviation: {reading.deviation:.1f}σ ({reading.hr - reading.baseline_hr:.0f} bpm above normal)")
    print(f"  Persistence: {reading.persist_s:.0f}s (threshold: {context.config.persist_s}s)")
    print(f"  Activity: {reading.activity}")
    print(f"  Signal Quality: {reading.quality:.0%}")

    print(f"\nWhy This Matters:")
    if alert.level == "monitor":
        print(f"  • MONITOR: Early warning - elevated but within grace period")
    elif alert.level == "notify":
        print(f"  • NOTIFY: Sustained elevation beyond clinical threshold")
        print(f"  • Patient check-in initiated: 'Are you OK?'")
    elif alert.level == "escalate":
        print(f"  • ESCALATE: Severe sustained elevation")
        print(f"  • Deviation ≥{context.config.deviation_trigger * 2}σ")
        print(f"  • Duration ≥{context.config.persist_s * 3}s")

    print(f"\nRecommended Action:")
    print(f"  {alert.next_step}")

    print("=" * 80 + "\n")


def print_dashboard_context(context: PatientContext) -> None:
    """Print the human-readable clinical effects shown on dashboard."""
    print("\n📱 DASHBOARD DISPLAY (What Patient/Caregiver Sees):")
    print("-" * 80)
    for effect in effects(context):
        print(f"  • {effect}")
    print("-" * 80 + "\n")
