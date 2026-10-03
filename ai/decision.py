"""Alert decision stub.

Rules gate first (quality OK, stationary, deviation persists), then an LLM reason.
Low signal quality produces no alert. TODO: implement both stages.
"""

from typing import Optional
from ai.contracts import Alert, PatientContext, Reading


def evaluate(reading: Reading, context: PatientContext) -> Optional[Alert]:
    """Return an Alert only when the level changes.

    TODO: require quality >= context.config.min_quality, activity == "resting",
    and deviation >= context.config.deviation_trigger for context.config.persist_s.
    Honor cooldown_s. Then ask the LLM for reason, voice_text, and next_step.
    """
    return None
