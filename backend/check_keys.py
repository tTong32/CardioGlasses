"""S-03: test each external service with one call. Run `python -m backend.check_keys`."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def check_finchnode() -> str:
    from backend.finchnode import fetch_patient_context
    from backend.main import local_patient

    context = fetch_patient_context(local_patient().config)
    return f"{context.patient_id}, age {context.age}, {len(context.conditions)} conditions, {context.risk_tier} risk"


def check_elevenlabs() -> str:
    from backend.voice import synthesize

    audio = synthesize("Test.")
    return f"{len(audio)} bytes of audio"


def check_anthropic() -> str:
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    # Retrieving a model validates the key without spending tokens.
    model = anthropic.Anthropic().models.retrieve("claude-opus-5-5")
    return f"key valid, {model.id} available"


def main() -> int:
    failures = 0
    for name, check in [
        ("FinchNode", check_finchnode),
        ("ElevenLabs", check_elevenlabs),
        ("Anthropic", check_anthropic),
    ]:
        try:
            print(f"OK    {name}: {check()}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
