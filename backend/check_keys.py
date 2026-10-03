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


def check_gemini() -> str:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    # Listing models validates the key without using any free-tier quota.
    names = [model.name for model in genai.Client(api_key=api_key).models.list()]
    flash = [name for name in names if "flash" in name]
    return f"key valid, {len(names)} models ({len(flash)} flash)"


def main() -> int:
    failures = 0
    for name, check in [
        ("FinchNode", check_finchnode),
        ("ElevenLabs", check_elevenlabs),
        ("Gemini", check_gemini),
    ]:
        try:
            print(f"OK    {name}: {check()}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
