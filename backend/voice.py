"""ElevenLabs voice for alerts, with cached and pre-generated fallback audio.

Audio files land in data/audio/ and are served at /audio/<name>. The dashboard plays
them; if no file can be produced it falls back to the browser's speech synthesis.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

AUDIO_DIR = ROOT / "data" / "audio"
API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_MODEL_ID = "eleven_flash_v2_5"

# Spoken when ElevenLabs or Wi-Fi is down. Generate with `python -m backend.voice --fallbacks`.
FALLBACK_TEXT = {
    "normal": "Your heart rate is back within your usual range. I'll keep monitoring.",
    "monitor": "I'm keeping a closer eye on your heart rate. No action needed yet.",
    "notify": "Your heart rate has stayed high while you're resting. Please sit down and check your phone.",
    "escalate": "Your heart rate has stayed high for a while. Please sit down, check your phone, and consider contacting your care team.",
}

log = logging.getLogger(__name__)


def _settings() -> tuple[str | None, str, str]:
    return (
        os.environ.get("ELEVENLABS_API_KEY") or None,
        os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID,
        os.environ.get("ELEVENLABS_MODEL_ID") or DEFAULT_MODEL_ID,
    )


def is_configured() -> bool:
    return _settings()[0] is not None


def synthesize(text: str, timeout: float = 8) -> bytes:
    """Return MP3 bytes for `text`. Raises if the key is missing or the call fails."""
    api_key, voice_id, model_id = _settings()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    response = requests.post(
        API_URL.format(voice_id=voice_id),
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": api_key, "accept": "audio/mpeg"},
        json={"text": text, "model_id": model_id},
        timeout=timeout,
    )
    response.raise_for_status()
    if not response.content:
        raise RuntimeError("ElevenLabs returned no audio")
    return response.content


def _cache_name(text: str) -> str:
    _key, voice_id, model_id = _settings()
    digest = hashlib.sha1(f"{voice_id}|{model_id}|{text}".encode("utf-8")).hexdigest()
    return f"tts_{digest[:16]}.mp3"


def fallback_name(level: str) -> str:
    return f"fallback_{level}.mp3"


def audio_for(text: str | None, level: str) -> tuple[str | None, str]:
    """Return `(file name in AUDIO_DIR, source)` for an alert's voice line.

    source is "cache", "elevenlabs", "fallback", or "none" (dashboard speaks it itself).
    """
    if not text:
        return None, "none"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    name = _cache_name(text)
    if (AUDIO_DIR / name).is_file():
        return name, "cache"
    try:
        (AUDIO_DIR / name).write_bytes(synthesize(text))
        return name, "elevenlabs"
    except Exception as exc:  # network, auth, quota: never block the alert
        log.warning("ElevenLabs failed, using fallback audio: %s", exc)
    if (AUDIO_DIR / fallback_name(level)).is_file():
        return fallback_name(level), "fallback"
    return None, "none"


def write_fallbacks() -> list[Path]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for level, text in FALLBACK_TEXT.items():
        path = AUDIO_DIR / fallback_name(level)
        path.write_bytes(synthesize(text))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="ElevenLabs voice tools.")
    parser.add_argument("--test", metavar="TEXT", help="Synthesize TEXT to data/audio/test.mp3")
    parser.add_argument("--fallbacks", action="store_true", help="Write one fallback clip per level")
    args = parser.parse_args()
    if not (args.test or args.fallbacks):
        parser.error("provide --test TEXT or --fallbacks")
    if args.test:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path = AUDIO_DIR / "test.mp3"
        path.write_bytes(synthesize(args.test))
        print(f"wrote {path}")
    if args.fallbacks:
        for path in write_fallbacks():
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
