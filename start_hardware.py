#!/usr/bin/env python3
"""Quick start script for connecting hardware to the AI pipeline + backend (Windows-compatible)."""

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


def check_backend():
    """Check if backend is running."""
    print("1. Checking backend...")
    try:
        requests.get("http://127.0.0.1:8000/health", timeout=2)
        print("✅ Backend is running at http://127.0.0.1:8000")
        return True
    except requests.RequestException:
        print("❌ Backend is NOT running")
        print("\nStart the backend in another terminal:")
        print("  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000")
        print()
        input("Press Enter once backend is started, or Ctrl+C to exit...")
        return check_backend()


def list_ports():
    """List available serial ports."""
    print("\n2. Available serial ports:")
    subprocess.run([sys.executable, "-m", "tools.serial_check", "--list"])
    print()


def get_current_port():
    """Get SERIAL_PORT from .env if it exists."""
    env_file = Path(".env")
    if not env_file.exists():
        return None

    content = env_file.read_text()
    match = re.search(r'^SERIAL_PORT=(.+)$', content, re.MULTILINE)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return None


def set_port(port):
    """Update SERIAL_PORT in .env."""
    env_file = Path(".env")

    if not env_file.exists():
        env_file.write_text(f"SERIAL_PORT={port}\n")
        print(f"✅ Created .env with SERIAL_PORT={port}")
        return

    content = env_file.read_text()
    if re.search(r'^SERIAL_PORT=', content, re.MULTILINE):
        # Update existing
        content = re.sub(r'^SERIAL_PORT=.*$', f'SERIAL_PORT={port}', content, flags=re.MULTILINE)
        env_file.write_text(content)
        print(f"✅ Updated .env with SERIAL_PORT={port}")
    else:
        # Append
        env_file.write_text(content + f"\nSERIAL_PORT={port}\n")
        print(f"✅ Added SERIAL_PORT={port} to .env")


def configure_port():
    """Configure serial port."""
    current = get_current_port()

    if current:
        print(f"📍 Current .env has: SERIAL_PORT={current}")
        response = input("Use this port? (y/n): ").strip().lower()
        if response == 'y':
            return current

    new_port = input("Enter serial port (e.g., /dev/cu.usbmodem14201 or COM5): ").strip()
    set_port(new_port)
    return new_port


def test_serial(port):
    """Quick serial test."""
    print("\n3. Testing serial stream (5 seconds)...")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tools.serial_check", "--port", port],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(5)
        proc.terminate()
        proc.wait()
    except Exception:
        pass
    print()


def choose_llm_mode():
    """Ask user about LLM mode."""
    print("4. Wording mode:")
    print("   a) Template mode (fast, no API key needed)")
    print("   b) LLM mode (Gemini wording, needs GEMINI_API_KEY in .env)")
    choice = input("Choose (a/b) [default: a]: ").strip().lower()
    return choice != 'b'


def get_local_ip():
    """Get local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def show_dashboard_urls():
    """Show dashboard URLs."""
    print("\n5. Dashboard URLs:")
    print("   Laptop: http://localhost:8000")
    ip = get_local_ip()
    if ip:
        print(f"   Phone:  http://{ip}:8000")
    print()


def start_stream(use_templates):
    """Start the live stream."""
    print("=" * 40)
    print("🚀 Starting live stream...")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    print()

    args = [sys.executable, "-m", "tools.live_stream"]
    if use_templates:
        args.append("--no-llm")

    try:
        subprocess.run(args)
    except KeyboardInterrupt:
        print("\n\n✅ Stopped")


def main():
    """Main setup flow."""
    print("🩺 CardioGlasses Hardware Setup")
    print("=" * 40)
    print()

    check_backend()
    list_ports()
    port = configure_port()
    test_serial(port)
    use_templates = choose_llm_mode()
    show_dashboard_urls()
    start_stream(use_templates)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled")
        sys.exit(0)
