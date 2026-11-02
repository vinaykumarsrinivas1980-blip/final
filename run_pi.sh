#!/usr/bin/env bash
# =====================================================================
# One-Command Launcher for Raspberry Pi Robot Assistant
#
# Usage:
#   ./run_pi.sh                 # Default mode (wake-word if configured in .env)
#   ./run_pi.sh --wake-word     # Hands-free wake word mode ("Spark")
#   ./run_pi.sh --push-to-talk  # Push-to-talk terminal mode (ENTER key)
# =====================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️ Python virtual environment not found. Running setup_rpi.sh..."
    bash setup_rpi.sh
fi

# Ensure .env exists
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "📋 Copying .env.example to .env..."
    cp .env.example .env
fi

# Boost ALSA hardware volume before running
if command -v amixer >/dev/null 2>&1; then
    amixer sset 'Master' 100% unmute >/dev/null 2>&1 || true
    amixer sset 'PCM' 100% unmute >/dev/null 2>&1 || true
    amixer sset 'Headphone' 100% unmute >/dev/null 2>&1 || true
    amixer sset 'Speaker' 100% unmute >/dev/null 2>&1 || true
fi

# Activate venv and run assistant
source venv/bin/activate
exec python main.py "$@"
