#!/usr/bin/env bash
# =====================================================================
# Raspberry Pi Setup & Diagnostic Script for up-robo Assistant
# Run on Raspberry Pi: bash setup_rpi.sh
# =====================================================================

set -e

echo "🚀 Starting Raspberry Pi Automated Setup & Diagnostics for up-robo..."
echo "---------------------------------------------------------------------"

# 1. Update APT and install essential system libraries
echo "📦 [1/5] Installing system audio, ALSA, and build libraries..."
sudo apt update
sudo apt install -y python3-pip python3-dev python3-venv portaudio19-dev libasound2-dev libffi-dev libssl-dev ffmpeg mpg123 espeak aplay alsa-utils

# 2. Add user to audio group for hardware permission
echo "🎙️ [2/5] Setting up audio permissions..."
sudo usermod -aG audio $USER || true

# 3. Unmute and boost ALSA volume
echo "🔊 [3/5] Boosting ALSA speaker volume to 100%..."
amixer sset 'Master' 100% unmute >/dev/null 2>&1 || true
amixer sset 'PCM' 100% unmute >/dev/null 2>&1 || true
amixer sset 'Headphone' 100% unmute >/dev/null 2>&1 || true
amixer sset 'Speaker' 100% unmute >/dev/null 2>&1 || true

# 4. Create and configure virtual environment
echo "🐍 [4/5] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip

echo "📥 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "🧠 Installing openwakeword without tflite-runtime..."
pip install openwakeword --no-deps

echo "⬇️ Pre-downloading openWakeWord ONNX models..."
python3 -c "import openwakeword; openwakeword.utils.download_models()"

# 5. Run test suite diagnostics
echo "---------------------------------------------------------------------"
echo "🧪 [5/5] Running diagnostics..."
python test_suite.py || true

echo "---------------------------------------------------------------------"
echo "✅ Raspberry Pi setup complete!"
echo "To start the assistant in hands-free wake word mode:"
echo "   source venv/bin/activate && python main.py --wake-word"
echo "---------------------------------------------------------------------"
