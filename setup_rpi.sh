#!/usr/bin/env bash
# =====================================================================
# Raspberry Pi Automated Setup & Diagnostic Suite for Robot Assistant
# Run on Raspberry Pi: bash setup_rpi.sh
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================================="
echo "🚀 Raspberry Pi Automated Setup & Diagnostics for Robot Assistant"
echo "====================================================================="

# 0. Check Hardware Architecture
ARCH=$(uname -m)
echo "🔍 Detected architecture: $ARCH"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    echo "⚠️  WARNING: Detected $ARCH architecture."
    echo "   Raspberry Pi OS 64-bit (aarch64) is strongly recommended for"
    echo "   official ONNXRuntime and openWakeWord wheel support."
fi

# 1. Update APT & install required system audio, ALSA, codecs, build tools
echo ""
echo "📦 [1/6] Installing system audio, ALSA, and build libraries..."
sudo apt update
sudo apt install -y python3-pip python3-dev python3-venv portaudio19-dev \
    libasound2-dev libasound2-plugins libffi-dev libssl-dev \
    ffmpeg mpg123 espeak-ng espeak alsa-utils sox

# 2. Setup user audio and hardware permissions
echo ""
echo "🎙️ [2/6] Configuring user permissions for audio..."
sudo usermod -aG audio "$USER" || true
if getent group gpio >/dev/null; then
    sudo usermod -aG gpio "$USER" || true
fi

# 3. Smart ALSA Audio Card Detection and Configuration
echo ""
echo "🔊 [3/6] Configuring ALSA audio routing..."

# Check if a USB audio card exists in /proc/asound/cards
USB_CARD_NUM=$(cat /proc/asound/cards 2>/dev/null | grep -i "usb" | head -n 1 | awk '{print $1}' || true)
if [ -n "$USB_CARD_NUM" ]; then
    echo "  • Found USB Audio Hardware on Card $USB_CARD_NUM. Setting up ~/.asoundrc..."
    cat > "$HOME/.asoundrc" <<EOF
pcm.!default {
    type asym
    playback.pcm {
        type plug
        slave.pcm "hw:${USB_CARD_NUM},0"
    }
    capture.pcm {
        type plug
        slave.pcm "hw:${USB_CARD_NUM},0"
    }
}
ctl.!default {
    type hw
    card ${USB_CARD_NUM}
}
EOF
else
    echo "  • No dedicated USB soundcard detected yet. Leaving default ALSA routing."
fi

# Boost ALSA hardware sliders to 100%
amixer sset 'Master' 100% unmute >/dev/null 2>&1 || true
amixer sset 'PCM' 100% unmute >/dev/null 2>&1 || true
amixer sset 'Headphone' 100% unmute >/dev/null 2>&1 || true
amixer sset 'Speaker' 100% unmute >/dev/null 2>&1 || true

# 4. Create and configure Python Virtual Environment
echo ""
echo "🐍 [4/6] Setting up Python virtual environment (venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "📥 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "🧠 Installing openWakeWord (optimized for ARM64/Linux without tflite)..."
pip install openwakeword --no-deps

echo "⬇️ Pre-caching openWakeWord neural models..."
python3 -c "import openwakeword; openwakeword.utils.download_models()" || true

# Ensure .env exists from template
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "📋 Creating .env from .env.example..."
    cp .env.example .env
fi

# 5. Make launcher scripts executable
chmod +x run_pi.sh setup_rpi.sh install_service.sh 2>/dev/null || true

# 6. Run Full Diagnostic Suite
echo ""
echo "---------------------------------------------------------------------"
echo "🧪 [5/6] Running System & Audio Diagnostic Suite..."
echo "---------------------------------------------------------------------"
python test_suite.py || true

echo ""
echo "====================================================================="
echo "✅ Raspberry Pi Setup Complete!"
echo "====================================================================="
echo "You can now run your assistant in multiple ways:"
echo ""
echo "1. Run manually via One-Command Launcher:"
echo "   ./run_pi.sh --wake-word"
echo ""
echo "2. Install as an auto-starting background service on boot:"
echo "   bash install_service.sh"
echo "====================================================================="
