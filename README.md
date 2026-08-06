# Raspberry Pi Cloud Voice Assistant 🤖🎙️

A modular, ultra-low-latency, zero-cost voice assistant built in Python for Raspberry Pi OS (64-bit) and Windows. Configured for **100% Headless Operation** without requiring an HDMI monitor, keyboard, or mouse. Operates in both **Push-To-Talk** and **Hands-Free Wake-Word ("Hey Jarvis" / "Max")** modes.

---

## 1. Tech Architecture & Zero-Cost Pipeline

- **Audio I/O**: `sounddevice` + `scipy` (Auto-detects USB mic/speakers, handles ALSA 1/2 channel stereo-to-mono downmixing, sample rate fallback).
- **Wake Word Detection**: `openWakeWord` (100% Offline local "Hey Jarvis" / "Max" ONNX wake word detection).
- **Speech-to-Text (STT)**: 
  1. Groq Whisper (`whisper-large-v3-turbo` — 100% Free & ultra-fast)
  2. Google Free Web STT (`speech_recognition` — 100% Free, no API keys required)
  3. OpenAI Whisper API (Optional fallback)
- **Large Language Model (LLM)**:
  - Groq API with 5-exchange short-term memory buffer and automatic multi-model fallback:
    `llama-3.3-70b-versatile` ➔ `llama-3.1-8b-instant` ➔ `gemma2-9b-it` ➔ `mixtral-8x7b-32768` (100% Free tier up to 14,400 requests/day).
- **Text-to-Speech (TTS)**: 5-tier fallback cascade ensuring speech output never fails:
  1. **Edge-TTS**: Free, ultra-realistic Microsoft Neural voices (no API key needed)
  2. **gTTS**: Free Google Translate TTS
  3. **pyttsx3**: 100% Offline local TTS engine (SAPI5 / eSpeak)
  4. **espeak CLI**: System-level offline fallback for Raspberry Pi/Linux
  5. **OpenAI TTS**: Optional cloud fallback
- **Headless Orchestrator**: `systemd` background service with network & USB hardware boot retries and graceful signal handling (`SIGTERM`/`SIGINT`).

---

## 2. Project Folder Structure

```
robo-assisant/
├── .env.example            # Template for API keys & device overrides
├── requirements.txt        # Python package dependencies
├── robot-assistant.service # Systemd background service unit file
├── audio_io.py             # Audio hardware discovery, VAD recording, ALSA playback & signal cleanup
├── wakeword.py             # openWakeWord offline local wake-word detector ("Hey Jarvis" / "Max")
├── prompts.py              # Robot personality, multi-lingual rules, and 0ms local canned replies
├── stt.py                  # Groq Whisper & Google Free Web STT handler
├── llm.py                  # Groq LLM engine with memory buffer, canned responses & model fallbacks
├── tts.py                  # 5-Tier TTS fallback (Edge-TTS, gTTS, pyttsx3, espeak)
├── test_suite.py           # Comprehensive integration & unit test suite
├── main.py                 # Main orchestrator CLI & systemd daemon loop
└── README.md               # Setup & SSH remote management documentation
```

---

## 3. How to Connect Laptop to Raspberry Pi (Remote SSH)

### Hardware Requirements
- **Raspberry Pi** (3B+/4B/5 with Raspberry Pi OS 64-bit)
- **Power Supply** (Official USB-C/Micro-USB)
- **USB Microphone** (Plugged into Pi USB port)
- **USB Speaker** (Plugged into Pi USB port)
- **Wi-Fi / Ethernet Network**
- **Laptop** (Windows, Mac, or Linux)
- *(No HDMI monitor, keyboard, or mouse required after initial Wi-Fi setup)*

---

### Step 1: Enable SSH on Raspberry Pi
If you haven't enabled SSH when writing the OS image using **Raspberry Pi Imager**:
1. Insert the microSD card into your laptop.
2. In the `boot` partition of the microSD card, create a blank file named `ssh` (no extension).
3. Insert the card into your Raspberry Pi and power it ON.

---

### Step 2: Discover Raspberry Pi IP Address on Laptop

Open Command Prompt / PowerShell (Windows) or Terminal (Mac/Linux) on your laptop:

```bash
# Ping the default Raspberry Pi hostname
ping raspberrypi.local
```

If `raspberrypi.local` resolves, note the IP address (e.g. `192.168.1.50`).

*Alternative IP discovery method:*
- Check your home Wi-Fi router's connected devices page.
- Or use `nmap` / `arp -a` from your laptop:
  ```bash
  arp -a | findstr -i "dc-a6-32 b8-27-eb"
  ```

---

### Step 3: Connect via SSH Terminal

From your laptop terminal, run:

```bash
ssh pi@raspberrypi.local
# Or using the specific IP address:
ssh pi@192.168.1.50
```

- **Default Username**: `pi` (or the username created during Pi Imager setup)
- **Password**: Type your Pi account password when prompted.

---

### Step 4: (Optional) Connect via VS Code Remote SSH

1. Open **VS Code** on your laptop.
2. Install the **Remote - SSH** extension by Microsoft.
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select **Remote-SSH: Connect to Host...**.
4. Type `pi@raspberrypi.local` or `pi@<IP_ADDRESS>`.
5. Open the folder `/home/pi/robo-assisant` to edit code live from your laptop.

---

## 4. Headless Setup & Auto-Start Configuration

### Step 1: Install System Audio Packages (Raspberry Pi OS)
After connecting via SSH, install PortAudio and ALSA utilities:

```bash
sudo apt update
sudo apt install -y python3-pip python3-dev portaudio19-dev libffi-dev libssl-dev ffmpeg mpg123 espeak aplay alsa-utils
```

---

### Step 2: Setup Python Environment & Dependencies
Clone or navigate to the repository on your Pi:

```bash
cd ~/robo-assisant
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install openwakeword without tflite (compatible with ARM64 / Python 3.13)
pip install openwakeword --no-deps
```

---

### Step 3: Configure Environment Variables
Create your `.env` configuration file:

```bash
cp .env.example .env
nano .env
```

Set your free Groq API key (get a free key at [console.groq.com](https://console.groq.com)):

```ini
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
TTS_VOICE=en-IN-NeerjaNeural
WAKE_MODEL=max
WAKE_THRESHOLD=0.40

# Audio Device Overrides (Optional - leave empty for auto-detecting USB mic/speaker)
MIC_DEVICE_INDEX=
SPEAKER_DEVICE_INDEX=
```

---

### Step 4: Install & Enable systemd Boot Service

To make the assistant start automatically when the Raspberry Pi boots:

```bash
# 1. Copy the systemd service file
sudo cp robot-assistant.service /etc/systemd/system/robot-assistant.service

# 2. Replace %u placeholder with your Pi username (e.g. pi)
sudo sed -i "s/%u/$USER/g" /etc/systemd/system/robot-assistant.service

# 3. Reload systemd daemon
sudo systemctl daemon-reload

# 4. Enable service to start automatically on boot
sudo systemctl enable robot-assistant.service

# 5. Start the service immediately
sudo systemctl start robot-assistant.service
```

---

## 5. Remote SSH Service Control Commands

Manage the background assistant service from your laptop via SSH:

```bash
# Check service status and health
sudo systemctl status robot-assistant

# Start the assistant service
sudo systemctl start robot-assistant

# Stop the assistant service
sudo systemctl stop robot-assistant

# Restart the assistant service
sudo systemctl restart robot-assistant
```

---

## 6. Live Logging & Monitoring

View real-time application logs over SSH:

```bash
# Stream live logs in real-time (tail output)
sudo journalctl -u robot-assistant -f

# View recent 100 log entries
sudo journalctl -u robot-assistant -n 100

# View logs from current boot cycle only
sudo journalctl -u robot-assistant -b

# View error messages only
sudo journalctl -u robot-assistant -p err
```

---

## 7. Independent Stage Diagnostics

Test each component individually to verify hardware and API setup:

```bash
# 1. Run full automated integration test suite
python test_suite.py

# 2. List all connected microphones and speakers
python audio_io.py --list

# 3. Test USB Mic recording and speaker playback
python audio_io.py --test-mic

# 4. Test Speech-to-Text (STT)
python stt.py test_recording.wav

# 5. Test Large Language Model (LLM)
python llm.py "What is the capital of Japan?"

# 6. Test Text-to-Speech (TTS)
python tts.py "Hello! Headless voice assistant test successful."
```

---

## 8. Hardware & Network Troubleshooting Guide

### A. Microphone Troubleshooting
```bash
# Check USB device physical connection
lsusb

# List hardware recording devices
arecord -l

# Adjust microphone capture volume levels
alsamixer
```

### B. Speaker Troubleshooting
```bash
# List hardware playback devices
aplay -l

# Test sound output via ALSA speaker test
speaker-test -t wav -c 2

# Force ALSA master volume to 100% and unmute
amixer sset 'Master' 100% unmute
amixer sset 'PCM' 100% unmute
```

### C. Network Connectivity Troubleshooting
```bash
# Verify Wi-Fi state
nmcli device status

# Test external DNS / internet reachability
ping -c 4 8.8.8.8
ping -c 4 console.groq.com
```

---

## 9. Disabling Auto-Boot & Reverting Changes

To stop auto-start on boot and revert to manual CLI execution:

```bash
# Disable systemd auto-start
sudo systemctl disable robot-assistant.service
sudo systemctl stop robot-assistant.service

# Remove systemd unit file
sudo rm /etc/systemd/system/robot-assistant.service
sudo systemctl daemon-reload

# Run manually anytime from terminal:
python main.py
```

---

## 10. Latency & Performance Breakdown

| Stage | Technology / Provider | Latency | Cost / Quota |
| :--- | :--- | :--- | :--- |
| **1. Mic Capture** | `sounddevice` (ALSA/WASAPI) | ~0.05s | Free / Local |
| **2. Wake-Word** | `openWakeWord` (ONNX) | ~0.08s | Free / Local |
| **3. STT** | Groq Whisper / Google Free STT | **0.3s - 0.7s** | Free (14,400 RPD) |
| **4. LLM** | Groq API (`llama-3.3-70b-versatile`) | **0.2s - 0.5s** | Free (14,400 RPD) |
| **5. TTS** | Edge-TTS (Neural) / pyttsx3 | **0.4s - 0.9s** | Free / Local |
| **TOTAL TURN** | **Complete Voice Cycle** | **~1.2s - 2.1s** | **100% Free** |
