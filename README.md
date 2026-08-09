# Raspberry Pi Cloud Voice Assistant 🤖🎙️

![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Raspberry_Pi_OS_%7C_Windows-red?style=for-the-badge&logo=raspberrypi&logoColor=white)
![LLM Engine](https://img.shields.io/badge/LLM-Groq_API_(Llama_3.3_70B)-f34f29?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A modular, ultra-low-latency, zero-cost voice assistant built in Python for **Raspberry Pi OS (64-bit)** and **Windows**. Configured for **Remote Desktop (VNC) Operation** without requiring a physical monitor, keyboard, or mouse. Operates in both **Push-To-Talk** and **Hands-Free Wake-Word ("Spark" / "Hey Jarvis" / "Max")** modes.

Developed by the student team at **S J C Institute of Technology (SJCIT)**, Department of Artificial Intelligence & Machine Learning.

---

## 1. Tech Architecture & Zero-Cost Pipeline ⚡

- **Audio I/O**: `sounddevice` + `scipy` (Auto-detects USB mic/speakers, handles ALSA 1/2 channel stereo-to-mono downmixing, sample rate fallback).
- **Wake Word Detection**: `openWakeWord` (100% Offline local "Spark" (`spark.onnx`), "Hey Jarvis", "Max" ONNX wake word detection).
- **Speech-to-Text (STT)**: 
  1. **Groq Whisper** (`whisper-large-v3-turbo` — 100% Free & ultra-fast)
  2. **Google Free Web STT** (`speech_recognition` — 100% Free, no API keys required)
  3. **OpenAI Whisper API** (Optional fallback)
- **Large Language Model (LLM)**:
  - Groq API with 5-exchange short-term memory buffer and automatic multi-model fallback:
    `llama-3.3-70b-versatile` ➔ `llama-3.1-8b-instant` ➔ `gemma2-9b-it` ➔ `mixtral-8x7b-32768` (100% Free tier up to 14,400 requests/day).
- **Text-to-Speech (TTS)**: 5-tier fallback cascade ensuring speech output never fails:
  1. **Edge-TTS**: Free, ultra-realistic Microsoft Neural voices (no API key needed)
  2. **gTTS**: Free Google Translate TTS
  3. **pyttsx3**: 100% Offline local TTS engine (SAPI5 / eSpeak)
  4. **espeak CLI**: System-level offline fallback for Raspberry Pi/Linux
  5. **OpenAI TTS**: Optional cloud fallback
- **Background Service Orchestrator**: `systemd` background service with network & USB hardware boot retries and graceful signal handling (`SIGTERM`/`SIGINT`).

---

## 2. Project Folder Structure 📁

```
up-robo/
├── .env.example            # Template for API keys & device overrides
├── requirements.txt        # Python package dependencies
├── robot-assistant.service # Systemd background service unit file
├── spark.onnx              # Custom trained local openWakeWord neural model file
├── audio_io.py             # Audio hardware discovery, VAD recording, ALSA playback & signal cleanup
├── wakeword.py             # openWakeWord offline local wake-word detector ("Spark" / "Hey Jarvis" / "Max")
├── prompts.py              # Robot personality, multi-lingual rules, and 0ms local canned replies
├── stt.py                  # Groq Whisper & Google Free Web STT handler
├── llm.py                  # Groq LLM engine with memory buffer, canned responses & model fallbacks
├── tts.py                  # 5-Tier TTS fallback (Edge-TTS, gTTS, pyttsx3, espeak)
├── test_suite.py           # Comprehensive integration & unit test suite
├── main.py                 # Main orchestrator CLI & systemd daemon loop
└── README.md               # Setup & VNC remote management documentation
```

### 📄 Source Code Modules:
- [main.py](file:///C:/Users/lrtha/up-robo/main.py) — System Orchestrator CLI & background daemon
- [audio_io.py](file:///C:/Users/lrtha/up-robo/audio_io.py) — Audio device enumeration, dynamic VAD recording & playback
- [wakeword.py](file:///C:/Users/lrtha/up-robo/wakeword.py) — Local openWakeWord ONNX detector (`spark.onnx`)
- [llm.py](file:///C:/Users/lrtha/up-robo/llm.py) — Groq LLM engine & conversation memory history
- [stt.py](file:///C:/Users/lrtha/up-robo/stt.py) — 3-Tier Speech-to-Text engine (Groq Whisper ➔ Google ➔ OpenAI)
- [tts.py](file:///C:/Users/lrtha/up-robo/tts.py) — 5-Tier Text-to-Speech cascade (Edge-TTS ➔ gTTS ➔ pyttsx3 ➔ espeak ➔ OpenAI)
- [prompts.py](file:///C:/Users/lrtha/up-robo/prompts.py) — Personality, instant 0ms canned replies & stop commands
- [test_suite.py](file:///C:/Users/lrtha/up-robo/test_suite.py) — Integration test suite runner

---

## 3. How to Connect Laptop to Raspberry Pi (Remote VNC) 💻📱

### Hardware Requirements
- **Raspberry Pi** (3B+/4B/5 with Raspberry Pi OS 64-bit)
- **Power Supply** (Official USB-C/Micro-USB)
- **USB Microphone** (Plugged into Pi USB port)
- **USB Speaker** (Plugged into Pi USB port)
- **Wi-Fi / Ethernet Network** (Raspberry Pi & Laptop connected to the same network)
- **Laptop / PC** (Windows, Mac, or Linux with a VNC Viewer client installed)

---

### Step 1: Enable VNC on Raspberry Pi
Enable the VNC server on your Raspberry Pi:
- **Via Raspberry Pi Desktop**: Go to **Menu ➔ Preferences ➔ Raspberry Pi Configuration ➔ Interfaces** tab and set **VNC** to **Enabled**.
- **Via Raspberry Pi Terminal**: Run `sudo raspi-config`, navigate to **Interface Options ➔ VNC**, select **Yes** to enable, and exit.

---

### Step 2: Discover Raspberry Pi IP Address on Laptop

Open Command Prompt / PowerShell (Windows) or Terminal (Mac/Linux) on your laptop:

```bash
# Ping the default Raspberry Pi hostname
ping raspberrypi.local
```

If `raspberrypi.local` resolves, note the IP address (e.g., `192.168.1.50`).

> [!TIP]
> *Alternative IP discovery method:*
> - Check your home Wi-Fi router's connected devices page.
> - Or run from your laptop:
>   ```bash
>   arp -a | findstr -i "dc-a6-32 b8-27-eb"
>   ```

---

### Step 3: Install VNC Viewer & Connect

1. Download and install **RealVNC Viewer** (or TigerVNC / TightVNC) on your laptop.
2. Open VNC Viewer.
3. In the top connection/address bar, type `raspberrypi.local` or the Pi's IP address (e.g., `192.168.1.50`).
4. Press **Enter** to initiate the connection.
5. When prompted, enter your Raspberry Pi **Username** (default `pi`) and **Password**.
6. The full Raspberry Pi desktop interface will open on your laptop screen.

---

### Step 4: Access Terminal & Manage Project

Once connected via VNC desktop:
1. Click the **Terminal icon** on the Raspberry Pi taskbar (or press `Ctrl+Alt+T`).
2. Navigate to the project directory:
   ```bash
   cd ~/robo-assisant
   ```
3. Run and manage the robot assistant directly from the graphical terminal window.

---

## 4. Environment Setup & Auto-Start Configuration ⚙️

### Step 1: Install System Audio Packages (Raspberry Pi OS)
In the Raspberry Pi Terminal (via VNC or local terminal), install PortAudio and ALSA utilities:

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
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
TTS_VOICE=en-IN-NeerjaNeural
WAKE_MODEL=spark
WAKE_THRESHOLD=0.40

# Audio Device Overrides (Optional - leave empty for auto-detecting USB mic/speaker)
MIC_DEVICE_INDEX=
SPEAKER_DEVICE_INDEX=
```

---

### 🚀 One-Command Launch (Activate Environment & Run)

You can activate the virtual environment and start the robot assistant in **a single command**:

#### On Linux / Raspberry Pi:
```bash
# Hands-Free Wake-Word Mode ("Spark" / "Max" / "Hey Jarvis")
source venv/bin/activate && python main.py --wake-word

# Push-To-Talk Mode (Press ENTER to talk)
source venv/bin/activate && python main.py
```

#### On Windows (PowerShell):
```powershell
# Hands-Free Wake-Word Mode
.\venv\Scripts\Activate.ps1; python main.py --wake-word

# Push-To-Talk Mode
.\venv\Scripts\Activate.ps1; python main.py
```

#### On Windows (CMD):
```cmd
venv\Scripts\activate.bat && python main.py --wake-word
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

## 5. Service Control Commands 🛠️

Manage the background assistant service from the Raspberry Pi terminal:

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

## 6. Live Logging & Monitoring 📊

View real-time application logs in the terminal:

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

## 7. Independent Stage Diagnostics 🧪

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
python tts.py "Hello! Voice assistant test successful."

# 7. Test openWakeWord wake word detector
python wakeword.py
```

---

## 8. Hardware & Network Troubleshooting Guide 🛠️

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

## 9. Disabling Auto-Boot & Reverting Changes 🔄

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

## 10. Latency & Performance Breakdown ⏱️

| Stage | Technology / Provider | Latency | Cost / Quota |
| :--- | :--- | :--- | :--- |
| **1. Mic Capture** | `sounddevice` (ALSA/WASAPI) | ~0.05s | Free / Local |
| **2. Wake-Word** | `openWakeWord` (ONNX Neural) | ~0.08s | Free / Local |
| **3. STT** | Groq Whisper / Google Free STT | **0.3s - 0.7s** | Free (14,400 RPD) |
| **4. LLM** | Groq API (`llama-3.3-70b-versatile`) | **0.2s - 0.5s** | Free (14,400 RPD) |
| **5. TTS** | Edge-TTS (Neural) / pyttsx3 | **0.4s - 0.9s** | Free / Local |
| **TOTAL TURN** | **Complete Voice Cycle** | **~1.2s - 2.1s** | **100% Free** |

---

## 🎓 Team & Credits

Developed at **S J C Institute of Technology (SJCIT)**, Chikkaballapur, Karnataka.
- **Department**: Computer Science & Engineering — Artificial Intelligence & Machine Learning
- **Principal**: Dr. G T Raju
- **Head of Department**: Dr. Vikas Reddy S
- **Developer Team**:
  - **Vinay Kumar**
  - **Tharun L R**
  - **Shravani**