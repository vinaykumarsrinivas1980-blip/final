# Raspberry Pi Cloud Voice Assistant 🤖🎙️

A modular, ultra-low-latency, zero-cost voice assistant built in Python for Raspberry Pi OS (64-bit) and Windows. Operates in both **Push-To-Talk** and **Hands-Free Wake-Word ("Hey Jarvis")** modes.

---

## 1. Tech Architecture & Zero-Cost Pipeline

- **Audio I/O**: `sounddevice` + `scipy` (Auto-detects USB mic/speakers, handles ALSA 1/2 channel stereo-to-mono downmixing, sample rate fallback).
- **Wake Word Detection**: `openWakeWord` (100% Offline, local "Hey Jarvis" wake word detection).
- **Speech-to-Text (STT)**: 
  1. Groq Whisper (`whisper-large-v3-turbo` — 100% Free & ultra-fast)
  2. Google Free Web STT (`speech_recognition` — 100% Free, no API keys or quota required)
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

---

## 2. Project Folder Structure

```
robo2/
├── .env.example            # Template for API keys & device overrides
├── requirements.txt        # Python package dependencies
├── audio_io.py             # Audio device discovery, ALSA channel fallback, recording & playback
├── wakeword.py             # openWakeWord offline local wake-word detector ("Hey Jarvis")
├── prompts.py              # Social robot personality, multi-lingual rules, and 0ms local canned replies
├── stt.py                  # Groq Whisper & Google Free Web STT handler
├── llm.py                  # Groq LLM client with memory buffer, canned responses, & multi-model fallback
├── tts.py                  # 5-Tier TTS fallback (Edge-TTS, gTTS, pyttsx3, espeak)
├── main.py                 # Interactive Voice Assistant CLI loop
└── README.md               # Setup & usage documentation
```

---

## 3. Setup Instructions for Raspberry Pi & Windows

### Step 1: System Package Installation (Raspberry Pi OS)
On Raspberry Pi (64-bit), install ALSA, PortAudio, and audio player utilities:

```bash
sudo apt update
sudo apt install -y python3-pip python3-dev portaudio19-dev libffi-dev libssl-dev ffmpeg mpg123 espeak aplay
```

### Step 2: Virtual Environment & Python Packages
Create a Python virtual environment and install dependencies:

```bash
cd robo2
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install core dependencies (including scikit-learn, onnxruntime, etc.)
pip install -r requirements.txt

# Install openwakeword without tflite dependency (compatible with Python 3.13 / ARM64)
pip install openwakeword --no-deps
```

### Step 3: Configure Environment Variables
Create your `.env` file from `.env.example`:

```bash
cp .env.example .env
nano .env
```

Set your free Groq API key (get a free key at [console.groq.com](https://console.groq.com)):

```ini
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
TTS_VOICE=en-IN-NeerjaNeural

# Audio Device Overrides (Optional - leave blank for auto-detecting USB devices)
MIC_DEVICE_INDEX=
SPEAKER_DEVICE_INDEX=
```

---

## 4. Independent Stage Diagnostics

Test each module independently to verify hardware and API configurations:

### A. Test Audio Hardware (Microphone & Speaker)
```bash
# List all enumerated microphones and speakers
python audio_io.py --list

# Record push-to-talk audio and play back test audio
python audio_io.py --test-mic
```

### B. Test Speech-to-Text (STT)
```bash
python stt.py test_recording.wav
```

### C. Test Large Language Model (LLM)
```bash
python llm.py "What is the capital of Japan?"
```

### D. Test Text-to-Speech (TTS)
```bash
python tts.py "Hello! Voice assistant test successful."
```

---

## 5. Running the Voice Assistant

### Option A: Hands-Free Mode (Wake-Word Detection)
Say **"Hey Jarvis"** to activate and speak naturally:

```bash
python main.py --wake-word
```

### Option B: Push-to-Talk Mode
Interactive CLI mode:

```bash
python main.py
```

- Press **ENTER** to start recording into your microphone.
- Press **ENTER** again to stop recording.
- Type `clear` + **ENTER** to reset short-term conversation memory.
- Type `q` or `exit` to quit.

### Voice Interrupt & Barge-In Commands
The assistant supports **real-time voice interrupt** both during speech recognition and active speaker 
playback:
- **Speech Stop Keywords**: Say `"stop"`, `"shut up"`, `"quiet"`, `"pause"`, `"cancel"`, `"nevermind"`, `"roko"`, `"ruko"`, `"band karo"`, `"chup"`, `"nillu"`, or `"saaku"`.
- **Live Playback Barge-In**: Speaking while the robot speaker is playing audio instantly cuts off speaker playback and resets the program to listening mode.
- Press **ENTER** (in Push-to-Talk mode) or **Ctrl+C** to interrupt.

---

## 6. Latency & Performance Breakdown

| Stage | Technology / Provider | Latency | Cost / Quota |
| :--- | :--- | :--- | :--- |
| **1. Mic Capture** | `sounddevice` (ALSA/WASAPI) | ~0.05s | Free / Local |
| **2. Wake-Word** | `openWakeWord` (ONNX) | ~0.08s | Free / Local |
| **3. STT** | Groq Whisper / Google Free STT | **0.3s - 0.7s** | Free (14,400 RPD) |
| **4. LLM** | Groq API (`llama-3.3-70b-versatile`) | **0.2s - 0.5s** | Free (14,400 RPD) |
| **5. TTS** | Edge-TTS (Neural) / pyttsx3 | **0.4s - 0.9s** | Free / Local |
| **TOTAL TURN** | **Complete Voice Cycle** | **~1.2s - 2.1s** | **100% Free** |
