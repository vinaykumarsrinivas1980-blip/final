"""
test_suite.py - Complete Hardware, Environment & API Diagnostics for Raspberry Pi and Windows
Run anytime on Raspberry Pi or PC: python test_suite.py
"""

import os
import sys
import time
import platform
import socket
from dotenv import load_dotenv

# Ensure UTF-8 output
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def print_section(title):
    print("\n" + "=" * 65)
    print(f" 🧪 {title}")
    print("=" * 65)

def check_pass(item_name, details=""):
    det = f" ({details})" if details else ""
    print(f"  ✅ [PASS] {item_name}{det}")

def check_warn(item_name, details=""):
    det = f" ({details})" if details else ""
    print(f"  ⚠️ [WARN] {item_name}{det}")

def check_fail(item_name, details=""):
    det = f" ({details})" if details else ""
    print(f"  ❌ [FAIL] {item_name}{det}")

def run_diagnostics():
    print("🚀 Running Raspberry Pi Voice Assistant Diagnostic Suite...\n")
    all_passed = True

    # 1. System & Architecture
    print_section("1. System Architecture & Environment")
    py_ver = sys.version.split()[0]
    os_name = platform.system()
    arch = platform.machine()
    print(f"  • Operating System : {os_name} {platform.release()}")
    print(f"  • Architecture     : {arch}")
    print(f"  • Python Version   : {py_ver}")

    if sys.version_info >= (3, 10):
        check_pass("Python Version", f"{py_ver} >= 3.10")
    else:
        check_fail("Python Version", f"{py_ver} is too old (need 3.10+)")
        all_passed = False

    if arch in ["aarch64", "arm64", "x86_64", "AMD64"]:
        check_pass("64-bit Architecture", arch)
    elif "arm" in arch.lower():
        check_warn("32-bit ARM Architecture Detected", f"{arch} - Note: 64-bit Raspberry Pi OS is strongly recommended for ONNX prebuilts")

    # 2. Network Connectivity
    print_section("2. Internet Connectivity & Cloud Reachability")
    net_ok = False
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        check_pass("Internet Reachability", "DNS 8.8.8.8 connected")
        net_ok = True
    except Exception as e:
        check_fail("Internet Reachability", f"No connection ({e})")
        all_passed = False

    # 3. Environment Variables & API Keys
    print_section("3. Environment Variables (.env)")
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        check_pass(".env File Found", env_path)
    else:
        check_warn(".env File Missing", "Run: cp .env.example .env")

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("your_"):
        check_pass("GROQ_API_KEY", f"Present ({groq_key[:6]}...{groq_key[-4:]})")
    else:
        check_warn("GROQ_API_KEY", "Missing or placeholder in .env. Assistant will use local canned answers.")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith("your_"):
        check_pass("OPENAI_API_KEY", f"Present ({openai_key[:6]}...{openai_key[-4:]})")
    else:
        print("  ℹ️  [INFO] OPENAI_API_KEY is optional (used for backup TTS/STT)")

    # 4. Audio Hardware Detection
    print_section("4. Audio Hardware & ALSA Configuration")
    try:
        import sounddevice as sd
        from audio_io import list_audio_devices, find_usb_device, get_working_device_index, play_beep_sound, boost_alsa_system_volume
        
        if sys.platform != 'win32':
            boost_alsa_system_volume()

        mic_idx = get_working_device_index('input')
        spk_idx = get_working_device_index('output')

        usb_mic = find_usb_device('input')
        usb_spk = find_usb_device('output')

        if mic_idx is not None:
            dev_name = sd.query_devices(mic_idx)['name']
            check_pass("Microphone Input Device", f"Index [{mic_idx}]: {dev_name}")
        else:
            check_fail("Microphone Input Device", "No working microphone detected!")
            all_passed = False

        if spk_idx is not None:
            dev_name = sd.query_devices(spk_idx)['name']
            check_pass("Speaker Output Device", f"Index [{spk_idx}]: {dev_name}")
        else:
            check_fail("Speaker Output Device", "No working speaker detected!")
            all_passed = False

        # Quick Beep Audio Test
        print("  🔊 Testing Speaker Hardware (playing short 1200Hz test tone)...")
        try:
            play_beep_sound(device_index=spk_idx, duration_ms=250, volume=0.7)
            check_pass("Speaker Beep Tone Playback", "Audio output triggered successfully")
        except Exception as e:
            check_warn("Speaker Beep Playback", f"Could not play beep: {e}")

    except Exception as e:
        check_fail("Audio Hardware Query", str(e))
        all_passed = False

    # 5. Core Python Modules & Dependencies
    print_section("5. Core Python Libraries")
    packages = [
        ("sounddevice", "sounddevice"),
        ("scipy", "scipy"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("groq", "groq"),
        ("pydub", "pydub"),
        ("edge_tts", "edge-tts"),
        ("gtts", "gTTS"),
        ("onnxruntime", "onnxruntime"),
        ("tqdm", "tqdm"),
        ("pygame", "pygame"),
        ("speech_recognition", "SpeechRecognition")
    ]

    for mod, pkg_name in packages:
        try:
            __import__(mod)
            check_pass(f"Library '{pkg_name}'")
        except (ImportError, OSError, Exception) as e:
            check_fail(f"Library '{pkg_name}'", f"Import error ({e})")
            all_passed = False

    # 6. Local ONNX Models & openWakeWord
    print_section("6. Offline Wake Word & Interrupt Models")
    spark_model = os.path.join(BASE_DIR, "spark.onnx")
    stop_model = os.path.join(BASE_DIR, "stop.onnx")

    if os.path.exists(spark_model):
        size_kb = os.path.getsize(spark_model) / 1024.0
        check_pass("Wake Word Model (spark.onnx)", f"{size_kb:.1f} KB")
    else:
        check_fail("Wake Word Model (spark.onnx)", "File not found")
        all_passed = False

    if os.path.exists(stop_model):
        size_kb = os.path.getsize(stop_model) / 1024.0
        check_pass("Voice Stop Model (stop.onnx)", f"{size_kb:.1f} KB")
    else:
        check_fail("Voice Stop Model (stop.onnx)", "File not found")
        all_passed = False

    try:
        from wakeword import WakeWordDetector
        detector = WakeWordDetector(model_name="spark", threshold=0.40)
        check_pass("openWakeWord Engine", "Initialized successfully with 'spark'")
    except Exception as e:
        check_fail("openWakeWord Engine", str(e))
        all_passed = False

    try:
        from wakeword import StopDetector
        stop_det = StopDetector(model_name="stop", threshold=0.55)
        check_pass("Stop Keyword Detector", "Loaded stop.onnx successfully")
    except Exception as e:
        check_fail("Stop Keyword Detector", str(e))
        all_passed = False

    # 7. LLM Engine & Fast Canned Responses
    print_section("7. LLM Reasoning Engine & Canned Replies")
    try:
        from llm import LLMEngine
        llm = LLMEngine()
        # Test 0ms canned reply
        canned = llm.generate_response("hello")
        if canned:
            check_pass("0ms Canned Response Engine", f"Matched: \"{canned}\"")
        else:
            check_warn("0ms Canned Response", "No reply generated")
    except Exception as e:
        check_fail("LLM Engine Test", str(e))
        all_passed = False

    # 8. Text-to-Speech (TTS)
    print_section("8. Text-to-Speech Engine (Edge-TTS / gTTS / Offline)")
    try:
        from tts import TextToSpeech
        tts = TextToSpeech()
        test_out = os.path.join(BASE_DIR, "test_diag_speech.mp3")
        res_file = tts.synthesize("Robot assistant diagnostic test passed.", output_filepath=test_out)
        if os.path.exists(res_file) and os.path.getsize(res_file) > 0:
            check_pass("TTS Speech Synthesis", f"Generated: {res_file} ({os.path.getsize(res_file)} bytes)")
            # Clean up test audio file
            try:
                os.remove(res_file)
            except Exception:
                pass
        else:
            check_fail("TTS Speech Synthesis", "Output file is empty or missing")
            all_passed = False
    except Exception as e:
        check_warn("TTS Speech Synthesis", f"Synthesis error: {e}")

    # Summary
    print("\n" + "=" * 65)
    if all_passed:
        print("🎉 ALL DIAGNOSTIC CHECKS PASSED!")
        print("Your Raspberry Pi voice assistant hardware and environment are ready.")
        print("\nTo start hands-free voice assistant:")
        print("   python main.py --wake-word")
        print("=" * 65 + "\n")
        return 0
    else:
        print("⚠️ SOME CHECKS NEED ATTENTION. Review the [FAIL] and [WARN] notices above.")
        print("=" * 65 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(run_diagnostics())
