"""
main.py - Main Voice Assistant Orchestrator (Headless Systemd & CLI Ready)
Wiring together Audio I/O, STT, LLM, TTS, and openWakeWord with boot retries & signal safety.
"""

import os
import sys
import time
import signal
import builtins
import argparse
from dotenv import load_dotenv

# Hide Pygame community support prompt & suppress C ALSA/ONNXRuntime warnings on load
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "3"

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Load environment variables relative to current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

_orig_print = builtins.print
def safe_print(*args, **kwargs):
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        clean_args = []
        for arg in args:
            if isinstance(arg, str):
                clean_args.append(arg.encode('ascii', 'ignore').decode('ascii'))
            else:
                clean_args.append(arg)
        try:
            _orig_print(*clean_args, **kwargs)
        except Exception:
            pass

builtins.print = safe_print

from audio_io import (
    list_audio_devices, find_usb_device, get_working_device_index,
    record_push_to_talk, record_smart_audio, record_timed_audio, play_audio,
    boost_alsa_system_volume, cleanup_audio_resources, is_shutdown_requested, set_shutdown_flag
)
from stt import SpeechToText
from llm import LLMEngine
from tts import TextToSpeech
from wakeword import WakeWordDetector, StopDetector
from prompts import is_stop_command


def handle_shutdown_signal(signum, frame):
    """Graceful signal handler for SIGTERM and SIGINT (systemd stop / Ctrl+C)."""
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n🛑 [SYSTEM] Received {sig_name} signal. Shutting down Robot Assistant cleanly...", flush=True)
    set_shutdown_flag()
    cleanup_audio_resources()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)


def wait_for_network(max_retries=10, delay=3):
    """Verifies network connectivity before cloud service initialization."""
    import socket
    print("[BOOT] Checking network connectivity...", flush=True)
    for attempt in range(1, max_retries + 1):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            print("[NETWORK] Connected to internet.", flush=True)
            return True
        except Exception:
            print(f"[NETWORK WARNING] Network not available yet (Attempt {attempt}/{max_retries}). Retrying in {delay}s...", flush=True)
            time.sleep(delay)
    print("[NETWORK WARNING] Network check timed out. Proceeding with local offline capabilities.", flush=True)
    return False


def get_device_indices():
    """Retrieves mic & speaker indices from env or resolves working audio hardware."""
    mic_str = os.getenv("MIC_DEVICE_INDEX")
    spk_str = os.getenv("SPEAKER_DEVICE_INDEX")

    mic_idx = int(mic_str.strip()) if mic_str and mic_str.strip().lstrip('-').isdigit() else None
    spk_idx = int(spk_str.strip()) if spk_str and spk_str.strip().lstrip('-').isdigit() else None

    resolved_mic = get_working_device_index('input', mic_idx)
    resolved_spk = get_working_device_index('output', spk_idx)

    return resolved_mic, resolved_spk


def wait_for_audio_hardware(max_retries=10, delay=3):
    """Waits for USB mic and speaker devices to register after Raspberry Pi boot."""
    print("[BOOT] Checking USB microphone and speaker hardware...", flush=True)
    for attempt in range(1, max_retries + 1):
        mic_idx, spk_idx = get_device_indices()
        if mic_idx is not None and spk_idx is not None:
            return mic_idx, spk_idx
        print(f"[AUDIO WARNING] Waiting for audio hardware initialization (Attempt {attempt}/{max_retries})...", flush=True)
        time.sleep(delay)
    return get_device_indices()


def get_device_name(index, kind='input'):
    if index is None:
        return "None Found"
    try:
        import sounddevice as sd
        dev = sd.query_devices(index)
        return f"[{index}] {dev['name']}"
    except Exception:
        return f"[{index}]"


def print_banner(wake_word_enabled=False, wake_model="max", voice_name="en-IN-NeerjaNeural", llm_model="qwen/qwen3.8-27b"):
    display_model = wake_model.replace("_", " ").title()
    mode_str = f"Hands-Free Wake Word ('{display_model}')" if wake_word_enabled else "Push-To-Talk (ENTER)"
    banner = f"""
    ============================================================
    🤖 RASPBERRY PI CLOUD VOICE ASSISTANT (HEADLESS SYSTEMD READY)
    ------------------------------------------------------------
    Interaction Mode : {mode_str}
    STT Engine       : Groq Whisper / OpenAI Whisper API
    LLM Engine       : Groq API ({llm_model})
    TTS Engine       : Edge-TTS Neural (Voice: {voice_name})
    ============================================================
    """
    print(banner, flush=True)


def main():
    print("[BOOT] Robot Assistant starting...", flush=True)

    parser = argparse.ArgumentParser(description="Raspberry Pi Cloud Voice Assistant")
    parser.add_argument("--wake-word", "-w", action="store_true", help="Enable hands-free wake word detection ('Max')")
    parser.add_argument("--push-to-talk", "-p", action="store_true", help="Use manual ENTER key push-to-talk mode")
    default_model = os.getenv("WAKE_MODEL", "spark")
    parser.add_argument("--wake-model", type=str, default=default_model, help=f"openWakeWord model name (default: {default_model})")
    default_threshold = float(os.getenv("WAKE_THRESHOLD", "0.40"))
    parser.add_argument("--wake-threshold", type=float, default=default_threshold, help=f"Wake word confidence threshold (default: {default_threshold})")
    default_voice = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
    parser.add_argument("--voice", "-v", type=str, default=default_voice, help=f"TTS voice (default: {default_voice})")
    parser.add_argument("--barge-in", action="store_true", help="Enable experimental voice barge-in listener during audio playback")
    args = parser.parse_args()

    # If running without interactive TTY (e.g. systemd service), force hands-free wake word mode
    is_interactive = sys.stdin.isatty()
    if not is_interactive and not args.push_to_talk:
        wake_word_enabled = True
    elif args.push_to_talk:
        wake_word_enabled = False
    else:
        wake_word_enabled = args.wake_word or os.getenv("WAKE_WORD_MODE", "true").lower() in ("true", "1", "yes")

    # Step 1: Wait for network readiness (Wi-Fi/Ethernet)
    wait_for_network(max_retries=8, delay=3)

    # Step 2: Detect Audio Hardware & Boost System Volume
    boost_alsa_system_volume()
    mic_idx, spk_idx = wait_for_audio_hardware(max_retries=10, delay=3)

    print(f"[AUDIO] Microphone detected: {get_device_name(mic_idx, 'input')}", flush=True)
    print(f"[AUDIO] Speaker detected: {get_device_name(spk_idx, 'output')}", flush=True)

    # Step 3: Initialize API Services & Wake-Word Engine
    print("[BOOT] Initializing API Clients & Speech Engines...", flush=True)
    try:
        stt_engine = SpeechToText()
        print("[STT] Ready", flush=True)

        llm_engine = LLMEngine()
        print("[LLM] Ready", flush=True)

        tts_engine = TextToSpeech(voice=args.voice)
        print(f"[TTS] Ready (Active Voice: {tts_engine.voice})", flush=True)
        
        wakeword_detector = None
        if wake_word_enabled:
            wakeword_detector = WakeWordDetector(model_name=args.wake_model, threshold=args.wake_threshold)
            print(f"[WAKEWORD] Ready (Model: '{args.wake_model}', Threshold: {args.wake_threshold})", flush=True)

        stop_detector = None
        stop_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop.onnx")
        if os.path.exists(stop_model_path):
            try:
                stop_thresh = float(os.getenv("STOP_THRESHOLD", "0.55"))
                stop_detector = StopDetector(model_name="stop", threshold=stop_thresh)
                print(f"[STOPDETECTOR] Ready (Model: 'stop.onnx', Threshold: {stop_detector.threshold})", flush=True)
            except Exception as e:
                print(f"⚠️ [STOPDETECTOR] Could not load stop model: {e}", file=sys.stderr, flush=True)

        print_banner(wake_word_enabled=wake_word_enabled, wake_model=args.wake_model, voice_name=tts_engine.voice, llm_model=llm_engine.model)
        print("[SYSTEM] Robot Assistant ready and listening.", flush=True)
    except Exception as e:
        print(f"❌ [BOOT ERROR] Failed to initialize services: {e}", file=sys.stderr, flush=True)
        print("Please check your .env file keys and dependencies.", file=sys.stderr, flush=True)
        sys.exit(1)

    print("------------------------------------------------------------", flush=True)
    if wake_word_enabled:
        print(f"  • Mode: HANDS-FREE (Say '{args.wake_model}' to start speaking)", flush=True)
        print("  • Press Ctrl+C or send SIGTERM to stop", flush=True)
    else:
        print("  • Mode: PUSH-TO-TALK", flush=True)
        print("  • Press [ENTER] to record speech (ENTER again to stop)", flush=True)
        print("  • Type 'clear' + ENTER to reset short-term memory", flush=True)
        print("  • Type 'q' or 'exit' + ENTER to quit", flush=True)
    print("------------------------------------------------------------\n", flush=True)

    # Step 4: Main Conversation Loop
    in_active_session = False

    while not is_shutdown_requested():
        try:
            if wake_word_enabled:
                if not in_active_session:
                    triggered = wakeword_detector.listen_for_wakeword(device_index=mic_idx)
                    if is_shutdown_requested():
                        break
                    if not triggered:
                        time.sleep(0.5)
                        continue
                    print(f"✨ [WAKE WORD DETECTED] Speak your prompt now...", flush=True)
                    in_active_session = True
            else:
                if not is_interactive:
                    print("⚠️ Push-to-talk mode requires interactive terminal. Switching to wake-word mode...", flush=True)
                    wake_word_enabled = True
                    continue
                cmd = input("\n👉 Press [ENTER] to start recording (or type q to exit): ").strip()
                if cmd.lower() in ['q', 'exit', 'quit']:
                    print("👋 Exiting Voice Assistant. Goodbye!", flush=True)
                    break
                elif cmd.lower() == 'clear':
                    llm_engine.clear_memory()
                    print("🧹 Conversation short-term memory cleared.", flush=True)
                    continue

            pipeline_start = time.time()

            # 1. AUDIO RECORDING
            max_rec_sec = float(os.getenv("MAX_RECORD_SECONDS", "5.0"))
            if wake_word_enabled:
                audio_file, rec_dur = record_smart_audio("input_audio.wav", device_index=mic_idx, silence_timeout=1.0, max_duration=max_rec_sec)
            else:
                print("\n🎙️  [1/4 LISTENING] Recording... Press [ENTER] to stop.", flush=True)
                audio_file, rec_dur = record_push_to_talk("input_audio.wav", device_index=mic_idx)

            if is_shutdown_requested():
                break

            if rec_dur < 0.5:
                print("⚠️ Recording too short (< 0.5s). Skipping...", flush=True)
                in_active_session = False
                continue

            # 2. SPEECH TO TEXT
            print("📝 [2/4 TRANSCRIBING] Converting audio to text...", flush=True)
            stt_start = time.time()
            user_text = stt_engine.transcribe(audio_file)
            stt_latency = time.time() - stt_start

            if not user_text:
                print("⚠️ No speech detected in recording. Resetting to wake-word mode...", flush=True)
                in_active_session = False
                continue

            print(f"🗣️  User Said: \"{user_text}\"", flush=True)

            # 2a. VOICE INTERRUPT / STOP COMMAND CHECK
            if is_stop_command(user_text):
                clean_stop = user_text.lower().strip()
                in_active_session = False
                if any(kw in clean_stop for kw in ["exit", "quit", "goodbye", "bye", "shutdown"]):
                    print(f"\n🛑 [STOP COMMAND] Exiting voice assistant per user request (\"{user_text}\"). Goodbye!\n", flush=True)
                    break
                else:
                    print(f"\n🛑 [STOP COMMAND] Stop command detected (\"{user_text}\"). Resetting to wake-word mode...\n", flush=True)
                    time.sleep(1.0)
                    continue

            # 3. LLM INFERENCE
            print("🧠 [3/4 THINKING] Requesting response from LLM...", flush=True)
            llm_start = time.time()
            response_text = llm_engine.generate_response(user_text)
            llm_latency = time.time() - llm_start

            print(f"🤖 Assistant: \"{response_text}\"", flush=True)

            # Guard: skip TTS if LLM returned empty/whitespace response
            if not response_text or not response_text.strip():
                print("⚠️ LLM returned empty response. Skipping TTS.", flush=True)
                in_active_session = False
                continue

            # 4. TEXT TO SPEECH
            print("🔊 [4/4 SPEAKING] Synthesizing speech...", flush=True)
            tts_start = time.time()
            speech_file = tts_engine.synthesize(response_text, "response_audio.mp3")
            tts_latency = time.time() - tts_start

            if is_shutdown_requested():
                break

            # 5. AUDIO PLAYBACK (VOICE 'STOP' & KEYBOARD 'M' INTERRUPT ACTIVE)
            stop_notice = "Say 'STOP' or press 'm' to stop speech" if stop_detector else "Press 'm' to stop speech"
            print(f"🔊 [4/4 SPEAKING] Playing speech response... ({stop_notice})", flush=True)
            play_start = time.time()
            was_interrupted = play_audio(
                speech_file,
                device_index=spk_idx,
                enable_interrupt=True,
                mic_device_index=mic_idx,
                wakeword_detector=None,   # Disabled during playback — own TTS voice triggers false wake word hits
                stt_engine=None,
                stop_detector=stop_detector  # Local ONNX 'stop' keyword handles voice barge-in
            )
            play_latency = time.time() - play_start

            if was_interrupted:
                print("\n🛑 [INTERRUPTED] TTS playback stopped by user interrupt.", flush=True)
                in_active_session = False
                time.sleep(0.3)
                continue

            time.sleep(0.3)

            total_latency = time.time() - pipeline_start

            # TIMING & LATENCY LOG
            print("\n" + "-" * 50, flush=True)
            print("⏱️  STAGE LATENCY SUMMARY:", flush=True)
            print(f"  • STT Latency   : {stt_latency:.2f}s", flush=True)
            print(f"  • LLM Latency   : {llm_latency:.2f}s", flush=True)
            print(f"  • TTS Latency   : {tts_latency:.2f}s", flush=True)
            print(f"  • Playback Time : {play_latency:.2f}s", flush=True)
            print(f"  • Total Turn    : {total_latency:.2f}s (Processing: {stt_latency + llm_latency + tts_latency:.2f}s)", flush=True)
            print("-" * 50, flush=True)
            in_active_session = False

        except KeyboardInterrupt:
            print("\n👋 Voice assistant stopped by user interrupt. Goodbye!", flush=True)
            break
        except Exception as err:
            print(f"\n❌ Pipeline Exception Error: {err}", file=sys.stderr, flush=True)
            in_active_session = False
            if is_interactive and not wake_word_enabled:
                print("Continuing loop... Press ENTER to try again.", flush=True)

    cleanup_audio_resources()
    print("[SYSTEM] Robot Assistant shut down successfully.", flush=True)

if __name__ == "__main__":
    main()
