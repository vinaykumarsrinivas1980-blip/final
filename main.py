"""
main.py - Main Voice Assistant Orchestrator
Interactive Push-to-Talk CLI loop wiring together Audio I/O, STT, LLM, and TTS.
"""

import os
import sys
import time

# Hide Pygame community support prompt & suppress C ALSA/ONNXRuntime warnings on load
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "3"

from dotenv import load_dotenv
import argparse
from audio_io import list_audio_devices, find_usb_device, get_working_device_index, record_push_to_talk, record_smart_audio, record_timed_audio, play_audio, boost_alsa_system_volume
from stt import SpeechToText
from llm import LLMEngine
from tts import TextToSpeech
from wakeword import WakeWordDetector
from prompts import is_stop_command

load_dotenv()


import builtins

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

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


def print_banner(wake_word_enabled=False, wake_model="max", voice_name="en-IN-NeerjaNeural"):
    display_model = wake_model.replace("_", " ").title()
    mode_str = f"Hands-Free Wake Word ('{display_model}')" if wake_word_enabled else "Push-To-Talk (ENTER)"
    banner = f"""
    ============================================================
    🤖 RASPBERRY PI CLOUD VOICE ASSISTANT
    ------------------------------------------------------------
    Interaction Mode : {mode_str}
    STT Engine       : Groq Whisper / OpenAI Whisper API
    LLM Engine       : Groq API (Llama 3.3 70B)
    TTS Engine       : Edge-TTS Neural (Voice: {voice_name})
    ============================================================
    """
    print(banner)

def get_device_indices():
    """Retrieves mic & speaker indices from env or resolves working audio hardware."""
    mic_str = os.getenv("MIC_DEVICE_INDEX")
    spk_str = os.getenv("SPEAKER_DEVICE_INDEX")

    mic_idx = int(mic_str.strip()) if mic_str and mic_str.strip().lstrip('-').isdigit() else None
    spk_idx = int(spk_str.strip()) if spk_str and spk_str.strip().lstrip('-').isdigit() else None

    resolved_mic = get_working_device_index('input', mic_idx)
    resolved_spk = get_working_device_index('output', spk_idx)

    return resolved_mic, resolved_spk

def get_device_name(index, kind='input'):
    if index is None:
        return "None Found"
    try:
        import sounddevice as sd
        dev = sd.query_devices(index)
        return f"[{index}] {dev['name']}"
    except Exception:
        return f"[{index}]"

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Cloud Voice Assistant")
    parser.add_argument("--wake-word", "-w", action="store_true", help="Enable hands-free wake word detection ('Max')")
    parser.add_argument("--push-to-talk", "-p", action="store_true", help="Use manual ENTER key push-to-talk mode")
    default_model = os.getenv("WAKE_MODEL", "max")
    parser.add_argument("--wake-model", type=str, default=default_model, help=f"openWakeWord model name (default: {default_model})")
    default_voice = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")
    parser.add_argument("--voice", "-v", type=str, default=default_voice, help=f"TTS voice (default: {default_voice})")
    parser.add_argument("--barge-in", action="store_true", help="Enable experimental voice barge-in listener during audio playback")
    args = parser.parse_args()

    if args.push_to_talk:
        wake_word_enabled = False
    else:
        wake_word_enabled = args.wake_word or os.getenv("WAKE_WORD_MODE", "true").lower() in ("true", "1", "yes")

    # Step 1: Initialize API Services & Wake-Word
    print("\n⏳ Initializing API Clients & Engines...")
    try:
        stt_engine = SpeechToText()
        llm_engine = LLMEngine()
        tts_engine = TextToSpeech(voice=args.voice)
        
        wakeword_detector = None
        if wake_word_enabled:
            wakeword_detector = WakeWordDetector(model_name=args.wake_model, threshold=0.20)

        print_banner(wake_word_enabled=wake_word_enabled, wake_model=args.wake_model, voice_name=tts_engine.voice)
        print("✅ All services initialized successfully.")
        print(f"🔊 Active TTS Voice: {tts_engine.voice}\n")
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}", file=sys.stderr)
        print("Please check your .env file keys and dependencies.")
        sys.exit(1)

    # Step 2: Detect Audio Hardware & Boost System Volume
    boost_alsa_system_volume()
    mic_idx, spk_idx = get_device_indices()
    print(f"🎙️  Microphone Device Index : {get_device_name(mic_idx, 'input')}")
    print(f"🔊 Speaker Device Index    : {get_device_name(spk_idx, 'output')}")

    print("------------------------------------------------------------")
    if wake_word_enabled:
        print(f"  • Mode: HANDS-FREE (Say '{args.wake_model}' to start speaking)")
        print("  • Press Ctrl+C to stop")
    else:
        print("  • Mode: PUSH-TO-TALK")
        print("  • Press [ENTER] to record speech (ENTER again to stop)")
        print("  • Type 'clear' + ENTER to reset short-term memory")
        print("  • Type 'q' or 'exit' + ENTER to quit")
    print("------------------------------------------------------------\n")

    # Step 3: Main Conversation Loop
    in_active_session = False

    while True:
        try:
            if wake_word_enabled:
                if not in_active_session:
                    triggered = wakeword_detector.listen_for_wakeword(device_index=mic_idx)
                    if not triggered:
                        time.sleep(0.5)
                        continue
                    print(f"✨ [WAKE WORD DETECTED] Speak your prompt now...")
                    in_active_session = True
            else:
                cmd = input("\n👉 Press [ENTER] to start recording (or type q to exit): ").strip()
                if cmd.lower() in ['q', 'exit', 'quit']:
                    print("👋 Exiting Voice Assistant. Goodbye!")
                    break
                elif cmd.lower() == 'clear':
                    llm_engine.clear_memory()
                    print("🧹 Conversation short-term memory cleared.")
                    continue

            pipeline_start = time.time()

            # 1. AUDIO RECORDING (Dynamic VAD in hands-free mode)
            max_rec_sec = float(os.getenv("MAX_RECORD_SECONDS", "5.0"))
            if wake_word_enabled:
                audio_file, rec_dur = record_smart_audio("input_audio.wav", device_index=mic_idx, silence_timeout=1.0, max_duration=max_rec_sec)
            else:
                print("\n🎙️  [1/4 LISTENING] Recording... Press [ENTER] to stop.")
                audio_file, rec_dur = record_push_to_talk("input_audio.wav", device_index=mic_idx)


            if rec_dur < 0.5:
                print("⚠️ Recording too short (< 0.5s). Skipping...")
                in_active_session = False
                continue

            # 2. SPEECH TO TEXT
            print("📝 [2/4 TRANSCRIBING] Converting audio to text...")
            stt_start = time.time()
            user_text = stt_engine.transcribe(audio_file)
            stt_latency = time.time() - stt_start

            if not user_text:
                print("⚠️ No speech detected in recording. Resetting to wake-word mode...")
                in_active_session = False
                continue

            print(f"🗣️  User Said: \"{user_text}\"")

            # 2a. VOICE INTERRUPT / STOP COMMAND CHECK
            if is_stop_command(user_text):
                clean_stop = user_text.lower().strip()
                in_active_session = False
                if any(kw in clean_stop for kw in ["exit", "quit", "goodbye", "bye", "shutdown"]):
                    print(f"\n🛑 [STOP COMMAND] Exiting voice assistant per user request (\"{user_text}\"). Goodbye!\n")
                    break
                else:
                    print(f"\n🛑 [STOP COMMAND] Stop command detected (\"{user_text}\"). Resetting to wake-word mode...\n")
                    time.sleep(1.0)
                    continue

            # 3. LLM INFERENCE
            print("🧠 [3/4 THINKING] Requesting response from LLM...")
            llm_start = time.time()
            response_text = llm_engine.generate_response(user_text)
            llm_latency = time.time() - llm_start

            print(f"🤖 Assistant: \"{response_text}\"")

            # 4. TEXT TO SPEECH
            print("🔊 [4/4 SPEAKING] Synthesizing speech...")
            tts_start = time.time()
            speech_file = tts_engine.synthesize(response_text, "response_audio.mp3")
            tts_latency = time.time() - tts_start

            # 5. AUDIO PLAYBACK (LISTENING FOR WAKE WORDS AND STOP COMMANDS)
            play_start = time.time()
            was_interrupted = play_audio(
                speech_file,
                device_index=spk_idx,
                enable_interrupt=True,
                mic_device_index=mic_idx,
                wakeword_detector=wakeword_detector,
                stt_engine=stt_engine
            )
            play_latency = time.time() - play_start

            if was_interrupted:
                print("\n🛑 [INTERRUPTED] Playback stopped by user.")
                in_active_session = False
                time.sleep(0.3)
                continue

            time.sleep(0.3)

            total_latency = time.time() - pipeline_start

            # TIMING & LATENCY LOG
            print("\n" + "-" * 50)
            print("⏱️  STAGE LATENCY SUMMARY:")
            print(f"  • STT Latency   : {stt_latency:.2f}s")
            print(f"  • LLM Latency   : {llm_latency:.2f}s")
            print(f"  • TTS Latency   : {tts_latency:.2f}s")
            print(f"  • Playback Time : {play_latency:.2f}s")
            print(f"  • Total Turn    : {total_latency:.2f}s (Processing: {stt_latency + llm_latency + tts_latency:.2f}s)")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n👋 Voice assistant stopped by user interrupt. Goodbye!")
            break
        except Exception as err:
            print(f"\n❌ Pipeline Exception Error: {err}", file=sys.stderr)
            if not wake_word_enabled:
                print("Continuing loop... Press ENTER to try again.")

if __name__ == "__main__":
    main()

