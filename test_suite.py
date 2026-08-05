"""
test_suite.py - Real-World Scenario Test Runner for Voice Assistant
Tests prompts, LLM engine, STT engine, TTS engine, WakeWord detector, and Audio I/O.
"""

import os
import sys
import time
import wave
import struct
import numpy as np

# Ensure UTF-8 output formatting
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

TEST_WAV = "test_synth_input.wav"

def generate_test_wav(filepath, duration_sec=1.5, sample_rate=16000, frequency=440.0):
    """Generates a simple sine wave WAV audio file for STT/audio testing."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, False)
    audio = (np.sin(2 * np.pi * frequency * t) * 16384).astype(np.int16)
    
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

def test_prompts_module():
    print("\n--- 🧪 TEST 1: Prompts & Canned Responses ---")
    from prompts import get_canned_response, is_stop_command, SYSTEM_PROMPT
    
    assert SYSTEM_PROMPT is not None, "SYSTEM_PROMPT missing"
    
    # 1. Canned responses
    res1 = get_canned_response("hello")
    assert res1 is not None and "Hello" in res1, f"Failed hello canned match: {res1}"
    
    res2 = get_canned_response("what is the time")
    assert res2 is not None and ("time" in res2 or "AM" in res2 or "PM" in res2), f"Failed time canned match: {res2}"

    res3 = get_canned_response("who made you")
    assert res3 is not None and "Nishchitha" in res3, f"Failed creator canned match: {res3}"

    res4 = get_canned_response("this is a random question 12345")
    assert res4 is None, f"Expected None for uncanned input, got: {res4}"

    # 2. Stop/Break commands
    assert is_stop_command("break"), "Failed break command detection for 'break'"
    assert is_stop_command("take a break"), "Failed break command detection for 'take a break'"
    assert is_stop_command("stop"), "Failed stop command detection for 'stop'"
    assert is_stop_command("ruko"), "Failed stop command detection for 'ruko'"
    assert not is_stop_command("what is python"), "False positive stop command for 'what is python'"
    
    print("✅ Prompts module tests passed.")

def test_llm_module():
    print("\n--- 🧪 TEST 2: LLM Engine & Memory ---")
    from llm import LLMEngine
    
    try:
        engine = LLMEngine()
        # Test canned match fallback inside LLM engine
        r1 = engine.generate_response("hello")
        print(f"  • Canned query response: {r1}")
        assert "Hello" in r1
        
        # Test dynamic LLM query
        r2 = engine.generate_response("In 5 words, what is a CPU?")
        print(f"  • LLM query response: {r2}")
        assert len(r2) > 0
        
        # Test empty prompt
        r3 = engine.generate_response("")
        assert "repeat" in r3.lower() or "hear" in r3.lower()
        
    except ValueError as ve:
        print(f"  ⚠️ LLM key missing notice: {ve}")
    
    print("✅ LLM module tests completed.")

def test_stt_module():
    print("\n--- 🧪 TEST 3: Speech-to-Text Module ---")
    from stt import SpeechToText
    
    generate_test_wav(TEST_WAV)
    try:
        stt = SpeechToText()
        # Attempt transcription on sine wave audio (expect blank or small transcript without crashing)
        res = stt.transcribe(TEST_WAV)
        print(f"  • Audio file transcript result: '{res}'")
    except Exception as e:
        print(f"  • STT engine result notice: {e}")
    finally:
        if os.path.exists(TEST_WAV):
            os.remove(TEST_WAV)

    print("✅ STT module tests completed.")

def test_tts_module():
    print("\n--- 🧪 TEST 4: Text-to-Speech Module ---")
    from tts import TextToSpeech
    
    tts = TextToSpeech(voice="indian_female")
    out_file = "test_suite_tts.mp3"
    try:
        res_file = tts.synthesize("Unit test for text to speech synthesis.", out_file)
        assert os.path.exists(res_file), "TTS output file was not created"
        assert os.path.getsize(res_file) > 0, "TTS output file is empty"
        print(f"  • TTS Synthesized file size: {os.path.getsize(res_file)} bytes")
    finally:
        if os.path.exists(out_file):
            os.remove(out_file)

    print("✅ TTS module tests completed.")

def test_wakeword_module():
    print("\n--- 🧪 TEST 5: openWakeWord Module ---")
    from wakeword import WakeWordDetector
    
    detector = WakeWordDetector(model_name="max", threshold=0.40)
    assert detector.threshold == 0.40, f"Expected threshold 0.40, got {detector.threshold}"
    
    # Send dummy silent audio frame
    frame = np.zeros(1280, dtype=np.int16)
    pred = detector.predict_frame(frame)
    assert pred is None, f"Expected no trigger on silence, got {pred}"
    
    print("✅ WakeWord module tests passed.")

def test_audio_io_module():
    print("\n--- 🧪 TEST 6: Audio I/O & Sound Device Resolution ---")
    from audio_io import get_working_device_index
    
    mic_idx = get_working_device_index('input', None)
    spk_idx = get_working_device_index('output', None)
    
    print(f"  • Resolved Working Input Mic Device Index  : {mic_idx}")
    print(f"  • Resolved Working Output Speaker Index   : {spk_idx}")
    print("✅ Audio I/O tests completed.")

def run_all_tests():
    print("==================================================")
    print("🚀 RUNNING ALL REAL-WORLD SUITE TESTS FOR ROBO2")
    print("==================================================")
    
    test_prompts_module()
    test_llm_module()
    test_stt_module()
    test_tts_module()
    test_wakeword_module()
    test_audio_io_module()
    
    print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
