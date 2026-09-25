"""
test_pause_listen.py — Quick test of the pause-then-listen playback flow.
Verifies the scoping fix and that play_audio doesn't crash during playback.
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Generate test audio if needed
audio_file = "response_audio.mp3"
if not os.path.exists(audio_file):
    import edge_tts, asyncio
    async def gen():
        await edge_tts.Communicate(
            "Machine learning is a branch of AI where computers learn patterns from data.",
            "en-IN-NeerjaNeural"
        ).save(audio_file)
    asyncio.run(gen())
    print(f"Generated: {audio_file}")

from wakeword import StopDetector
from audio_io import play_audio, get_working_device_index

print("Loading StopDetector...")
stop_det = StopDetector(model_name="stop", threshold=0.55)

mic_idx = get_working_device_index('input', None)
spk_idx = get_working_device_index('output', None)
print(f"Mic: {mic_idx}, Speaker: {spk_idx}")

print("\n=== TEST: play_audio with stop_detector (pause-then-listen) ===")
print("Playing audio... say 'STOP' during playback to test interrupt.")
print("Or just let it play through to confirm no false triggers.\n")

start = time.time()
was_interrupted = play_audio(
    audio_file,
    device_index=spk_idx,
    enable_interrupt=True,
    mic_device_index=mic_idx,
    wakeword_detector=None,
    stt_engine=None,
    stop_detector=stop_det
)
elapsed = time.time() - start

print(f"\n=== RESULT ===")
print(f"Interrupted: {was_interrupted}")
print(f"Playback time: {elapsed:.2f}s")
if not was_interrupted:
    print("PASS: Audio played to completion without false triggers!")
else:
    print("Audio was interrupted (either by stop keyword or keyboard 'm').")
