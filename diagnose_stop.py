"""
diagnose_stop.py — Live diagnostic for stop keyword detection during TTS playback.
Plays response_audio.mp3 while monitoring mic RMS, adaptive baseline, and stop.onnx scores.
"""

import os, sys, time, threading, numpy as np
import sounddevice as sd
from scipy import signal as sig

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from audio_io import get_working_device_index, suppress_c_stderr
from wakeword import StopDetector

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

def main():
    mic_idx = get_working_device_index('input', None)
    spk_idx = get_working_device_index('output', None)
    print(f"Mic: {mic_idx}, Speaker: {spk_idx}")

    # Load stop detector
    stop_det = StopDetector(model_name="stop", threshold=0.55)

    # Find audio to play
    audio_file = "response_audio.mp3"
    if not os.path.exists(audio_file):
        # Generate a quick test clip
        import edge_tts, asyncio
        async def gen():
            await edge_tts.Communicate(
                "Artificial intelligence is a field of computing that helps machines perform tasks that usually need human intelligence.",
                "en-IN-NeerjaNeural"
            ).save(audio_file)
        asyncio.run(gen())
        print(f"Generated test audio: {audio_file}")

    # Start pygame playback
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
    pygame.mixer.music.load(audio_file)

    rms_history = []
    RMS_HISTORY_MAX = 30
    RMS_SPIKE_FACTOR = 2.5
    RMS_MINIMUM_FLOOR = 300
    stop_consecutive = 0

    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Playing audio while monitoring mic")
    print("Say 'STOP' loudly during playback to test detection")
    print("=" * 80)
    print(f"{'Frame':>6} | {'RMS':>8} | {'Baseline':>8} | {'Spike?':>6} | {'Threshold':>10} | {'ONNX Score':>10} | {'Consec':>6} | {'Action'}")
    print("-" * 80)

    frame_count = [0]
    stop_event = threading.Event()

    def mic_callback(indata, frames, time_info, status):
        nonlocal stop_consecutive
        if stop_event.is_set():
            return

        frame_count[0] += 1
        audio_f32 = indata.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))

        # Update baseline
        rms_history.append(rms)
        if len(rms_history) > RMS_HISTORY_MAX:
            rms_history.pop(0)

        baseline = sum(rms_history) / len(rms_history) if rms_history else 0.0
        spike_thresh = max(baseline * RMS_SPIKE_FACTOR, RMS_MINIMUM_FLOOR)
        is_spike = rms > spike_thresh

        # Check stop detector
        onnx_score = 0.0
        action = ""

        if is_spike:
            # Run the ONNX model
            if indata.ndim > 1 and indata.shape[1] > 1:
                frame = indata.mean(axis=1).astype(np.float32)
            else:
                frame = indata.flatten().astype(np.float32)
            if len(frame) != CHUNK_SIZE and len(frame) > 0:
                frame = sig.resample(frame, CHUNK_SIZE)
            frame_i16 = np.clip(frame * 1.5, -32768, 32767).astype(np.int16)
            pred = stop_det.model.predict(frame_i16)
            onnx_score = pred.get('stop', 0.0)

            if onnx_score >= stop_det.threshold:
                stop_consecutive += 1
                if stop_consecutive >= 2:
                    action = ">>> STOP TRIGGERED <<<"
                    stop_event.set()
                    try:
                        pygame.mixer.music.stop()
                    except:
                        pass
                else:
                    action = f"hit {stop_consecutive}/2"
            else:
                stop_consecutive = 0
        else:
            stop_consecutive = 0

        # Print every 3rd frame to reduce spam
        if frame_count[0] % 3 == 0 or is_spike or action:
            spike_str = "YES" if is_spike else "no"
            print(f"{frame_count[0]:>6} | {rms:>8.0f} | {baseline:>8.0f} | {spike_str:>6} | {spike_thresh:>10.0f} | {onnx_score:>10.4f} | {stop_consecutive:>6} | {action}")

    # Query mic sample rate
    try:
        dev_info = sd.query_devices(mic_idx, 'input')
        hw_sr = int(dev_info.get('default_samplerate', 44100))
    except:
        hw_sr = 44100

    block_size = int(round(0.08 * hw_sr))

    print(f"\nMic hardware sample rate: {hw_sr} Hz, block size: {block_size}")
    print("Starting playback + mic monitoring...\n")

    pygame.mixer.music.play()

    try:
        with suppress_c_stderr():
            with sd.InputStream(
                samplerate=hw_sr,
                blocksize=block_size,
                channels=1,
                dtype='int16',
                device=mic_idx,
                callback=mic_callback
            ):
                while pygame.mixer.music.get_busy() and not stop_event.is_set():
                    sd.sleep(50)
    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 80)
    if stop_event.is_set():
        print("RESULT: Stop command was detected!")
    else:
        print("RESULT: Playback completed without stop trigger.")
    print(f"Total frames processed: {frame_count[0]}")
    if rms_history:
        print(f"Final baseline RMS: {sum(rms_history)/len(rms_history):.0f}")
        print(f"Max RMS seen: {max(rms_history):.0f}")
    print("=" * 80)

    pygame.mixer.quit()

if __name__ == "__main__":
    main()
