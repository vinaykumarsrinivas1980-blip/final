"""
wakeword.py - Open-Source Offline Wake-Word Detection Module
Uses openWakeWord to detect custom wake phrases ("hey_jarvis", "alexa", etc.) locally.
100% Free, Privacy-Friendly, and requires NO API keys or internet connection.
"""

import os
import sys
import time
import numpy as np

# Suppress ONNXRuntime C-level GPU discovery warnings on ARM / Raspberry Pi
os.environ["ORT_LOGGING_LEVEL"] = "3"
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except Exception:
    pass

import sounddevice as sd
from scipy import signal
from audio_io import get_working_device_index, suppress_c_stderr, is_shutdown_requested

import io
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# openWakeWord expects 16kHz mono audio, processed in frames of 1280 samples (80 ms)
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

# Map custom/short wake word names to openWakeWord pretrained models
WAKE_WORD_ALIASES = {
    "max": "alexa",
    "spark": "alexa",
    "jarvis": "hey_jarvis",
    "mycroft": "hey_mycroft",
    "rhasspy": "hey_rhasspy",
}

class WakeWordDetector:
    def __init__(self, model_name="max", threshold=None):
        """
        Initializes openWakeWord model.

        :param model_name: Primary model name or trigger phrase ('max', 'alexa', 'hey_jarvis', 'hey_mycroft')
        :param threshold: Detection confidence threshold between 0.0 and 1.0 (default 0.40)
        """
        self.model_name = model_name
        if threshold is None:
            try:
                threshold = float(os.getenv("WAKE_THRESHOLD", "0.40"))
            except Exception:
                threshold = 0.40
        self.threshold = max(0.05, min(0.95, threshold))
        self.oww_model = None
        self._load_model()

    def _load_model(self):
        try:
            import openwakeword
            from openwakeword.model import Model
            
            # Download models if not already cached locally (with try/except to prevent startup hangs)
            try:
                with suppress_c_stderr():
                    openwakeword.utils.download_models()
            except Exception:
                pass

            # Resolve model path / alias (e.g., check for local custom spark.onnx model first)
            target_key = self.model_name.lower().strip()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            custom_model_path = os.path.join(base_dir, f"{target_key}.onnx")

            models_to_try = []
            if os.path.exists(custom_model_path):
                models_to_try.append(custom_model_path)
            
            actual_model = WAKE_WORD_ALIASES.get(target_key, target_key)
            if actual_model not in models_to_try:
                models_to_try.append(actual_model)
            
            for fallback in ["alexa", "hey_jarvis"]:
                if fallback not in models_to_try:
                    models_to_try.append(fallback)

            with suppress_c_stderr():
                try:
                    self.oww_model = Model(wakeword_models=models_to_try, inference_framework="onnx")
                except Exception:
                    self.oww_model = Model(wakeword_models=["alexa", "hey_jarvis"], inference_framework="onnx")

            display_phrase = self.model_name.replace("_", " ").title()
            print(f"✅ [WakeWord] Loaded openWakeWord models (Trigger Phrase: '{display_phrase}', Sensitivity Threshold: {self.threshold})")
        except Exception as e:
            print(f"❌ [WakeWord Error] Failed to initialize openWakeWord: {e}", file=sys.stderr)
            print("💡 Tip for Raspberry Pi / Python 3.13: Install openwakeword without tflite-runtime by running:", file=sys.stderr)
            print("   pip install scikit-learn onnxruntime && pip install openwakeword --no-deps", file=sys.stderr)
            raise e

    def predict_frame(self, indata, override_threshold=None):
        """Processes an audio chunk (mono/stereo, any sample rate) and returns triggered model name or None."""
        if not self.oww_model:
            return None

        thresh = override_threshold if override_threshold is not None else self.threshold

        # Convert to mono if 2D array (stereo)
        if indata.ndim > 1 and indata.shape[1] > 1:
            audio_frame = indata.mean(axis=1).astype(np.float32)
        else:
            audio_frame = indata.flatten().astype(np.float32)

        # Resample to 1280 samples (16kHz 80ms chunk) if captured at a hardware sample rate != 16000Hz
        if len(audio_frame) != CHUNK_SIZE and len(audio_frame) > 0:
            audio_frame = signal.resample(audio_frame, CHUNK_SIZE)

        audio_frame = np.clip(audio_frame * 1.5, -32768, 32767).astype(np.int16)
        prediction = self.oww_model.predict(audio_frame)

        target_key = self.model_name.lower().strip()
        actual_target = WAKE_WORD_ALIASES.get(target_key, target_key)

        for m_name, score in prediction.items():
            if score >= thresh:
                # Prioritize target model or high confidence trigger
                if m_name == actual_target or m_name == target_key or score >= max(thresh * 1.5, 0.40):
                    return m_name
        return None

    def listen_for_wakeword(self, device_index=None, timeout=None):
        """
        Listens to microphone input until the wake word is detected.

        :param device_index: Microphone device index for sounddevice (None for default mic).
        :param timeout: Optional timeout in seconds to give up listening.
        :return: True if wake word detected, False if timed out or interrupted.
        """
        if not self.oww_model:
            raise RuntimeError("WakeWordDetector model is not initialized.")

        if self.oww_model:
            self.oww_model.reset()

        display_phrase = self.model_name.replace("_", " ").title()
        print(f"\n👂 [WAKE WORD] Listening for '{display_phrase}'...")
        start_time = time.time()
        detected = False
        detected_name = ""

        target_device = get_working_device_index('input', device_index)

        # Build list of sample rates and channel counts supported by hardware
        max_chans = 2
        rates_to_try = []
        if target_device is not None:
            try:
                with suppress_c_stderr():
                    dev_info = sd.query_devices(target_device, 'input')
                    max_chans = int(dev_info.get('max_input_channels', 2))
                    hw_sr = int(dev_info.get('default_samplerate', 44100))
                    if hw_sr > 0:
                        rates_to_try.append(hw_sr)
            except Exception:
                pass

        for preferred_sr in [44100, 48000, 16000, 22050, 8000]:
            if preferred_sr not in rates_to_try:
                rates_to_try.append(preferred_sr)

        channels_to_try = []
        if max_chans >= 1:
            channels_to_try.append(1)
        if max_chans >= 2:
            channels_to_try.append(2)
        if not channels_to_try:
            channels_to_try = [1, 2]

        speech_streak = 0
        def audio_callback(indata, frames, time_info, status):
            nonlocal detected, detected_name, speech_streak
            if status:
                pass
            
            # 1. Check openWakeWord neural model hits ('alexa', 'hey_jarvis', etc.)
            model_hit = self.predict_frame(indata)
            if model_hit:
                detected = True
                detected_name = model_hit
                return

        try:
            opened = False
            last_err = None

            # Attempt 1: Try resolved target_device with supported channels & sample rates
            for ch in channels_to_try:
                for sr in rates_to_try:
                    block_size = int(round(0.08 * sr))
                    try:
                        with suppress_c_stderr():
                            sd.check_input_settings(device=target_device, samplerate=sr, channels=ch, dtype='int16')
                            with sd.InputStream(
                                samplerate=sr,
                                blocksize=block_size,
                                channels=ch,
                                dtype='int16',
                                device=target_device,
                                callback=audio_callback
                            ):
                                opened = True
                                while not detected and not is_shutdown_requested():
                                    if timeout and (time.time() - start_time) > timeout:
                                        print("⏳ [WAKE WORD] Listening timed out.")
                                        return False
                                    sd.sleep(50)
                                if is_shutdown_requested():
                                    return False
                        break
                    except Exception as err:
                        last_err = err
                        continue
                if opened:
                    break

            # Attempt 2: Fallback to system default input device (device=None) if target_device failed
            if not opened and target_device is not None:
                for ch in [1, 2]:
                    for sr in [44100, 48000, 16000]:
                        block_size = int(round(0.08 * sr))
                        try:
                            with suppress_c_stderr():
                                sd.check_input_settings(device=None, samplerate=sr, channels=ch, dtype='int16')
                                with sd.InputStream(
                                    samplerate=sr,
                                    blocksize=block_size,
                                    channels=ch,
                                    dtype='int16',
                                    device=None,
                                    callback=audio_callback
                                ):
                                    opened = True
                                    while not detected:
                                        if timeout and (time.time() - start_time) > timeout:
                                            print("⏳ [WAKE WORD] Listening timed out.")
                                            return False
                                        sd.sleep(50)
                            break
                        except Exception as err:
                            last_err = err
                            continue
                    if opened:
                        break

            if not opened:
                print(f"❌ [WAKE WORD Error] Microphone stream failed: {last_err}", file=sys.stderr)
                return False

            print(f"⚡ [WAKE WORD DETECTED] Triggered phrase: '{detected_name or self.model_name}'!")
            try:
                from audio_io import play_beep_sound
                play_beep_sound()
            except Exception:
                pass

            self.oww_model.reset()  # Reset internal prediction buffers
            return True

        except KeyboardInterrupt:
            print("\n⏹️ [WAKE WORD] Stopped by user.")
            raise
        except Exception as err:
            print(f"❌ [WAKE WORD Error] Streaming error: {err}", file=sys.stderr)
            return False

class StopDetector:
    """Dedicated local openWakeWord detector for the 'stop' keyword."""
    def __init__(self, model_name="stop", threshold=None):
        target_key = model_name.lower().strip()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        custom_model_path = os.path.join(base_dir, f"{target_key}.onnx")
        if not os.path.exists(custom_model_path):
            raise FileNotFoundError(f"Stop keyword ONNX model not found: {custom_model_path}")

        if threshold is None:
            try:
                threshold = float(os.getenv("STOP_THRESHOLD", "0.55"))
            except Exception:
                threshold = 0.55
        self.threshold = max(0.05, min(0.95, threshold))
        self.model_name = target_key

        from openwakeword.model import Model
        with suppress_c_stderr():
            self.model = Model(wakeword_models=[custom_model_path], inference_framework="onnx")
        print(f"✅ [StopDetector] Loaded local ONNX interrupt model: {custom_model_path} (Threshold: {self.threshold})")

    def check_frame(self, indata, override_threshold=None):
        """Checks raw audio frame for 'stop' keyword. Returns True if detected."""
        if not self.model:
            return False
        thresh = override_threshold if override_threshold is not None else self.threshold

        if indata.ndim > 1 and indata.shape[1] > 1:
            frame = indata.mean(axis=1).astype(np.float32)
        else:
            frame = indata.flatten().astype(np.float32)

        if len(frame) != CHUNK_SIZE and len(frame) > 0:
            frame = signal.resample(frame, CHUNK_SIZE)

        frame = np.clip(frame * 1.5, -32768, 32767).astype(np.int16)
        pred = self.model.predict(frame)
        for name, score in pred.items():
            if score >= thresh:
                return True
        return False


if __name__ == "__main__":
    print("Testing openWakeWord local detector...")
    detector = WakeWordDetector(model_name="hey_jarvis", threshold=0.5)
    
    print("\nSay 'Hey Jarvis' into your microphone to test detection!")
    success = detector.listen_for_wakeword()
    if success:
        print("🎉 SUCCESS! Wake word detected accurately.")
    else:
        print("❌ Detection stopped or failed.")
