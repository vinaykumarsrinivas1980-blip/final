
"""
audio_io.py - Audio Hardware & Input/Output Module
Handles device enumeration, microphone recording, and speaker playback.
"""

import os
import sys
import time
import wave
import threading
import numpy as np

# Hide Pygame community support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Suppress Linux ALSA / PortAudio C-library error messages globally
if sys.platform != 'win32':
    try:
        from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def _alsa_no_error_handler(filename, line, function, err, fmt):
            pass
        _c_alsa_handler = ERROR_HANDLER_FUNC(_alsa_no_error_handler)
        for alsa_lib in ['libasound.so.2', 'libasound.so']:
            try:
                asound = cdll.LoadLibrary(alsa_lib)
                asound.snd_lib_error_set_handler(_c_alsa_handler)
                break
            except Exception:
                pass
    except Exception:
        pass

import sounddevice as sd
from scipy import signal
from scipy.io import wavfile

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Default settings optimized for Speech-to-Text (Whisper)
SAMPLE_RATE = 16000  # 16kHz mono is standard for speech processing
CHANNELS = 1
DTYPE = 'int16'

class suppress_c_stderr:
    """Context manager to suppress low-level ALSA / PortAudio C-library stderr warnings on Linux."""
    def __enter__(self):
        if sys.platform != 'win32':
            try:
                self.null_fd = os.open(os.devnull, os.O_WRONLY)
                self.old_stderr = os.dup(2)
                os.dup2(self.null_fd, 2)
            except Exception:
                self.null_fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.platform != 'win32' and getattr(self, 'null_fd', None) is not None:
            try:
                os.dup2(self.old_stderr, 2)
                os.close(self.old_stderr)
                os.close(self.null_fd)
            except Exception:
                pass


def list_audio_devices():
    """Prints all available audio input and output devices."""
    print("\n" + "=" * 60)
    print(" AUDIO DEVICES AVAILABLE ")
    print("=" * 60)
    devices = sd.query_devices()
    input_devices = []
    output_devices = []

    for index, dev in enumerate(devices):
        max_in = dev['max_input_channels']
        max_out = dev['max_output_channels']
        name = dev['name']
        sr = int(dev['default_samplerate'])

        if max_in > 0:
            input_devices.append((index, name, max_in, sr))
        if max_out > 0:
            output_devices.append((index, name, max_out, sr))

    print("\n--- INPUT DEVICES (Microphones) ---")
    for idx, name, chans, sr in input_devices:
        is_usb = "usb" in name.lower()
        tag = " [RECOMMENDED USB]" if is_usb else ""
        print(f"  [{idx}] {name} (In Channels: {chans}, Default SR: {sr}Hz){tag}")

    print("\n--- OUTPUT DEVICES (Speakers) ---")
    for idx, name, chans, sr in output_devices:
        is_usb = "usb" in name.lower()
        tag = " [RECOMMENDED USB]" if is_usb else ""
        print(f"  [{idx}] {name} (Out Channels: {chans}, Default SR: {sr}Hz){tag}")

    print("=" * 60 + "\n")
    return input_devices, output_devices

def find_usb_device(kind='input'):
    """
    Auto-detects USB Microphone or USB Speaker on Raspberry Pi & Windows.
    Searches device names for USB & ALSA keywords ('usb', 'pnp', 'c-media', 'soundcard', 'headset', 'mic', 'sysdefault', 'plughw').
    """
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    ch_key = 'max_input_channels' if kind == 'input' else 'max_output_channels'

    # Priority 1: Exact 'usb' match in device name
    for index, dev in enumerate(devices):
        name = dev['name'].lower()
        if 'usb' in name and dev.get(ch_key, 0) > 0:
            return index

    # Priority 2: Common Raspberry Pi USB Audio soundcard chipsets & ALSA descriptors
    rpi_keywords = ['pnp', 'c-media', 'soundcard', 'usb-audio', 'headset', 'mic', 'microphone', 'sysdefault', 'plughw']
    for index, dev in enumerate(devices):
        name = dev['name'].lower()
        if any(kw in name for kw in rpi_keywords) and dev.get(ch_key, 0) > 0:
            return index

    return None


def get_working_device_index(kind='input', specified_idx=None):
    """
    Resolves a valid audio device index for recording ('input') or playback ('output').
    Priority:
    1. Explicit user-specified index (if valid and has channels > 0)
    2. Auto-detected USB device via find_usb_device()
    3. OS default device from sounddevice settings (if valid >= 0 and has channels > 0)
    4. First available hardware device with matching channels > 0 from system enumeration
    """
    ch_key = 'max_input_channels' if kind == 'input' else 'max_output_channels'

    # 1. Specified index validation
    if specified_idx is not None:
        try:
            info = sd.query_devices(specified_idx)
            if info.get(ch_key, 0) > 0:
                return specified_idx
        except Exception:
            pass

    # 2. Auto-detect USB hardware device
    usb_idx = find_usb_device(kind)
    if usb_idx is not None:
        return usb_idx

    # 3. Sounddevice OS default device
    try:
        def_idx = sd.default.device[0 if kind == 'input' else 1]
        if def_idx is not None and def_idx >= 0:
            info = sd.query_devices(def_idx)
            if info.get(ch_key, 0) > 0:
                return def_idx
    except Exception:
        pass

    # 4. Enumerate all devices and pick first valid matching device
    try:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev.get(ch_key, 0) > 0:
                return idx
    except Exception:
        pass

    return None


class AudioRecorder:
    def __init__(self, device_index=None, sample_rate=SAMPLE_RATE):
        self.device_index = get_working_device_index('input', device_index)
        self.sample_rate = sample_rate
        self.actual_sample_rate = sample_rate
        self.is_recording = False
        self.frames = []
        self._thread = None
        self.error_msg = None

    def _record_loop(self):
        """Background thread function to capture audio frames with sample rate & channel fallback for Raspberry Pi."""
        def callback(indata, frames, time_info, status):
            if status:
                pass  # Suppress ALSA underflow/overflow status warnings on Pi
            if self.is_recording:
                # If recorded in stereo (2 channels), downmix to mono (1 channel) for STT
                if indata.ndim > 1 and indata.shape[1] > 1:
                    mono_chunk = indata.mean(axis=1, keepdims=True).astype(DTYPE)
                    self.frames.append(mono_chunk)
                else:
                    self.frames.append(indata.copy())

        target_device = self.device_index if self.device_index is not None else get_working_device_index('input')

        # Determine sample rates supported by microphone
        rates_to_try = [self.sample_rate]
        if target_device is not None:
            try:
                dev_info = sd.query_devices(target_device, 'input')
                hw_sr = int(dev_info.get('default_samplerate', 44100))
                if hw_sr not in rates_to_try:
                    rates_to_try.append(hw_sr)
            except Exception:
                pass

        for fallback_sr in [44100, 48000, 22050, 8000]:
            if fallback_sr not in rates_to_try:
                rates_to_try.append(fallback_sr)

        # Channel configurations: 1 channel (mono) first, fallback to 2 channels (stereo) for Pi USB soundcards
        channels_to_try = [CHANNELS]
        if 2 not in channels_to_try:
            channels_to_try.append(2)

        opened = False
        for ch in channels_to_try:
            for sr in rates_to_try:
                try:
                    with suppress_c_stderr():
                        with sd.InputStream(
                            samplerate=sr,
                            channels=ch,
                            dtype=DTYPE,
                            device=target_device,
                            callback=callback
                        ):
                            self.actual_sample_rate = sr
                            opened = True
                            while self.is_recording:
                                sd.sleep(50)
                    break
                except Exception as err:
                    self.error_msg = str(err)
                    continue
            if opened:
                break

        if not opened and self.error_msg:
            print(f"❌ Microphone recording stream error: {self.error_msg}", file=sys.stderr)

    def start_recording(self):
        """Starts recording audio in a background thread."""
        self.is_recording = True
        self.frames = []
        self.error_msg = None
        self._thread = threading.Thread(target=self._record_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop_recording(self, output_filepath="temp_input.wav"):
        """Stops recording and saves accumulated audio to a 16-bit WAV file."""
        self.is_recording = False
        if self._thread:
            self._thread.join()

        if not self.frames:
            print("⚠️ No audio frames captured.", file=sys.stderr)
            return output_filepath, 0.0

        # Concatenate recorded audio frames
        audio_data = np.concatenate(self.frames, axis=0)
        duration = len(audio_data) / float(self.actual_sample_rate)

        # Resample to standard 16kHz if captured at a hardware sample rate != 16000Hz
        if self.actual_sample_rate != SAMPLE_RATE:
            num_samples = int(round(len(audio_data) * SAMPLE_RATE / float(self.actual_sample_rate)))
            if audio_data.ndim > 1:
                resampled_channels = []
                for c in range(audio_data.shape[1]):
                    resampled_c = signal.resample(audio_data[:, c].astype(np.float32), num_samples)
                    resampled_channels.append(resampled_c)
                audio_data = np.column_stack(resampled_channels).astype(DTYPE)
            else:
                audio_data = signal.resample(audio_data.astype(np.float32), num_samples).astype(DTYPE)

        # Write to WAV file
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        with wave.open(output_filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS if audio_data.ndim == 1 else audio_data.shape[1])
            wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        return output_filepath, duration


def record_push_to_talk(output_filepath="temp_input.wav", device_index=None):
    """
    Push-to-Talk interactive CLI mode.
    Press ENTER to start recording, press ENTER again to stop recording.
    """
    recorder = AudioRecorder(device_index=device_index)
    recorder.start_recording()
    
    print("🎙️  [LISTENING] Recording... Press [ENTER] to stop.")
    input()
    
    print("⏹️  [STOPPED] Finalizing audio file...")
    filepath, duration = recorder.stop_recording(output_filepath)
    print(f"💾 Audio saved: {filepath} ({duration:.2f} seconds)")
    return filepath, duration

def record_smart_audio(output_filepath="temp_input.wav", device_index=None, silence_timeout=1.0, max_duration=5.0):
    """
    Dynamic VAD Audio Recording.
    Automatically starts recording, detects speech onset, and stops as soon as
    1.0 second of silence is detected (or max_duration safety cap is reached).
    """
    recorder = AudioRecorder(device_index=device_index)
    recorder.start_recording()
    
    print("🎙️  [LISTENING] Speak your command now...")
    
    start_time = time.time()
    last_speech_time = time.time()
    speech_started = False
    
    # RMS threshold for detecting active speech
    SPEECH_RMS_THRESHOLD = 300
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            print(f"⏳ [STOPPED] Max recording duration reached ({max_duration:.0f}s).")
            break
            
        if recorder.frames:
            # Check recent audio frame volume (RMS)
            latest_chunk = recorder.frames[-1]
            if len(latest_chunk) > 0:
                rms = np.sqrt(np.mean(latest_chunk.astype(np.float32)**2))
                
                if rms > SPEECH_RMS_THRESHOLD:
                    if not speech_started:
                        speech_started = True
                    last_speech_time = time.time()
                elif speech_started:
                    # Check silence duration after speech started
                    silence_duration = time.time() - last_speech_time
                    if silence_duration >= silence_timeout:
                        print(f"⏹️  [STOPPED] Silence detected ({silence_duration:.1f}s). Processing...")
                        break
        
        sd.sleep(50)
        
    filepath, duration = recorder.stop_recording(output_filepath)
    print(f"💾 Audio captured: {filepath} ({duration:.2f} seconds)")
    return filepath, duration


def record_timed_audio(output_filepath="temp_input.wav", duration_seconds=5.0, device_index=None):
    """
    Records audio for a fixed duration automatically after wake-word detection.
    """
    print(f"🎙️  [LISTENING] Recording command for {duration_seconds:.1f} seconds...")
    recorder = AudioRecorder(device_index=device_index)
    recorder.start_recording()
    time.sleep(duration_seconds)
    print("⏹️  [STOPPED] Processing recorded audio...")
    filepath, duration = recorder.stop_recording(output_filepath)
    return filepath, duration



def boost_alsa_system_volume():
    """Unmutes and boosts ALSA hardware volume sliders to 100% on Raspberry Pi / Linux."""
    if sys.platform != 'win32':
        for control in ['PCM', 'Master', 'Headphone', 'Speaker']:
            try:
                os.system(f"amixer sset '{control}' 100% unmute >/dev/null 2>&1")
            except Exception:
                pass


def normalize_and_boost_audio(data, gain_db=6.0):
    """
    Peak-normalizes audio amplitude and applies digital gain boost for loud speaker playback.
    """
    if data is None or len(data) == 0:
        return data

    try:
        is_int16 = (data.dtype == np.int16)
        float_data = data.astype(np.float32)
        max_val = np.max(np.abs(float_data))
        
        if max_val > 0:
            target_max = 32000.0 if is_int16 else 0.95
            scale = (target_max / max_val) * (10.0 ** (gain_db / 20.0))
            float_data = float_data * scale

        if is_int16:
            np.clip(float_data, -32768, 32767, out=float_data)
            return float_data.astype(np.int16)
        else:
            np.clip(float_data, -1.0, 1.0, out=float_data)
            return float_data
    except Exception:
        return data


def check_keypress_interrupt():
    """Returns True if user has pressed ENTER or key in Windows CLI."""
    if sys.platform == 'win32':
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in [b'\r', b'\n', b' ']:
                return True
    return False

def play_audio(filepath, device_index=None, enable_interrupt=True, mic_device_index=None, wakeword_detector=None, stt_engine=None):
    """
    Plays an MP3 or WAV audio file through the specified speaker device index.
    Supports 100% reliable voice barge-in interruption via openWakeWord ('Hey Jarvis' / 'Alexa'), voice stop commands ('stop'), and keypresses.
    Returns True if playback was interrupted, False if completed normally.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found for playback: {filepath}")

    import pygame
    try:
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            except Exception:
                pygame.mixer.init()
    except Exception:
        pass

    interrupted = [False]
    stop_event = threading.Event()

    def mic_interrupt_listener():
        """Background listener monitoring mic for openWakeWord ('Hey Jarvis') and voice stop commands ('stop', 'ruko', etc.) during playback."""
        target_mic = get_working_device_index('input', mic_device_index)

        max_chans = 2
        rates_to_try = [16000]
        if target_mic is not None:
            try:
                dev_info = sd.query_devices(target_mic, 'input')
                max_chans = int(dev_info.get('max_input_channels', 2))
                hw_sr = int(dev_info.get('default_samplerate', 44100))
                if hw_sr not in rates_to_try:
                    rates_to_try.append(hw_sr)
            except Exception:
                pass

        for fallback_sr in [44100, 48000, 22050, 8000]:
            if fallback_sr not in rates_to_try:
                rates_to_try.append(fallback_sr)

        channels_to_try = []
        if max_chans >= 1:
            channels_to_try.append(1)
        if max_chans >= 2:
            channels_to_try.append(2)
        if not channels_to_try:
            channels_to_try = [1, 2]

        # Import prompts for stop command verification
        try:
            from prompts import is_stop_command
        except Exception:
            def is_stop_command(txt): return any(w in txt.lower() for w in ["stop", "quiet", "cancel", "ruko", "band"])

        pcm_frames = []
        last_stt_check_time = 0.0
        frame_counter = 0

        actual_sr = 16000

        def callback(indata, frames, time_info, status):
            nonlocal last_stt_check_time
            if stop_event.is_set():
                return

            # Listen EXCLUSIVELY for spoken STOP command ("stop") during playback
            if stt_engine and len(indata) > 0:
                rms = float(np.sqrt(np.mean(indata.astype(np.float32)**2)))
                # Detect normal spoken voice near mic (RMS energy > 350)
                if rms > 350:
                    pcm_frames.append(indata.copy())
                    current_time = time.time()
                    
                    # Process accumulated ~0.5s speech chunk for rapid response
                    if len(pcm_frames) >= 6 and (current_time - last_stt_check_time) > 0.5:
                        last_stt_check_time = current_time
                        audio_chunk = np.concatenate(pcm_frames, axis=0)
                        pcm_frames.clear()

                        def async_stop_check(chunk_data, sr_used):
                            try:
                                # Resample audio chunk to 16000Hz if captured at hardware rate (e.g. 44100Hz)
                                if sr_used != SAMPLE_RATE and len(chunk_data) > 0:
                                    num_samples = int(round(len(chunk_data) * SAMPLE_RATE / float(sr_used)))
                                    if chunk_data.ndim > 1:
                                        chunk_data = signal.resample(chunk_data.astype(np.float32), num_samples, axis=0).astype(DTYPE)
                                    else:
                                        chunk_data = signal.resample(chunk_data.astype(np.float32), num_samples).astype(DTYPE)

                                temp_path = "temp_barge.wav"
                                with wave.open(temp_path, 'wb') as wf:
                                    wf.setnchannels(1 if chunk_data.ndim == 1 else chunk_data.shape[1])
                                    wf.setsampwidth(2)
                                    wf.setframerate(SAMPLE_RATE)
                                    wf.writeframes(chunk_data.tobytes())

                                # Quietly check transcribed speech for stop command
                                old_stdout = sys.stdout
                                try:
                                    sys.stdout = open(os.devnull, 'w')
                                    txt = stt_engine.transcribe(temp_path)
                                finally:
                                    sys.stdout.close()
                                    sys.stdout = old_stdout

                                import re
                                clean_txt = txt.lower().strip() if txt else ""
                                clean_words = set(re.sub(r'[^\w\s]', '', clean_txt).split())
                                if clean_txt and ("stop" in clean_words or "stop" in clean_txt):
                                    print(f"\n🛑 [STOP COMMAND] Playback stopped by user voice command (\"stop\")!")
                                    interrupted[0] = True
                                    try:
                                        pygame.mixer.music.stop()
                                    except Exception:
                                        pass
                                    stop_event.set()
                            except Exception:
                                pass

                        t_barge = threading.Thread(target=async_stop_check, args=(audio_chunk, actual_sr), daemon=True)
                        t_barge.start()
                else:
                    if len(pcm_frames) > 20:
                        pcm_frames.clear()

        for ch in channels_to_try:
            for sr in rates_to_try:
                block_size = int(round(0.08 * sr))
                try:
                    with suppress_c_stderr():
                        with sd.InputStream(
                            samplerate=sr,
                            blocksize=block_size,
                            channels=ch,
                            dtype='int16',
                            device=target_mic,
                            callback=callback
                        ):
                            actual_sr = sr
                            while not stop_event.is_set():
                                sd.sleep(50)
                    return
                except Exception:
                    continue

    if enable_interrupt or wakeword_detector is not None:
        t = threading.Thread(target=mic_interrupt_listener, daemon=True)
        t.start()

    try:
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            if check_keypress_interrupt():
                interrupted[0] = True
                pygame.mixer.music.stop()
                print("\n🛑 [KEYPRESS INTERRUPT] Playback stopped by user keypress!")
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted[0] = True
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
    except Exception as e:
        # Fallback to system CLI players on Linux / Raspberry Pi if pygame encounters issue
        if sys.platform != 'win32':
            for player_cmd in [f"mpg123 -q '{filepath}'", f"ffplay -nodisp -autoexit -loglevel quiet '{filepath}'", f"mpv --no-terminal '{filepath}'"]:
                ret = os.system(player_cmd)
                if ret == 0:
                    break
        else:
            try:
                sr, data = wavfile.read(filepath) if filepath.endswith('.wav') else (None, None)
                if data is not None:
                    sd.play(data, sr)
                    sd.wait()
            except Exception:
                pass
    finally:
        stop_event.set()
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

    return interrupted[0]





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio IO Diagnostic Utility")
    parser.add_argument("--list", action="store_true", help="List audio devices")
    parser.add_argument("--test-mic", action="store_true", help="Test mic recording & playback")
    args = parser.parse_args()

    if args.list or len(sys.argv) == 1:
        list_audio_devices()
        
    if args.test_mic:

        list_audio_devices()
        mic_idx = find_usb_device('input')
        spk_idx = find_usb_device('output')
        print(f"Auto-selected USB Mic index: {mic_idx}, USB Speaker index: {spk_idx}")
        wav_path, dur = record_push_to_talk("test_recording.wav", device_index=mic_idx)
        print(f"Playing back test recording through speaker index {spk_idx}...")
        play_audio(wav_path, device_index=spk_idx)
        print("✅ Mic & Speaker test completed successfully.")
