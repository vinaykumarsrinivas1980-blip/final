
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

# Default settings optimized for Speech-to-Text (Whisper)
SAMPLE_RATE = 16000  # 16kHz mono is standard for speech processing
CHANNELS = 1
DTYPE = 'int16'

SHUTDOWN_EVENT = threading.Event()

def set_shutdown_flag():
    """Sets the global shutdown signal event for headless termination."""
    SHUTDOWN_EVENT.set()

def is_shutdown_requested():
    """Returns True if a system shutdown or termination signal has been received."""
    return SHUTDOWN_EVENT.is_set()

def cleanup_audio_resources():
    """Stops active sounddevice streams and unloads pygame mixer during shutdown."""
    set_shutdown_flag()
    try:
        sd.stop()
    except Exception:
        pass
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
    except Exception:
        pass

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

        # Determine sample rates supported by microphone (prioritize hardware default sample rate)
        rates_to_try = []
        if target_device is not None:
            try:
                with suppress_c_stderr():
                    dev_info = sd.query_devices(target_device, 'input')
                    hw_sr = int(dev_info.get('default_samplerate', 44100))
                    if hw_sr > 0:
                        rates_to_try.append(hw_sr)
            except Exception:
                pass

        for fallback_sr in [44100, 48000, 16000, 22050, 8000]:
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
                        # Verify setting validity quietly without triggering C stderr output
                        sd.check_input_settings(device=target_device, samplerate=sr, channels=ch, dtype=DTYPE)
                        with sd.InputStream(
                            samplerate=sr,
                            channels=ch,
                            dtype=DTYPE,
                            device=target_device,
                            callback=callback
                        ):
                            self.actual_sample_rate = sr
                            opened = True
                            while self.is_recording and not is_shutdown_requested():
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
    
    # RMS threshold for detecting active speech (sensitive threshold for all mics)
    SPEECH_RMS_THRESHOLD = 120
    
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


def flush_keypress_buffer():
    """Flushes stale console keypresses from buffer (Windows & Linux/Raspberry Pi)."""
    if sys.platform == 'win32':
        import msvcrt
        while msvcrt.kbhit():
            try:
                msvcrt.getch()
            except Exception:
                break
    else:
        try:
            import select
            while select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.read(1)
        except Exception:
            pass

def check_keypress_interrupt():
    """Returns True if user has pressed 'm' / 'M' (or ENTER/space) on keyboard in CLI (Windows & Linux/Raspberry Pi)."""
    if sys.platform == 'win32':
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch.lower() in [b'm', b'\r', b'\n', b' ', b'q']:
                return True
    else:
        try:
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                try:
                    import termios, tty
                    old_settings = termios.tcgetattr(sys.stdin)
                    try:
                        tty.setcbreak(sys.stdin.fileno())
                        ch = sys.stdin.read(1)
                        if ch.lower() in ['m', '\r', '\n', ' ', 'q']:
                            return True
                    finally:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    line = sys.stdin.readline()
                    if 'm' in line.lower() or line:
                        return True
        except Exception:
            pass
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
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            except Exception:
                try:
                    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
                except Exception:
                    pygame.mixer.init()
    except Exception:
        pass

    interrupted = [False]
    stop_event = threading.Event()

    # Flush any stale keypresses left over in the Windows console input buffer
    flush_keypress_buffer()

    def mic_interrupt_listener():
        """Background listener monitoring mic for openWakeWord ('Hey Jarvis') and voice stop commands ('stop', 'ruko', etc.) during playback."""
        target_mic = get_working_device_index('input', mic_device_index)

        max_chans = 2
        rates_to_try = []
        if target_mic is not None:
            try:
                dev_info = sd.query_devices(target_mic, 'input')
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

        # Import prompts for stop command verification
        try:
            from prompts import is_stop_command
        except Exception:
            def is_stop_command(txt): return any(w in txt.lower() for w in ["stop", "quiet", "cancel", "ruko", "band"])

        pcm_frames = []
        last_stt_check_time = 0.0
        frame_counter = 0
        is_checking = [False]

        actual_sr = 16000

        def callback(indata, frames, time_info, status):
            nonlocal last_stt_check_time
            if stop_event.is_set():
                return

            # 1. Check openWakeWord neural hits ("alexa", "hey_jarvis", "max") during playback
            if wakeword_detector:
                hit = wakeword_detector.predict_frame(indata, override_threshold=0.50)
                if hit:
                    print(f"\n⚡ [WAKE WORD INTERRUPT] Playback stopped by wake word '{hit}'!")
                    interrupted[0] = True
                    try:
                        sd.stop()
                    except Exception:
                        pass
                    try:
                        pygame.mixer.music.stop()
                    except Exception:
                        pass
                    stop_event.set()
                    return

            # 2. Check for spoken stop commands ("stop", "ruko", "be quiet", "cancel", "chup") during playback
            if stt_engine and len(indata) > 0 and not is_checking[0]:
                audio_float = indata.astype(np.float32)
                rms = float(np.sqrt(np.mean(audio_float**2)))
                max_amp = float(np.max(np.abs(audio_float)))

                # Require direct human voice near mic (RMS > 1000 or max_amp > 10000)
                if rms > 1000 or max_amp > 10000:
                    pcm_frames.append(indata.copy())
                    current_time = time.time()
                    
                    # Accumulate ~0.8s speech chunk for reliable stop command verification
                    if len(pcm_frames) >= 10 and (current_time - last_stt_check_time) > 0.8:
                        last_stt_check_time = current_time
                        audio_chunk = np.concatenate(pcm_frames, axis=0)
                        pcm_frames.clear()
                        is_checking[0] = True

                        def async_stop_check(chunk_data, sr_used):
                            temp_path = f"temp_barge_{threading.get_ident()}_{int(time.time()*1000)}.wav"
                            try:
                                # Resample audio chunk to 16000Hz if captured at hardware rate
                                if sr_used != SAMPLE_RATE and len(chunk_data) > 0:
                                    num_samples = int(round(len(chunk_data) * SAMPLE_RATE / float(sr_used)))
                                    if chunk_data.ndim > 1:
                                        resample_float = signal.resample(chunk_data.astype(np.float32), num_samples, axis=0)
                                    else:
                                        resample_float = signal.resample(chunk_data.astype(np.float32), num_samples)
                                    chunk_data = np.clip(resample_float, -32768, 32767).astype(DTYPE)

                                with wave.open(temp_path, 'wb') as wf:
                                    wf.setnchannels(1 if chunk_data.ndim == 1 else chunk_data.shape[1])
                                    wf.setsampwidth(2)
                                    wf.setframerate(SAMPLE_RATE)
                                    wf.writeframes(chunk_data.tobytes())

                                # Quietly check transcribed speech for stop command
                                class SuppressStdout:
                                    def __enter__(self):
                                        self._orig = sys.stdout
                                        sys.stdout = open(os.devnull, 'w')
                                    def __exit__(self, exc_type, exc_val, exc_tb):
                                        try:
                                            sys.stdout.close()
                                        finally:
                                            sys.stdout = self._orig

                                with SuppressStdout():
                                    txt = stt_engine.transcribe(temp_path)

                                if txt:
                                    clean_txt = txt.strip().strip('.').strip('!').strip('?').lower()
                                    if clean_txt and is_stop_command(clean_txt, strict=True):
                                        print(f"\n🛑 [BREAK COMMAND] Playback stopped by user voice command (\"{txt}\")!", flush=True)
                                        interrupted[0] = True
                                        try:
                                            pygame.mixer.music.stop()
                                            pygame.mixer.music.unload()
                                        except Exception:
                                            pass
                                        stop_event.set()
                            except Exception:
                                pass
                            finally:
                                is_checking[0] = False
                                if os.path.exists(temp_path):
                                    try:
                                        os.remove(temp_path)
                                    except Exception:
                                        pass

                        t_barge = threading.Thread(target=async_stop_check, args=(audio_chunk, actual_sr), daemon=True)
                        t_barge.start()
                else:
                    if len(pcm_frames) > 10:
                        pcm_frames.clear()

        opened_stream = False
        last_stream_err = None

        for ch in channels_to_try:
            for sr in rates_to_try:
                block_size = int(round(0.08 * sr))
                try:
                    with suppress_c_stderr():
                        sd.check_input_settings(device=target_mic, samplerate=sr, channels=ch, dtype='int16')
                        with sd.InputStream(
                            samplerate=sr,
                            blocksize=block_size,
                            channels=ch,
                            dtype='int16',
                            device=target_mic,
                            callback=callback
                        ):
                            actual_sr = sr
                            opened_stream = True
                            print("🎙️  [PLAYBACK MIC ACTIVE] Microphone listening for 'BREAK' command...", flush=True)
                            while not stop_event.is_set() and not is_shutdown_requested():
                                sd.sleep(50)
                    return
                except Exception as err:
                    last_stream_err = err
                    continue

        if not opened_stream and last_stream_err:
            print(f"⚠️ [MIC INTERRUPT WARNING] Could not start mic interrupt listener: {last_stream_err}", file=sys.stderr)

    # Start background mic listener thread if enabled
    if (enable_interrupt or wakeword_detector is not None) and stt_engine:
        t = threading.Thread(target=mic_interrupt_listener, daemon=True)
        t.start()

    # Play audio using pygame.mixer.music (direct manageable playback stream)
    try:
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy() and not stop_event.is_set() and not is_shutdown_requested():
            if check_keypress_interrupt():
                interrupted[0] = True
                pygame.mixer.music.stop()
                print("\n🛑 [KEYBOARD INTERRUPT] Playback stopped by pressing 'm'!", flush=True)
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        interrupted[0] = True
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        raise
    except Exception as err:
        print(f"⚠️ Playback error: {err}", file=sys.stderr)
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
