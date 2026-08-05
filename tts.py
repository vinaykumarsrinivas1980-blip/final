"""
tts.py - Text-to-Speech Module
Supports Edge-TTS (free neural voices), gTTS, and OpenAI TTS API.
"""

import os
import sys
import time
import asyncio
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Preset voice shortcuts for convenience
VOICE_ALIASES = {
    "alloy": "en-IN-NeerjaNeural",
    "echo": "en-IN-PrabhatNeural",
    "fable": "en-IN-NeerjaExpressiveNeural",
    "onyx": "en-IN-PrabhatNeural",
    "nova": "en-IN-NeerjaNeural",
    "shimmer": "en-IN-NeerjaNeural",
    "indian_female": "en-IN-NeerjaNeural",
    "indian_male": "en-IN-PrabhatNeural",
    "indian_expressive": "en-IN-NeerjaExpressiveNeural",
    "hindi_female": "hi-IN-SwaraNeural",
    "hindi_male": "hi-IN-MadhurNeural",
    "us_female": "en-US-AvaNeural",
    "us_male": "en-US-BrianNeural",
    "uk_female": "en-GB-SoniaNeural",
    "uk_male": "en-GB-RyanNeural",
}

class TextToSpeech:
    def __init__(self, voice=None):
        selected_voice = voice or os.getenv("TTS_VOICE") or "en-IN-NeerjaNeural"
        # Resolve voice shortcut alias if provided
        self.voice = VOICE_ALIASES.get(selected_voice.lower(), selected_voice)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def _edge_tts_sync(self, text, output_filepath):
        """Asynchronously run edge-tts to generate high quality speech with volume boost."""
        import edge_tts
        async def _speak(v):
            communicate = edge_tts.Communicate(text, v, volume="+50%")
            await communicate.save(output_filepath)
        
        def _run_async(coro):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(asyncio.run, coro).result()
            else:
                return asyncio.run(coro)

        try:
            _run_async(_speak(self.voice))
        except Exception:
            # Fallback to standard Indian voice if custom voice name is unsupported
            fallback_voice = "en-IN-NeerjaNeural"
            _run_async(_speak(fallback_voice))

    def synthesize(self, text, output_filepath="temp_response.mp3"):
        """
        Synthesizes spoken audio from text input using Edge-TTS (free neural) or gTTS/OpenAI.
        """
        if not text or not text.strip():
            raise ValueError("Cannot synthesize speech from empty text.")

        start_time = time.time()

        # 1. Try Edge-TTS (Free, Ultra-realistic Neural Voice)
        try:
            self._edge_tts_sync(text, output_filepath)
            elapsed = time.time() - start_time
            file_size_kb = os.path.getsize(output_filepath) / 1024.0
            print(f"⚡ [TTS (Edge Neural: {self.voice})] Audio synthesized in {elapsed:.2f}s ({file_size_kb:.1f} KB -> {output_filepath})")
            return output_filepath
        except Exception as e:
            print(f"⚠️ Edge-TTS failed ({e}), trying gTTS fallback...")

        # 2. Try gTTS (Google Translate TTS - Free)
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='en')
            tts.save(output_filepath)
            elapsed = time.time() - start_time
            print(f"⚡ [TTS (gTTS)] Audio synthesized in {elapsed:.2f}s")
            return output_filepath
        except Exception as e:
            print(f"⚠️ gTTS failed ({e}), trying OpenAI TTS fallback...")

        # 3. Try pyttsx3 (100% Offline Local TTS - Free, no internet or API key required)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('volume', 1.0)
            wav_out = output_filepath if output_filepath.endswith('.wav') else output_filepath.rsplit('.', 1)[0] + '.wav'
            engine.save_to_file(text, wav_out)
            engine.runAndWait()
            if os.path.exists(wav_out) and os.path.getsize(wav_out) > 0:
                elapsed = time.time() - start_time
                print(f"⚡ [TTS (pyttsx3 Offline)] Audio synthesized in {elapsed:.2f}s -> {wav_out}")
                return wav_out
        except Exception as e:
            print(f"⚠️ pyttsx3 offline TTS failed ({e}), trying OpenAI TTS fallback...")

        # 4. Try OpenAI TTS (if credits available)
        if self.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text
                )
                response.stream_to_file(output_filepath)
                elapsed = time.time() - start_time
                print(f"⚡ [TTS (OpenAI)] Audio synthesized in {elapsed:.2f}s")
                return output_filepath
            except Exception as e:
                print(f"⚠️ OpenAI TTS failed ({e}), trying system espeak fallback...")

        # 5. Try system CLI fallback (espeak at max volume -a 200 on Linux / Raspberry Pi)
        if sys.platform != 'win32':
            try:
                wav_out = output_filepath.rsplit('.', 1)[0] + '.wav'
                ret = os.system(f"espeak -a 200 -v en \"{text}\" -w \"{wav_out}\" 2>/dev/null")
                if ret == 0 and os.path.exists(wav_out) and os.path.getsize(wav_out) > 0:
                    elapsed = time.time() - start_time
                    print(f"⚡ [TTS (espeak CLI)] Audio synthesized in {elapsed:.2f}s -> {wav_out}")
                    return wav_out
            except Exception:
                pass

        raise RuntimeError("Text-to-Speech synthesis failed on all engines.")


if __name__ == "__main__":
    sample_text = sys.argv[1] if len(sys.argv) > 1 else "Hello! This is a test of free neural text to speech."
    print(f"Testing TTS with text: \"{sample_text}\"")

    tts = TextToSpeech()
    output_file = tts.synthesize(sample_text, "test_output.mp3")
    print(f"✅ Audio file created: {output_file}")
