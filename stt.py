"""
stt.py - Speech-to-Text Module
Supports Groq Whisper API (whisper-large-v3-turbo) and OpenAI Whisper API.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

class SpeechToText:
    def __init__(self, groq_api_key=None, openai_api_key=None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        self.groq_client = None
        self.openai_client = None

        if self.groq_api_key:
            from groq import Groq
            self.groq_client = Groq(api_key=self.groq_api_key)

        if self.openai_api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.openai_api_key)

        if not self.groq_client and not self.openai_client:
            print("⚠️ No Groq/OpenAI API keys found. SpeechToText will rely on Google Free Web STT fallback.", file=sys.stderr)

    def transcribe(self, audio_filepath, prompt=None):
        """
        Transcribe audio file using Groq Whisper (free & ultra-fast) or OpenAI Whisper.
        """
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

        if os.path.getsize(audio_filepath) == 0:
            raise ValueError("Audio recording file is empty (0 bytes). Check microphone input.")

        start_time = time.time()
        
        # 1. Try Groq Whisper (Fastest & 100% Free)
        if self.groq_client:
            try:
                with open(audio_filepath, "rb") as audio_file:
                    params = {
                        "model": "whisper-large-v3-turbo",
                        "file": audio_file,
                        "language": "en"
                    }
                    if prompt:
                        params["prompt"] = prompt

                    response = self.groq_client.audio.transcriptions.create(**params)
                    elapsed = time.time() - start_time
                    transcript = response.text.strip()

                    print(f"⚡ [STT (Groq Whisper)] Transcribed in {elapsed:.2f}s: \"{transcript}\"")
                    return transcript
            except Exception as e:
                print(f"⚠️ Groq STT failed ({e}), attempting OpenAI fallback...")

        # 2. Try Google Free Web STT (No API key or quota required)
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(audio_filepath) as source:
                audio_data = r.record(source)
            transcript = r.recognize_google(audio_data)
            if transcript and transcript.strip():
                elapsed = time.time() - start_time
                print(f"⚡ [STT (Google Free Web STT)] Transcribed in {elapsed:.2f}s: \"{transcript.strip()}\"")
                return transcript.strip()
        except Exception as e:
            print(f"⚠️ Google Free STT failed ({e}), attempting OpenAI fallback...")

        # 3. OpenAI Whisper Fallback
        if self.openai_client:
            try:
                with open(audio_filepath, "rb") as audio_file:
                    params = {
                        "model": "whisper-1",
                        "file": audio_file,
                        "language": "en"
                    }
                    if prompt:
                        params["prompt"] = prompt

                    response = self.openai_client.audio.transcriptions.create(**params)
                    elapsed = time.time() - start_time
                    transcript = response.text.strip()

                    print(f"⚡ [STT (OpenAI Whisper)] Transcribed in {elapsed:.2f}s: \"{transcript}\"")
                    return transcript
            except Exception as e:
                print(f"⚠️ OpenAI Whisper failed ({e})...", file=sys.stderr)

        raise RuntimeError("Speech-to-Text failed on all providers.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stt.py <path_to_audio_file.wav>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Testing STT with file: {filepath}")
    stt_engine = SpeechToText()
    try:
        text = stt_engine.transcribe(filepath)
        print(f"\nFinal Result:\n{text}")
    except Exception as err:
        print(f"Test Failed: {err}")
