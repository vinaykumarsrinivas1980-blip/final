"""
llm.py - Large Language Model Module
Handles conversational reasoning using Groq API with short-term memory buffer.
"""

import os
import sys
import time
from dotenv import load_dotenv
from groq import Groq, GroqError

load_dotenv()

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
from datetime import datetime
from prompts import SYSTEM_PROMPT, get_canned_response

HISTORY_FILE = "conversation_history.json"

class LLMEngine:
    def __init__(self, api_key=None, model=None, system_prompt=None, max_memory_exchanges=5):
        """
        :param api_key: Groq API key.
        :param model: Groq model string (defaults to llama-3.3-70b-versatile).
        :param system_prompt: Guidance for the LLM behavior.
        :param max_memory_exchanges: Number of past user/assistant turns to retain.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY missing! Please set it in your .env file or pass it to LLMEngine()."
            )
        self.client = Groq(api_key=self.api_key)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_memory = max_memory_exchanges * 2  # 2 messages per exchange (user + assistant)
        self.history = self._load_history()

    def _load_history(self):
        """Loads saved conversation history from disk."""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"📖 Loaded {len(data)//2} past conversation turns from '{HISTORY_FILE}'.")
                        return data[-self.max_memory:]
            except Exception as e:
                print(f"⚠️ Could not load history file: {e}", file=sys.stderr)
        return []

    def _save_history(self):
        """Saves current conversation history to disk."""
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Could not save history file: {e}", file=sys.stderr)

    def _truncate_history(self):
        """Ensures conversation history does not exceed memory limit."""
        if len(self.history) > self.max_memory:
            self.history = self.history[-self.max_memory:]

    def clear_memory(self):
        """Clears short-term conversation context."""
        self.history = []
        self._save_history()


    def generate_response(self, user_text):
        """
        Generates LLM reply given user input text, maintaining conversation context.

        :param user_text: Text input from STT.
        :return: Generated assistant reply string.
        """
        if not user_text or not user_text.strip():
            return "I couldn't hear what you said. Could you please repeat that?"

        # 1. Fast local canned response check (0ms latency, 100% offline match)
        canned_reply = get_canned_response(user_text)
        if canned_reply:
            print(f"⚡ [Canned Response (0ms Local Match)]: \"{canned_reply}\"")
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": canned_reply})
            self._truncate_history()
            self._save_history()
            return canned_reply

        # Inject real-time system date and time into system prompt
        now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        dynamic_system_prompt = f"{self.system_prompt} Current system date and time: {now_str}."

        # Build full messages payload with system prompt + memory
        messages = [{"role": "system", "content": dynamic_system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})


        start_time = time.time()
        models_to_try = [self.model]
        for fallback in ["llama-3.1-8b-instant", "gemma2-9b-it", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error = None
        for m in models_to_try:
            try:
                completion = self.client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=60
                )

                reply = completion.choices[0].message.content.strip()
                elapsed = time.time() - start_time

                # Update memory history
                self.history.append({"role": "user", "content": user_text})
                self.history.append({"role": "assistant", "content": reply})
                self._truncate_history()
                self._save_history()

                print(f"⚡ [LLM ({m})] Generated in {elapsed:.2f}s: \"{reply}\"")
                return reply
            except Exception as e:
                print(f"⚠️ LLM model '{m}' failed ({e}), trying fallback model...", file=sys.stderr)
                last_error = e

        print(f"❌ [LLM Error] All models failed: {last_error}", file=sys.stderr)
        return "Sorry, I encountered an issue reaching my reasoning service."


if __name__ == "__main__":
    test_prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello! Who are you and what can you do?"
    print(f"Testing Groq LLM with prompt: \"{test_prompt}\"")
    
    llm = LLMEngine()
    response = llm.generate_response(test_prompt)
    print(f"\nFinal LLM Response:\n{response}")
