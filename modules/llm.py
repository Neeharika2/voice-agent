import google.generativeai as genai
import sys
import os
import time
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

genai.configure(api_key=config.GEMINI_API_KEY)

# System prompt optimized for friendly voice responses
VOICE_SYSTEM_PROMPT = """You are a warm, friendly voice assistant having a casual conversation.

IMPORTANT RULES:
- Keep responses SHORT and CONVERSATIONAL (1-3 sentences max)
- Talk like a friend would - use casual, natural language
- NEVER say "As an AI" or "I don't experience" - just answer naturally
- Be warm, supportive, and engaging
- Ask follow-up questions to keep conversation going
- If someone asks "do you know what happened" or "guess what" - show excitement and curiosity!"""

VOICE_EMOTIONS = ("neutral", "happy", "excited", "calm", "sad", "serious", "whisper", "curious")


class LLM:
    """
    Optimized LLM wrapper for Gemini with streaming and low-latency voice responses.
    """

    def __init__(self, model_name=None):
        self.model_name = model_name or getattr(config, 'LLM_MODEL', 'gemini-2.0-flash')
        self.max_tokens = getattr(config, 'LLM_MAX_TOKENS', 512)
        self._system_prompt = VOICE_SYSTEM_PROMPT

        # Generation config for faster responses
        self.gen_config = genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=self.max_tokens,
            top_p=0.9,
        )

        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config=self.gen_config
        )

        self.chat_session = self.model.start_chat(history=[])
        print(f"LLM initialized: {self.model_name}")

    def generate_stream(self, text, verbose=True):
        """
        Stream LLM response with latency tracking.

        Yields: (chunk_text, is_done, stats_dict)
        """
        stats = {
            'start_time': None,
            'first_token_time': None,
            'end_time': None,
            'tokens': 0,
            'ttft_ms': None,
            'total_ms': None
        }

        try:
            stats['start_time'] = time.perf_counter()

            # Prepend system prompt to first message in conversation
            if len(self.chat_session.history) == 0:
                text = f"{self._system_prompt}\n\nUser says: {text}\nAssistant:"

            # Stream response
            response = self.chat_session.send_message(text, stream=True)

            for chunk in response:
                try:
                    chunk_text = chunk.text
                except Exception as ce:
                    # Check for blocked content
                    if hasattr(chunk, 'prompt_feedback'):
                        print(f"[Blocked: {chunk.prompt_feedback}]")
                    continue

                if chunk_text:
                    # First token - TTFT
                    if stats['first_token_time'] is None:
                        stats['first_token_time'] = time.perf_counter()
                        stats['ttft_ms'] = round(
                            (stats['first_token_time'] - stats['start_time']) * 1000, 2
                        )
                        if verbose:
                            print(f"[TTFT: {stats['ttft_ms']:.0f}ms] ", end="", flush=True)

                    stats['tokens'] += 1
                    yield chunk_text, False, stats.copy()

            # Done
            stats['end_time'] = time.perf_counter()
            stats['total_ms'] = round(
                (stats['end_time'] - stats['start_time']) * 1000, 2
            )

            if verbose:
                print(f"\n[Total: {stats['total_ms']:.0f}ms | Tokens: {stats['tokens']}]")

            yield "", True, stats

        except Exception as e:
            print(f"\n[LLM Error: {e}]")
            import traceback
            traceback.print_exc()
            yield f"[Error: {e}]", True, {'error': str(e)}

    def generate(self, text, verbose=True):
        """Non-streaming (blocking) - use generate_stream for voice."""
        start = time.perf_counter()
        response = self.chat_session.send_message(text)
        elapsed = (time.perf_counter() - start) * 1000

        if verbose:
            print(f"LLM: {elapsed:.0f}ms")

        return response.text, {'total_ms': round(elapsed, 2)}

    def _extract_json_payload(self, raw_text):
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return None

        # Handle responses wrapped in markdown code fences.
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if len(lines) >= 3:
                raw_text = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw_text[start:end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    def generate_voice_response(self, user_text, verbose=True):
        """
        Ask LLM for a friendly spoken reply plus emotion label.

        Returns: ({"text": str, "emotion": str}, stats_dict)
        """
        start = time.perf_counter()
        emotion_list = ", ".join(VOICE_EMOTIONS)

        prompt = (
            f"{self._system_prompt}\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            '{"text":"<friendly spoken reply>","emotion":"<one emotion>"}\n'
            f"Allowed emotions: {emotion_list}.\n"
            "Emotion guidelines:\n"
            "- neutral: default for most responses\n"
            "- happy: good news, celebrations, positive outcomes\n"
            "- excited: surprising, enthusiastic moments\n"
            "- calm: reassuring, patient, soothing responses\n"
            "- sad: empathy for difficult situations\n"
            "- serious: important warnings, instructions\n"
            "- whisper: intimate, private, or sleepy moments\n"
            "- curious: asking questions, wanting to know more\n"
            "Choose one emotion that best matches your reply.\n"
            "No markdown. No explanation. JSON only.\n\n"
            f"User: {user_text}"
        )

        response = self.chat_session.send_message(prompt)
        elapsed = (time.perf_counter() - start) * 1000
        payload = self._extract_json_payload(getattr(response, "text", ""))

        if not payload:
            fallback_text = (getattr(response, "text", "") or "").strip()
            payload = {"text": fallback_text, "emotion": "neutral"}

        text = str(payload.get("text", "")).strip()
        emotion = str(payload.get("emotion", "neutral")).strip().lower()

        if emotion not in VOICE_EMOTIONS:
            emotion = "neutral"

        if not text:
            text = "I am here with you. Could you say that one more time?"
            emotion = "calm"

        if verbose:
            print(f"LLM: {elapsed:.0f}ms | emotion={emotion}")

        return {"text": text, "emotion": emotion}, {"total_ms": round(elapsed, 2)}

    def reset(self):
        """Clear conversation history."""
        self.chat_session = self.model.start_chat(history=[])


if __name__ == "__main__":
    llm = LLM()

    print("\nStreaming test:")
    print("-" * 40)

    for prompt in ["Hello", "What is 2+2?"]:
        print(f"\nUser: {prompt}")
        print("Assistant: ", end="")

        for chunk, done, stats in llm.generate_stream(prompt):
            if not done:
                print(chunk, end="", flush=True)
        print()