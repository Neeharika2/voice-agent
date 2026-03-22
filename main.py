"""
Voice Agent Main Entry Point

Orchestrates STT → LLM → TTS pipeline with streaming for low-latency responses.
"""

import re
import signal
import sys
import time
from modules.stt import STT
from modules.llm import LLM
from modules.tts import TTS
import config


# Global for cleanup
_tts_engine = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nShutting down...")
    if _tts_engine:
        _tts_engine.shutdown()
    sys.exit(0)


def extract_speakable_units(text: str, min_chars: int, max_chars: int):
    """Return (ready_units, remaining_buffer) for low-latency incremental TTS."""
    if not text:
        return [], ""

    delimiters = ".!?;,:\n"
    units = []
    cursor = 0

    for idx, ch in enumerate(text):
        if ch in delimiters:
            segment = text[cursor:idx + 1].strip()
            if len(segment) >= min_chars or ch in ".!?\n":
                units.append(segment)
                cursor = idx + 1

    remaining = text[cursor:].lstrip()

    # If the buffer grows too long without punctuation, flush to nearest word break.
    if len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at > 0:
            units.append(remaining[:split_at].strip())
            remaining = remaining[split_at + 1:].lstrip()

    return [u for u in units if u], remaining


def main():
    global _tts_engine

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print("Initializing STT...")
    stt = STT()

    print("Initializing LLM...")
    llm = LLM()

    print("Initializing TTS...")
    tts = TTS()
    _tts_engine = tts

    print("\n" + "=" * 50)
    print("Voice Agent Ready! Speak into your microphone.")
    print("Press Ctrl+C to exit")
    print("=" * 50 + "\n")

    while True:
        try:
            turn_start = time.perf_counter()

            # 1. Listen
            text, stt_stats = stt.listen(verbose=True)
            stt_done = time.perf_counter()

            if not text:
                continue

            # 2. Start TTS pipeline
            tts.start_streaming()

            # 3. Stream LLM, queue sentences immediately
            print("Agent: ", end="", flush=True)

            buffer = ""
            unit_count = 0
            llm_start = time.perf_counter()
            llm_ttft_ms = None
            first_unit_queue_ms = None
            stream_min_chars = int(getattr(config, "TTS_STREAM_MIN_CHARS", 20))
            stream_max_chars = int(getattr(config, "TTS_STREAM_MAX_CHARS", 90))

            for chunk, done, stats in llm.generate_stream(text, verbose=False):
                if stats and stats.get("ttft_ms") is not None and llm_ttft_ms is None:
                    llm_ttft_ms = float(stats["ttft_ms"])

                if chunk:
                    buffer += chunk

                    ready_units, buffer = extract_speakable_units(
                        buffer,
                        min_chars=stream_min_chars,
                        max_chars=stream_max_chars,
                    )

                    for unit in ready_units:
                        print(unit, end=" ", flush=True)
                        tts.queue_sentence(unit)
                        unit_count += 1
                        if first_unit_queue_ms is None:
                            first_unit_queue_ms = (time.perf_counter() - llm_start) * 1000

            llm_done = time.perf_counter()

            # Handle remaining text
            if buffer.strip():
                print(buffer, end="", flush=True)
                tts.queue_sentence(buffer.strip())
                unit_count += 1
                if first_unit_queue_ms is None:
                    first_unit_queue_ms = (time.perf_counter() - llm_start) * 1000

            print("\n")

            # 4. Wait for all audio to play
            tts_stats = {}
            if unit_count > 0:
                tts_stats = tts.finish_streaming()

            # 5. Clear mic buffer for next input
            stt.clear_audio_buffer()

            if bool(getattr(config, "LATENCY_LOGS", True)):
                stt_ms = (stt_done - turn_start) * 1000
                llm_total_ms = (llm_done - llm_start) * 1000
                tts_total_ms = 0.0
                if tts_stats.get("start_time") and tts_stats.get("end_time"):
                    tts_total_ms = (tts_stats["end_time"] - tts_stats["start_time"]) * 1000

                total_turn_ms = (time.perf_counter() - turn_start) * 1000
                print(
                    "[Latency:Turn] "
                    f"stt={stt_ms:.0f}ms "
                    f"stt_post={stt_stats.get('processing_ms', 0):.0f}ms "
                    f"llm_ttft={llm_ttft_ms or 0:.0f}ms "
                    f"llm_total={llm_total_ms:.0f}ms "
                    f"queue_first={first_unit_queue_ms or 0:.0f}ms "
                    f"tts_total={tts_total_ms:.0f}ms "
                    f"turn_total={total_turn_ms:.0f}ms "
                    f"units={unit_count}"
                )

        except KeyboardInterrupt:
            signal_handler(None, None)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()