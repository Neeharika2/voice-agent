import queue
import sys
import os
import json
import time
from queue import Empty
import sounddevice as sd
from vosk import Model, KaldiRecognizer

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config


class STT:
    def __init__(self, model_path=config.VOSK_MODEL_PATH, blocksize=None):
        self.blocksize = blocksize or config.STT_BLOCKSIZE
        self.silence_timeout = config.STT_SILENCE_TIMEOUT
        self.audio_queue = queue.Queue()

        self._check_model_performance(model_path)

        print(f"Loading Vosk model from '{model_path}'...")
        self.model = Model(model_path)
        self.recognizer = None

        print(f"STT initialized: chunk_size={self.blocksize} samples ({self.blocksize/config.SAMPLE_RATE*1000:.1f}ms)")

    def clear_audio_buffer(self):
        """Drain queued microphone frames to avoid stale or echoed audio."""
        try:
            while True:
                self.audio_queue.get_nowait()
        except Empty:
            pass

    def _check_model_performance(self, model_path):
        try:
            total_size = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(model_path)
                for f in files
            )
            size_mb = total_size / (1024 * 1024)
            if size_mb > 500:
                print(f"\n⚠ WARNING: Model is {size_mb:.0f}MB - large models have higher latency")
                print("For sub-100ms latency, use a small model (~50MB)")
                print("Download from: https://alphacephei.com/vosk/models\n")
        except:
            pass

    def _init_recognizer(self):
        self.recognizer = KaldiRecognizer(self.model, config.SAMPLE_RATE)
        self.recognizer.SetWords(True)

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def listen(self, verbose=True):
        """
        Listen for speech and return transcription with latency stats.

        Returns: (text, stats_dict)
        """
        self._init_recognizer()
        self.clear_audio_buffer()

        stats = {
            'speech_end': None,  # When partial became empty (user stopped)
            'end_time': None,    # When final result returned
        }

        if verbose:
            print("\nListening... (Speak now)")

        stream = sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=self.blocksize,
            dtype='int16',
            channels=1,
            callback=self._callback
        )

        with stream:
            silence_start = None
            last_partial = ""
            had_speech = False  # Track if we've seen any speech

            while True:
                data = self.audio_queue.get()
                now = time.perf_counter()

                # Process with Vosk
                if self.recognizer.AcceptWaveform(data):
                    # Vosk finalized an utterance
                    stats['end_time'] = now
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        if verbose:
                            self._print_stats(text, stats)
                        return text, self._compute_stats(stats)

                # Check partials - this is how we detect REAL speech
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get("partial", "").strip()

                if partial_text:
                    had_speech = True
                    silence_start = None  # Reset silence timer

                    if partial_text != last_partial:
                        if verbose:
                            print(f"\r  Live: {partial_text}        ", end="", flush=True)
                        last_partial = partial_text

                else:
                    # No partial = silence
                    # Only start silence detection AFTER speech was detected
                    if had_speech:
                        if silence_start is None:
                            silence_start = now
                            stats['speech_end'] = now  # Mark when speech ended
                        elif (now - silence_start) >= self.silence_timeout:
                            # Silence timeout - force finalize
                            stats['end_time'] = now
                            result = json.loads(self.recognizer.FinalResult())
                            text = result.get("text", "").strip()
                            if text:
                                if verbose:
                                    self._print_stats(text, stats)
                                return text, self._compute_stats(stats)
                            else:
                                # No text, reset and continue listening
                                silence_start = None
                                had_speech = False
                                if verbose:
                                    print("\r  (listening...)        ", end="", flush=True)

    def _compute_stats(self, stats):
        result = {
            'chunk_ms': round(self.blocksize / config.SAMPLE_RATE * 1000, 2),
        }

        # Processing time: speech end → final result (AFTER user stops speaking)
        if stats['speech_end'] and stats['end_time']:
            result['processing_ms'] = round(
                (stats['end_time'] - stats['speech_end']) * 1000, 2
            )

        return result

    def _print_stats(self, text, stats):
        s = self._compute_stats(stats)
        print(f"\n\n{'='*55}")
        print(f"STT: \"{text}\"")
        print(f"{'-'*55}")
        if 'processing_ms' in s:
            print(f"Processing (after speech): {s['processing_ms']:.1f} ms")
        print(f"{'='*55}")


if __name__ == "__main__":
    stt = STT()
    print("\nSpeak into your microphone. Ctrl+C to stop.")
    while True:
        try:
            text, stats = stt.listen()
        except KeyboardInterrupt:
            print("\nStopped.")
            break

            