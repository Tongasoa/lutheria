"""Segmentation de parole en continu autour de Silero VAD (ONNX, CPU).

Le modèle est injecté sous forme de fonction `proba_fn(window_f32) -> float`
pour rester testable sans dépendance au modèle réel (voir ADR 0001/0004).
"""

from collections import deque
from typing import Callable, NamedTuple

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512  # fenêtre Silero @16 kHz (= 32 ms)
BYTES_PER_SAMPLE = 2  # PCM int16 mono
WINDOW_BYTES = WINDOW_SAMPLES * BYTES_PER_SAMPLE


class Segment(NamedTuple):
    audio: bytes  # PCM int16 16 kHz mono
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class VADSegmenter:
    """Transforme un flux PCM arbitrairement découpé en segments de parole.

    - endpointing : fin de segment après `silence_ms` sans parole ;
    - pré-roll : `prebuffer_windows` fenêtres gardées en mémoire tampon pour ne
      pas rognner le début des mots ;
    - coupe de sécurité : segment émis dès que sa durée atteint
      `max_segment_seconds`.
    """

    def __init__(
        self,
        proba_fn: Callable[[np.ndarray], float],
        silence_ms: int = 400,
        threshold: float = 0.5,
        max_segment_seconds: int = 15,
        prebuffer_windows: int = 10,
        tail_windows: int = 5,
    ) -> None:
        self.proba_fn = proba_fn
        self.silence_ms = silence_ms
        self.threshold = threshold
        self.max_segment_seconds = max_segment_seconds
        self.prebuffer_windows = prebuffer_windows
        self.tail_windows = tail_windows

        self._pending = bytearray()  # échantillons int16 en attente d'une fenêtre complète
        self._in_speech = False
        self._preroll: deque[np.ndarray] = deque(maxlen=max(prebuffer_windows, 0))
        self._segment_windows: list[np.ndarray] = []
        self._segment_probas: list[float] = []
        self._silence_ms_acc = 0
        self._samples_seen = 0  # compteur absolu d'échantillons consommés
        self._segment_start_sample = 0

    # --- état exposé (tests / supervision) ---

    @property
    def pending_samples(self) -> int:
        return len(self._pending) // BYTES_PER_SAMPLE

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def silence_ms_accumulated(self) -> int:
        return self._silence_ms_acc

    # --- API principale ---

    def process(self, pcm: bytes) -> list[Segment]:
        self._pending.extend(pcm)
        window_bytes = WINDOW_SAMPLES * BYTES_PER_SAMPLE
        segments: list[Segment] = []
        while len(self._pending) >= window_bytes:
            raw = self._pending[:window_bytes]
            del self._pending[:window_bytes]
            self._samples_seen += WINDOW_SAMPLES
            seg = self._process_window(
                np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            )
            if seg is not None:
                segments.append(seg)
        return segments

    def flush(self) -> Segment | None:
        """Émet le segment en cours (fin de flux) même sans silence détecté."""
        if not self._in_speech or not self._segment_windows:
            return None
        segment = self._emit(trim_tail=False)
        self._reset()
        return segment

    # --- interne ---

    def _process_window(self, window: np.ndarray) -> Segment | None:
        proba = float(self.proba_fn(window))
        window_ms = 1000.0 * WINDOW_SAMPLES / SAMPLE_RATE

        if not self._in_speech:
            if proba >= self.threshold:
                self._in_speech = True
                self._segment_windows = list(self._preroll)
                self._segment_probas = [self.threshold] * len(self._preroll)
                self._segment_start_sample = self._samples_seen - len(self._segment_windows) * WINDOW_SAMPLES
                self._append_window(window, proba)
            else:
                self._preroll.append(window)
            return None

        self._append_window(window, proba)

        if proba < self.threshold:
            self._silence_ms_acc += window_ms
            if self._silence_ms_acc >= self.silence_ms:
                segment = self._emit(trim_tail=True)
                self._reset()
                return segment
        else:
            self._silence_ms_acc = 0
            if self._duration_ms() >= self.max_segment_seconds * 1000:
                segment = self._emit(trim_tail=False)
                self._reset(keep_speech=True)
                return segment
        return None

    def _append_window(self, window: np.ndarray, proba: float) -> None:
        self._segment_windows.append(window)
        self._segment_probas.append(proba)

    def _start_sample(self) -> int:
        return self._segment_start_sample

    def _duration_ms(self) -> float:
        return 1000.0 * len(self._segment_windows) * WINDOW_SAMPLES / SAMPLE_RATE

    def _emit(self, trim_tail: bool) -> Segment:
        windows = self._segment_windows
        probas = self._segment_probas
        if trim_tail:
            # on ne garde au maximum que `tail_windows` fenêtres silencieuses en fin de segment
            while len(windows) > self.tail_windows and probas[-1] < self.threshold:
                windows.pop()
                probas.pop()
        start = self._start_sample()
        audio = (np.concatenate(windows) * 32767.0).astype(np.int16).tobytes()
        end_ms = (start + len(windows) * WINDOW_SAMPLES) / SAMPLE_RATE * 1000
        return Segment(audio=audio, start_ms=start / SAMPLE_RATE * 1000, end_ms=end_ms)

    def _reset(self, keep_speech: bool = False) -> None:
        self._in_speech = keep_speech
        self._segment_windows = []
        self._segment_probas = []
        if not keep_speech:
            self._preroll.clear()
