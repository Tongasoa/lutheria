"""ASR : transcription malgache des segments audio via faster-whisper (CTranslate2).

L'interface `ASREngine` est volontairement minimale pour rester mockable ; le
modèle réel est chargé paresseusement (`WhisperASR.ensure_loaded`) afin que les
tests et le démarrage du serveur n'exigent aucun poids sur disque.
"""

from typing import NamedTuple, Protocol

import numpy as np

SAMPLE_RATE = 16000


class Transcription(NamedTuple):
    text: str
    language: str


class ASREngine(Protocol):
    def transcribe(self, audio: bytes, sample_rate: int = SAMPLE_RATE) -> Transcription: ...


class WhisperASR:
    """Enveloppe faster-whisper : PCM int16 -> texte, langue forcée, greedy."""

    def __init__(self, model_path: str, language: str = "mg", device: str = "cpu",
                 compute_type: str = "int8") -> None:
        self._model_path = model_path
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._model = None  # chargé à la première utilisation

    def ensure_loaded(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_path, device=self._device, compute_type=self._compute_type
            )

    def transcribe(self, audio: bytes, sample_rate: int = SAMPLE_RATE) -> Transcription:
        self.ensure_loaded()
        float_audio = self._to_float32(audio, sample_rate)
        raw_segments, info = self._model.transcribe(
            float_audio, language=self._language, beam_size=1
        )
        text = " ".join(s.text.strip() for s in raw_segments).strip()
        return Transcription(text=text, language=getattr(info, "language", self._language))

    @staticmethod
    def _to_float32(audio: bytes, sample_rate: int) -> np.ndarray:
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != SAMPLE_RATE and len(samples) > 0:
            n_out = int(round(len(samples) * SAMPLE_RATE / sample_rate))
            samples = np.interp(
                np.linspace(0.0, len(samples) - 1, n_out),
                np.arange(len(samples), dtype=np.float64),
                samples,
            ).astype(np.float32)
        return samples


def build_asr(settings) -> WhisperASR:
    """Fabrique l'ASR depuis la configuration (chemin modèle swap-able par env)."""
    return WhisperASR(
        model_path=settings.asr_model,
        language=settings.asr_language,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
    )
