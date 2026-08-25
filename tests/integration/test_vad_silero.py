"""Tests d'intégration VAD avec le vrai modèle Silero (lents — marque `integration`)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.vad import SAMPLE_RATE, VADSegmenter
from server.vad_model import get_silero_probas

FIXTURES = Path(__file__).parent.parent / "fixtures" / "audio"


def load_16k_mono(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(len(x) * SAMPLE_RATE / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x.astype(np.float32)


@pytest.mark.integration
@pytest.mark.parametrize(
    "filename,min_speech_ratio",
    [("speech_sample1.flac", 0.5), ("speech_sample2_15s.wav", 0.3)],
)
def test_la_parole_reelle_est_segmentee(filename, min_speech_ratio):
    sig = load_16k_mono(FIXTURES / filename)
    seg = VADSegmenter(proba_fn=get_silero_probas())

    segments = []
    chunk = 1600  # 100 ms, ordre de grandeur des frames WS
    for i in range(0, len(sig), chunk):
        segments.extend(seg.process((sig[i : i + chunk] * 32767).astype(np.int16).tobytes()))
    final = seg.flush()
    if final is not None:
        segments.append(final)

    total_audio_ms = sum(s.duration_ms for s in segments)
    assert len(segments) >= 1
    assert total_audio_ms >= min_speech_ratio * len(sig) / SAMPLE_RATE * 1000
    # cohérence temporelle : les segments sont dans l'ordre et disjoints
    for a, b in zip(segments, segments[1:]):
        assert b.start_ms >= a.end_ms


@pytest.mark.integration
def test_le_silence_ne_produit_rien():
    rng = np.random.default_rng(1)
    silence = (rng.normal(0, 0.001, SAMPLE_RATE * 2)).astype(np.float32)
    seg = VADSegmenter(proba_fn=get_silero_probas())
    for i in range(0, len(silence), 1600):
        assert seg.process((silence[i : i + 1600] * 32767).astype(np.int16).tobytes()) == []
    assert seg.flush() is None
