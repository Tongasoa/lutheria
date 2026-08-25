"""Tests d'intégration ASR avec un vrai modèle faster-whisper (marque `integration`).

Utilise `tiny` (téléchargé au premier lancement, ~75 Mo) et une langue connue
pour valider la mécanique bout-en-bout PCM -> texte, indépendamment du
fine-tuning malgache (voir ADR 0002).
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.asr import WhisperASR

FIXTURES = Path(__file__).parent.parent / "fixtures" / "audio"


@pytest.fixture(scope="module")
def engine():
    return WhisperASR(model_path="tiny", language="en", device="cpu", compute_type="int8")


def load_16k_mono_int16(path: Path) -> bytes:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        n = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return (x * 32767).astype(np.int16).tobytes()


@pytest.mark.integration
def test_transcription_non_vide_sur_parole_reelle(engine):
    audio = load_16k_mono_int16(FIXTURES / "speech_sample1.flac")
    result = engine.transcribe(audio)
    assert isinstance(result.text, str)
    assert len(result.text) > 3
    assert result.language == "en"


@pytest.mark.integration
def test_silence_renvoie_texte_vide_ou_inoffensif(engine):
    audio = np.zeros(16000 * 2, dtype=np.int16).tobytes()  # 2 s de silence
    result = engine.transcribe(audio)
    assert len(result.text) < 200  # pas d'hallucination délirante sur du silence


@pytest.mark.integration
def test_resampling_8k_fonctionne(engine):
    x, _ = sf.read(FIXTURES / "speech_sample2_15s.wav", dtype="float32")
    audio = (x[:8000] * 32767).astype(np.int16).tobytes()  # 0,5 s @16k -> simulé 8k
    result = engine.transcribe(audio, sample_rate=8000)
    assert isinstance(result.text, str)  # ne plante pas ; texte libre à cette durée
