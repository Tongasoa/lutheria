"""Tests du chargement de la configuration (pydantic-settings)."""

import pytest
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

from server.config import Settings


def test_defaults():
    s = Settings(mic_token="t", _env_file=None)
    assert s.asr_model == "Tongasoa/whisper-malagasy-medium-full-v2"
    assert s.mt_src_lang == "plt_Latn"
    assert s.mt_tgt_lang == "fra_Latn"
    assert s.vad_silence_ms == 400
    assert s.max_ws_frame_bytes == 65536
    assert s.max_segment_seconds == 15


def test_lang_codes_mt_sont_valides():
    """Un code hors FLORES-200 ne lève pas d'erreur : il résout vers <unk>.

    `mlg` est le code ISO 639-3 du malgache, mais NLLB utilise les codes
    FLORES-200 où il vaut `plt_Latn` (Plateau Malagasy). Le code fautif
    produisait un préfixe <unk> silencieux, sans crash ni perte visible.
    """
    s = Settings(mic_token="t", _env_file=None)
    assert s.mt_src_lang in FAIRSEQ_LANGUAGE_CODES
    assert s.mt_tgt_lang in FAIRSEQ_LANGUAGE_CODES


def test_env_override(monkeypatch):
    monkeypatch.setenv("LUTHERIA_MIC_TOKEN", "secret")
    monkeypatch.setenv("LUTHERIA_VAD_SILENCE_MS", "600")
    monkeypatch.setenv("LUTHERIA_MAX_SEGMENT_SECONDS", "20")
    s = Settings(_env_file=None)
    assert s.mic_token == "secret"
    assert s.vad_silence_ms == 600
    assert s.max_segment_seconds == 20


def test_mic_token_required(monkeypatch):
    """Le token producteur n'a pas de valeur par défaut exploitable."""
    monkeypatch.delenv("LUTHERIA_MIC_TOKEN", raising=False)
    with pytest.raises(Exception):
        Settings(_env_file=None)
