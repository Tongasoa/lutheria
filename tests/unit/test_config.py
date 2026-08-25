"""Tests du chargement de la configuration (pydantic-settings)."""

import pytest

from server.config import Settings


def test_defaults():
    s = Settings(mic_token="t", _env_file=None)
    assert s.asr_model == "Tongasoa/whisper-malagasy-medium-full-v2"
    assert s.mt_src_lang == "mlg_Latn"
    assert s.mt_tgt_lang == "fra_Latn"
    assert s.vad_silence_ms == 400
    assert s.max_ws_frame_bytes == 65536
    assert s.max_segment_seconds == 15


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
