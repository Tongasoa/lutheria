"""Tests de l'interface ASREngine et de son implémentation faster-whisper (mockée)."""

import numpy as np
import pytest

from server.asr import Transcription, WhisperASR


class TestTranscription:
    def test_fields(self):
        t = Transcription(text="salama", language="mg")
        assert t.text == "salama"
        assert t.language == "mg"


class TestWhisperASR:
    """L'implémentation est testée avec un moteur interne simulé (pas de modèle)."""

    def make_engine(self, segments, monkeypatch):
        engine = WhisperASR(model_path="fake", language="mg", device="cpu", compute_type="int8")

        class FakeInner:
            def transcribe(self, audio, language=None, beam_size=None):
                return iter(segments), type("Info", (), {"language": "mg"})()

        monkeypatch.setattr(engine, "_model", FakeInner())
        return engine

    def test_transcribe_joint_les_segments(self, monkeypatch):
        engine = self.make_engine(
            [type("S", (), {"text": " manao "})(), type("S", (), {"text": " ahoana "})()],
            monkeypatch,
        )
        result = engine.transcribe(b"\x00\x00" * 1600)
        assert result == Transcription(text="manao ahoana", language="mg")

    def test_audio_vide_renvoie_texte_vide(self, monkeypatch):
        engine = self.make_engine([], monkeypatch)
        assert engine.transcribe(b"").text == ""

    def test_pcm_int16_normalise_en_float32(self, monkeypatch):
        captured = {}

        class FakeInner:
            def transcribe(self, audio, language=None, beam_size=None):
                captured["audio"] = audio
                return iter([]), None

        engine = WhisperASR(model_path="fake", language="mg", device="cpu", compute_type="int8")
        monkeypatch.setattr(engine, "_model", FakeInner())
        pcm = np.array([16384, -16384], dtype=np.int16).tobytes()
        engine.transcribe(pcm)
        np.testing.assert_allclose(captured["audio"], [0.5, -0.5])

    def test_resampling_8k_vers_16k(self, monkeypatch):
        captured = {}

        class FakeInner:
            def transcribe(self, audio, language=None, beam_size=None):
                captured["audio"] = audio
                captured["sr"] = getattr(audio, "__len__", lambda: 0)()
                return iter([]), None

        engine = WhisperASR(model_path="fake", language="mg", device="cpu", compute_type="int8")
        monkeypatch.setattr(engine, "_model", FakeInner())
        # 1 s de silence à 8 kHz -> doit être rééchantillonné à ~16k échantillons
        engine.transcribe(np.zeros(8000, dtype=np.int16).tobytes(), sample_rate=8000)
        assert len(captured["audio"]) == pytest.approx(16000, abs=100)
