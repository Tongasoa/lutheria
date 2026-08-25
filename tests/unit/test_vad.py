"""Tests du segmenteur VAD (fenêtrage Silero, endpointing, coupe max).

Le modèle est simulé : `proba_fn` est une fonction injectée qui renvoie la
probabilité de parole pour chaque fenêtre de 512 échantillons.
"""

import numpy as np
import pytest

from server.vad import Segment, VADSegmenter

SR = 16000
WINDOW = 512  # échantillons par fenêtre Silero @16 kHz (32 ms)
WINDOW_BYTES = WINDOW * 2  # int16


def make_pcm(n_samples: int) -> bytes:
    """PCM int16 silencieux (zéros), n_samples quelconque."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def speech_pcm(n_samples: int, amplitude: int = 8000) -> bytes:
    t = np.arange(n_samples) / SR
    wave = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    return wave.tobytes()


def constant_probas(value: float):
    return lambda window: value


def scripted_probas(values_per_window):
    """Proba fixée fenêtre par fenêtre (dernière valeur répétée si dépassé)."""
    state = {"i": 0}

    def fn(window):
        i = min(state["i"], len(values_per_window) - 1)
        state["i"] += 1
        return values_per_window[i]

    return fn


@pytest.fixture
def seg():
    return VADSegmenter(proba_fn=constant_probas(0.0))


class TestFenetrage:
    def test_tailles_de_chunk_arbitraires_ne_perturbent_pas(self, seg):
        """Des chunks non multiples de la fenêtre sont bufferisés correctement."""
        out = seg.process(make_pcm(WINDOW_BYTES + 100))
        assert out == []
        out = seg.process(make_pcm(37))  # reste du buffer
        assert out == []
        # 1 fenêtre complète + 137 échantillons en attente
        assert seg.pending_samples == 137

    def test_aucun_segment_sur_silence_continu(self, seg):
        out = seg.process(make_pcm(SR * 3))  # 3 s de silence
        assert out == []


class TestEndpointing:
    def test_parole_puis_silence_emet_un_segment(self):
        # 20 fenêtres de parole (~640 ms) puis silence > 400 ms
        probas = [0.9] * 20 + [0.1] * 40
        seg = VADSegmenter(proba_fn=scripted_probas(probas))
        segments = seg.process(speech_pcm(60 * WINDOW))
        assert len(segments) == 1
        s = segments[0]
        # le segment couvre les fenêtres de parole (+ pré-roll) mais pas le silence final
        assert WINDOW_BYTES <= len(s.audio) <= 30 * WINDOW_BYTES
        assert s.end_ms > s.start_ms

    def test_micro_pause_ne_coupe_pas_le_segment(self):
        """Une pause < silence_ms au milieu de la parole ne ferme pas le segment."""
        # parole, 5 fenêtres de silence (~160 ms < 400 ms), parole, puis silence final
        pattern = [0.9] * 10 + [0.1] * 5 + [0.9] * 10 + [0.1] * 20
        seg = VADSegmenter(proba_fn=scripted_probas(pattern))
        segments = seg.process(speech_pcm(len(pattern) * WINDOW))
        assert len(segments) == 1
        # pré-roll(10) + parole(10) + micro-pause(5) conservées, silence final rogné
        assert 20 * WINDOW_BYTES <= len(segments[0].audio) <= 26 * WINDOW_BYTES

    def test_deux_enonces_separes_donnet_deux_segments(self):
        # énoncé, longue pause (>400 ms = >12 fenêtres), énoncé
        pattern = [0.9] * 10 + [0.1] * 20 + [0.9] * 10 + [0.1] * 40
        seg = VADSegmenter(proba_fn=scripted_probas(pattern))
        segments = seg.process(speech_pcm(len(pattern) * WINDOW))
        assert len(segments) == 2

    def test_preroll_inclus_au_debut_du_segment(self):
        """Le pré-roll (10 fenêtres) précède le déclenchement."""
        probas = [0.1] * 15 + [0.9] * 10 + [0.1] * 40
        seg = VADSegmenter(
            proba_fn=scripted_probas(probas),
            prebuffer_windows=10,
            silence_ms=400,
        )
        segments = seg.process(speech_pcm(len(probas) * WINDOW))
        assert len(segments) == 1
        # déclenchement vers la fenêtre 15 ; segment doit démarrer ~fenêtre 5
        start_ms = segments[0].start_ms
        assert 4 * 32 <= start_ms <= 8 * 32


class TestCoupeMax:
    def test_monologue_long_est_coupe(self):
        # 50 s de parole continue, max 15 s -> au moins 3 segments
        seg = VADSegmenter(proba_fn=constant_probas(0.9), max_segment_seconds=15)
        segments = seg.process(speech_pcm(SR * 50))
        assert len(segments) >= 3
        for s in segments[:-1]:  # les coupures sont à ~max_segment_seconds
            assert s.duration_ms <= 15 * 1000 + 32


class TestFlush:
    def test_flush_emet_la_parole_en_cours(self):
        probas = [0.1] * 5 + [0.9] * 20  # parole non terminée par un silence
        seg = VADSegmenter(proba_fn=scripted_probas(probas))
        segments = seg.process(speech_pcm(25 * WINDOW))
        assert segments == []  # pas encore de fin détectée
        final = seg.flush()
        assert final is not None
        assert len(final.audio) >= 15 * WINDOW_BYTES

    def test_flush_sans_parole_renvoie_none(self, seg):
        seg.process(make_pcm(SR))
        assert seg.flush() is None


class TestReset:
    def test_etat_reinitialise_apres_emission(self):
        pattern = [0.9] * 10 + [0.1] * 20 + [0.9] * 10 + [0.1] * 40
        seg = VADSegmenter(proba_fn=scripted_probas(pattern))
        seg.process(speech_pcm(len(pattern) * WINDOW))
        assert not seg.in_speech
        assert seg.silence_ms_accumulated == pytest.approx(400, abs=64)


class TestFormatSegment:
    def test_audio_bytes_int16(self):
        seg = VADSegmenter(proba_fn=constant_probas(0.9))
        segments = seg.process(speech_pcm(30 * WINDOW))
        assert all(isinstance(s.audio, bytes) for s in segments)

    def test_named_tuple_fields(self):
        fields = Segment._fields
        assert set(fields) == {"audio", "start_ms", "end_ms"}
