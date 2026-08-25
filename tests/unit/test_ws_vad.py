"""Tests du branchement VAD dans /ws/mic observé à travers les appels à l'ASR."""

import time

import numpy as np

from server.config import Settings
from server.main import create_app
from server.vad import WINDOW_BYTES, VADSegmenter


def scripted_probas(values_per_window):
    state = {"i": 0}

    def fn(window):
        i = min(state["i"], len(values_per_window) - 1)
        state["i"] += 1
        return values_per_window[i]

    return fn


class RecordingASR:
    """Faux ASR : enregistre les audios reçus, ne diffuse rien."""

    def __init__(self):
        self.audios = []

    def transcribe(self, audio, sample_rate=16000):
        self.audios.append(audio)
        from server.asr import Transcription

        return Transcription(text="", language="mg")


def make_vad_factory(probas):
    def factory(settings, proba_fn=None):
        return VADSegmenter(
            proba_fn=scripted_probas(probas),
            silence_ms=settings.vad_silence_ms,
            max_segment_seconds=settings.max_segment_seconds,
        )

    return factory


def speech_pcm(n_windows: int) -> bytes:
    t = np.arange(n_windows * 512) / 16000
    return (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16).tobytes()


def make_app(asr, probas=None):
    settings = Settings(mic_token="tok", _env_file=None)
    factory = make_vad_factory(probas) if probas else (lambda s, p=None: VADSegmenter(proba_fn=lambda w: 0.9, max_segment_seconds=1))
    app = create_app(settings, vad_factory=factory, asr_factory=lambda s: asr)
    from fastapi.testclient import TestClient

    return app, TestClient(app)


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_les_segments_de_parole_parviennent_a_l_asr():
    # 10 fenêtres de parole puis 20 de silence -> exactement 1 segment
    asr = RecordingASR()
    app, client = make_app(asr, [0.9] * 10 + [0.1] * 20)
    with client.websocket_connect("/ws/mic?token=tok") as ws:
        for _ in range(30):
            ws.send_bytes(b"\x00" * WINDOW_BYTES)
            ws.receive_bytes()
    assert wait_until(lambda: len(asr.audios) == 1)
    # le segment couvre au moins la parole, sans le silence final entier
    assert WINDOW_BYTES * 5 <= len(asr.audios[0]) <= WINDOW_BYTES * 25


def test_flush_a_la_deconnexion():
    asr = RecordingASR()
    app, client = make_app(asr, [0.1] * 2 + [0.9] * 15)  # parole jamais close par silence
    with client.websocket_connect("/ws/mic?token=tok") as ws:
        for _ in range(17):
            ws.send_bytes(b"\x00" * WINDOW_BYTES)
            ws.receive_bytes()
    assert wait_until(lambda: len(asr.audios) == 1)  # émis par le flush final


def test_parole_continue_est_coupee_toutes_les_1s():
    asr = RecordingASR()
    app, client = make_app(asr)  # max_segment_seconds=1 -> coupes toutes ~31 fenêtres
    with client.websocket_connect("/ws/mic?token=tok") as ws:
        for _ in range(80):  # ~2,5 s de parole continue
            ws.send_bytes(speech_pcm(1))
            ws.receive_bytes()
    assert wait_until(lambda: len(asr.audios) == 3)  # 2 coupes + flush final


def test_file_ne_depasse_jamais_sa_capacite():
    from server.main import SEGMENT_QUEUE_MAXSIZE

    asr = RecordingASR()
    app, client = make_app(asr)
    with client.websocket_connect("/ws/mic?token=tok"):
        assert app.state.segment_queue.maxsize == SEGMENT_QUEUE_MAXSIZE
