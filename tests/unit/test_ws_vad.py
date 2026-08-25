"""Tests du branchement VAD dans /ws/mic : les segments arrivent dans la file."""

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

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


def make_vad_factory(probas):
    def factory(settings):
        return VADSegmenter(
            proba_fn=scripted_probas(probas),
            silence_ms=settings.vad_silence_ms,
            max_segment_seconds=settings.max_segment_seconds,
        )

    return factory


def speech_pcm(n_windows: int) -> bytes:
    t = np.arange(n_windows * 512) / 16000
    return (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16).tobytes()


def drain(queue: asyncio.Queue):
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def test_les_segments_de_parole_arrivent_dans_la_file():
    # 10 fenêtres de parole puis 20 de silence -> 1 segment émis pendant l'envoi
    probas = [0.9] * 10 + [0.1] * 20
    app = create_app(Settings(mic_token="tok", _env_file=None), vad_factory=make_vad_factory(probas))
    client = TestClient(app)
    with client.websocket_connect("/ws/mic?token=tok") as ws:
        for _ in range(30):
            ws.send_bytes(b"\x00" * WINDOW_BYTES)
            ws.receive_bytes()  # ack applicatif par fenêtre traitée
        segments = drain(app.state.segment_queue)
    assert len(segments) == 1
    assert segments[0].duration_ms > 0


def test_flush_a_la_deconnexion():
    probas = [0.1] * 2 + [0.9] * 15  # parole jamais close par un silence
    app = create_app(Settings(mic_token="tok", _env_file=None), vad_factory=make_vad_factory(probas))
    client = TestClient(app)
    with client.websocket_connect("/ws/mic?token=tok") as ws:
        for _ in range(17):
            ws.send_bytes(b"\x00" * WINDOW_BYTES)
            ws.receive_bytes()
    segments = drain(app.state.segment_queue)
    assert len(segments) == 1  # émis au flush lors de la déconnexion


def test_file_pleine_ne_bloque_pas_le_producteur():
    from server.main import SEGMENT_QUEUE_MAXSIZE

    # parole continue coupée toutes les 1 s : émissions à ~32, 64 fenêtres puis flush
    app = create_app(
        Settings(mic_token="tok", _env_file=None),
        vad_factory=lambda s: VADSegmenter(proba_fn=lambda w: 0.9, max_segment_seconds=1),
    )
    client = TestClient(app)
    with client.websocket_connect("/ws/mic?token=tok") as ws:
        for _ in range(80):  # ~2,5 s de parole
            ws.send_bytes(speech_pcm(1))
            ws.receive_bytes()
    segments = drain(app.state.segment_queue)
    assert app.state.segment_queue.maxsize == SEGMENT_QUEUE_MAXSIZE
    assert len(segments) == 3  # coupe à 1 s, coupe à 2 s, flush final
