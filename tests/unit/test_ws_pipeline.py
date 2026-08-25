"""Tests du pipeline : segments VAD -> ASR -> diffusion `partial` aux lecteurs."""

import asyncio
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.asr import Transcription
from server.config import Settings
from server.main import create_app
from server.vad import WINDOW_BYTES


def scripted_probas(values_per_window):
    state = {"i": 0}

    def fn(window):
        i = min(state["i"], len(values_per_window) - 1)
        state["i"] += 1
        return values_per_window[i]

    return fn


def speech_pcm(n_windows: int) -> bytes:
    t = np.arange(n_windows * 512) / 16000
    return (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16).tobytes()


class FakeASR:
    def __init__(self, results=None, raise_on_first=False):
        self.results = list(results or [])
        self.raise_on_first = raise_on_first
        self.calls = []

    def transcribe(self, audio, sample_rate=16000):
        self.calls.append(audio)
        if self.raise_on_first and len(self.calls) == 1:
            raise RuntimeError("boom ASR")
        return Transcription(text=self.results.pop(0) if self.results else "", language="mg")


def make_app(asr, probas):
    def vad_factory(settings, proba_fn=None):
        from server.vad import VADSegmenter

        return VADSegmenter(
            proba_fn=scripted_probas(probas),
            silence_ms=settings.vad_silence_ms,
            max_segment_seconds=settings.max_segment_seconds,
        )

    settings = Settings(mic_token="tok", _env_file=None)
    app = create_app(settings, vad_factory=vad_factory, asr_factory=lambda s: asr)
    return app, TestClient(app)


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_le_texte_malgache_est_diffuse_en_partial():
    # 1 segment de parole (10 fenêtres) fermé par 20 fenêtres de silence
    asr = FakeASR(results=["manao ahoana"])
    app, client = make_app(asr, [0.9] * 10 + [0.1] * 20)
    with client.websocket_connect("/ws/listen") as listener:
        with client.websocket_connect("/ws/mic?token=tok") as mic:
            for _ in range(30):
                mic.send_bytes(b"\x00" * WINDOW_BYTES)
                mic.receive_bytes()
        msg = listener.receive_json()
    assert isinstance(msg["ts"], float)
    assert {k: msg[k] for k in ("id", "state", "text_mg")} == {
        "id": 1,
        "state": "partial",
        "text_mg": "manao ahoana",
    }


def test_les_ids_s_incrementent_entre_segments():
    asr = FakeASR(results=["premier", "deuxieme"])
    # deux énoncés séparés par une longue pause
    app, client = make_app(asr, [0.9] * 8 + [0.1] * 20 + [0.9] * 8 + [0.1] * 20)
    with client.websocket_connect("/ws/listen") as listener:
        with client.websocket_connect("/ws/mic?token=tok") as mic:
            for _ in range(56):
                mic.send_bytes(b"\x00" * WINDOW_BYTES)
                mic.receive_bytes()
        m1 = listener.receive_json()
        m2 = listener.receive_json()
    assert (m1["id"], m2["id"]) == (1, 2)
    assert [m1["text_mg"], m2["text_mg"]] == ["premier", "deuxieme"]


def test_transcription_vide_n_est_pas_diffusee():
    asr = FakeASR(results=[""])  # faux positif VAD -> texte vide
    app, client = make_app(asr, [0.9] * 10 + [0.1] * 20)
    received = []

    class SpyHub:
        async def publish(self, msg):
            received.append(msg)

    app.state.hub = SpyHub()
    with client.websocket_connect("/ws/mic?token=tok") as mic:
        for _ in range(30):
            mic.send_bytes(b"\x00" * WINDOW_BYTES)
            mic.receive_bytes()
    assert wait_until(lambda: len(asr.calls) == 1)
    assert wait_until(lambda: not asr.results and not received)


def test_echec_asr_ne_tue_pas_le_pipeline():
    asr = FakeASR(results=["recuperation"], raise_on_first=True)
    app, client = make_app(asr, [0.9] * 8 + [0.1] * 20 + [0.9] * 8 + [0.1] * 20)
    with client.websocket_connect("/ws/listen") as listener:
        with client.websocket_connect("/ws/mic?token=tok") as mic:
            for _ in range(56):
                mic.send_bytes(b"\x00" * WINDOW_BYTES)
                mic.receive_bytes()
        msg = listener.receive_json()  # seul le 2e segment aboutit
    assert {k: msg[k] for k in ("id", "state", "text_mg")} == {
        "id": 2,
        "state": "partial",
        "text_mg": "recuperation",
    }


def test_le_worker_est_cree_puis_annule_au_shutdown():
    asr = FakeASR(results=[])
    app, client = make_app(asr, [0.9] * 10 + [0.1] * 20)
    with TestClient(app) as entered:  # entre dans le cycle de vie complet
        with entered.websocket_connect("/ws/mic?token=tok"):
            assert app.state._worker_task is not None
            task = app.state._worker_task
    assert task.done() or task.cancelled()
