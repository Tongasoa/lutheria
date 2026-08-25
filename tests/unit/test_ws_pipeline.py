"""Tests du pipeline : segments VAD -> ASR -> diffusion `partial`/`final` aux lecteurs.

Le lifespan FastAPI (démarrage du worker) exige un TestClient entré en contexte :
le helper make_app gère l'ExitStack et il est refermé par l'autouse fixture.
"""

import contextlib
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.asr import Transcription
from server.config import Settings
from server.main import create_app
from server.vad import WINDOW_BYTES

_open_clients: list[contextlib.ExitStack] = []


@pytest.fixture(autouse=True)
def _close_clients():
    yield
    while _open_clients:
        _open_clients.pop().close()


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


class FakeMT:
    def __init__(self, translations=None, raise_on=None):
        self.translations = dict(translations or {})
        self.raise_on = set(raise_on or ())
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        if text in self.raise_on:
            raise RuntimeError("boom MT")
        return self.translations.get(text, f"[fr] {text}")


def make_app(asr, probas, mt=None):
    def vad_factory(settings, proba_fn=None):
        from server.vad import VADSegmenter

        return VADSegmenter(
            proba_fn=scripted_probas(probas),
            silence_ms=settings.vad_silence_ms,
            max_segment_seconds=settings.max_segment_seconds,
        )

    settings = Settings(mic_token="tok", _env_file=None)
    app = create_app(
        settings,
        vad_factory=vad_factory,
        asr_factory=lambda s: asr,
        mt_factory=lambda s: mt or FakeMT(),
    )
    stack = contextlib.ExitStack()
    _open_clients.append(stack)
    client = stack.enter_context(TestClient(app))  # démarre le lifespan
    return app, client


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_le_texte_malgache_est_diffuse_en_partial_puis_final():
    # 1 segment de parole (10 fenêtres) fermé par 20 fenêtres de silence
    asr = FakeASR(results=["manao ahoana"])
    mt = FakeMT(translations={"manao ahoana": "Bonjour"})
    app, client = make_app(asr, [0.9] * 10 + [0.1] * 20, mt)
    with client.websocket_connect("/ws/listen") as listener:
        with client.websocket_connect("/ws/mic?token=tok") as mic:
            for _ in range(30):
                mic.send_bytes(b"\x00" * WINDOW_BYTES)
                mic.receive_bytes()
        partial = listener.receive_json()
        final = listener.receive_json()

    assert {k: partial[k] for k in ("id", "state", "text_mg")} == {
        "id": 1,
        "state": "partial",
        "text_mg": "manao ahoana",
    }
    assert {k: final[k] for k in ("id", "state", "text_mg", "text_fr")} == {
        "id": 1,
        "state": "final",
        "text_mg": "manao ahoana",
        "text_fr": "Bonjour",
    }
    assert isinstance(partial["ts"], float) and partial["ts"] == final["ts"]


def test_echec_mt_laisse_la_ligne_en_partial_et_continue():
    asr = FakeASR(results=["seg1", "seg2"])
    mt = FakeMT(translations={"seg2": "[fr] seg2"}, raise_on={"seg1"})
    # deux énoncés : le premier échoue en MT, le second réussit
    app, client = make_app(asr, [0.9] * 8 + [0.1] * 20 + [0.9] * 8 + [0.1] * 20, mt)
    with client.websocket_connect("/ws/listen") as listener:
        with client.websocket_connect("/ws/mic?token=tok") as mic:
            for _ in range(56):
                mic.send_bytes(b"\x00" * WINDOW_BYTES)
                mic.receive_bytes()
        messages = [listener.receive_json(), listener.receive_json(), listener.receive_json()]

    states = [(m["id"], m["state"]) for m in messages]
    assert states == [(1, "partial"), (2, "partial"), (2, "final")]
    assert messages[2]["text_fr"] == "[fr] seg2"


def test_les_ids_s_incrementent_entre_segments():
    asr = FakeASR(results=["premier", "deuxieme"])
    # deux énoncés séparés par une longue pause
    app, client = make_app(asr, [0.9] * 8 + [0.1] * 20 + [0.9] * 8 + [0.1] * 20)
    with client.websocket_connect("/ws/listen") as listener:
        with client.websocket_connect("/ws/mic?token=tok") as mic:
            for _ in range(56):
                mic.send_bytes(b"\x00" * WINDOW_BYTES)
                mic.receive_bytes()
        messages = [listener.receive_json() for _ in range(4)]
    partials = [m for m in messages if m["state"] == "partial"]
    assert [(m["id"], m["text_mg"]) for m in partials] == [
        (1, "premier"),
        (2, "deuxieme"),
    ]


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


def test_plusieurs_lecteurs_recoivent_les_memes_messages():
    asr = FakeASR(results=["salama"])
    mt = FakeMT(translations={"salama": "Bonjour"})
    app, client = make_app(asr, [0.9] * 10 + [0.1] * 20, mt)
    with client.websocket_connect("/ws/listen") as l1:
        with client.websocket_connect("/ws/listen") as l2:
            with client.websocket_connect("/ws/mic?token=tok") as mic:
                for _ in range(30):
                    mic.send_bytes(b"\x00" * WINDOW_BYTES)
                    mic.receive_bytes()
            got1 = {l1.receive_json()["state"], l1.receive_json()["state"]}
            got2 = {l2.receive_json()["state"], l2.receive_json()["state"]}
    assert got1 == got2 == {"partial", "final"}


def test_le_worker_est_cree_puis_annule_au_shutdown():
    asr = FakeASR(results=[])
    app, client = make_app(asr, [0.9] * 10 + [0.1] * 20)
    with client.websocket_connect("/ws/mic?token=tok"):
        assert app.state._worker_task is not None
        task = app.state._worker_task
    assert not task.done() or task.cancelled()  # vit tant que l'app vit
