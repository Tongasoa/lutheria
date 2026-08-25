"""Tests des pages clients servies par le serveur (montage statique)."""

import pytest
from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app


def test_mic_html_est_serve():
    app = create_app(Settings(mic_token="t", _env_file=None))
    client = TestClient(app)
    r = client.get("/mic.html")
    assert r.status_code == 200
    assert "Démarrer le micro" in r.text


def test_index_est_une_page_daccueil():
    app = create_app(Settings(mic_token="t", _env_file=None))
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "/mic.html" in r.text and "/listen.html" in r.text


def test_listen_html_est_serve():
    app = create_app(Settings(mic_token="t", _env_file=None))
    client = TestClient(app)
    r = client.get("/listen.html")
    assert r.status_code == 200
    assert "traduction en direct" in r.text


def test_les_routes_ws_restent_prioritaires():
    app = create_app(Settings(mic_token="tok", _env_file=None))
    client = TestClient(app)
    # /ws/mic ne doit PAS être intercepté par le mount statique (refus WS, pas du HTML)
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/mic"):  # sans token -> close 4401
            pass
