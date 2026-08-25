"""Tests du squelette WebSocket : auth, frames, unicité du producteur."""

import pytest
from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app


@pytest.fixture
def app():
    settings = Settings(mic_token="tok", _env_file=None)
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as c:  # démarre le lifespan (pipeline + worker)
        yield c


def test_mic_without_token_rejected(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/mic"):
            pass


def test_mic_with_wrong_token_rejected(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/mic?token=nope"):
            pass


def test_mic_with_token_accepted(client):
    with client.websocket_connect("/ws/mic?token=tok") as ws:
        ws.send_bytes(b"\x00" * 64)  # frame PCM valide


def test_listen_without_token_accepted(client):
    with client.websocket_connect("/ws/listen") as ws:
        pass


def test_second_producer_rejected(client):
    with client.websocket_connect("/ws/mic?token=tok"):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/mic?token=tok"):
                pass


def test_oversized_frame_rejected(client):
    with client.websocket_connect("/ws/mic?token=tok") as ws:
        ws.send_bytes(b"\x00" * (65536 + 1))
        msg = ws.receive()
        assert msg["type"] == "websocket.close"
        assert msg["code"] == 4400
