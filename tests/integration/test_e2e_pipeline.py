"""Test E2E : flux complet avec vrais modèles (Silero + Whisper + NLLB).

Reproduit le protocole exact du client web : PCM 16 kHz envoyé par frames
binaires de ~100 ms sur /ws/mic, réception partial+final sur /ws/listen.
Marque `integration` (lent, télécharge les modèles au premier run).
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server.asr import build_asr
from server.config import Settings
from server.main import build_segmenter, create_app
from server.mt import build_mt

FIXTURES = Path(__file__).parent.parent / "fixtures" / "audio"
CHUNK_BYTES = 3200  # 100 ms de PCM int16 mono 16 kHz (taille des trames du client web)


@pytest.fixture(scope="module")
def settings():
    return Settings(
        mic_token="e2e-token",
        asr_model="tiny",  # rapide ; langue forcée "en" pour la fixture anglaise
        asr_language="en",
        mt_device="cpu",
        _env_file=None,
    )


@pytest.mark.integration
def test_flux_complet_micro_vers_lecteurs(settings):
    app = create_app(settings, vad_factory=build_segmenter,
                     asr_factory=build_asr, mt_factory=build_mt)
    with TestClient(app) as client:  # lifespan : pipeline + worker démarrés
        x, sr = sf.read(FIXTURES / "speech_sample1.flac", dtype="float32", always_2d=True)
        x = x.mean(axis=1)[: sr * 8]  # 8 s suffisent
        if sr != 16000:
            n = int(len(x) * 16000 / sr)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
        pcm = (x[: (len(x) // 1600) * 1600] * 32767).astype(np.int16).tobytes()

        with client.websocket_connect("/ws/listen") as listener:
            with client.websocket_connect("/ws/mic?token=e2e-token") as mic:
                for i in range(0, len(pcm), CHUNK_BYTES):
                    mic.send_bytes(pcm[i : i + CHUNK_BYTES])
                    mic.receive_bytes()  # ack applicatif, comme le client JS

            partial = listener.receive_json()
            final = listener.receive_json()

        assert partial["state"] == "partial"
        assert isinstance(partial["text_mg"], str)
        assert final["id"] == partial["id"]
        assert final["state"] == "final"
        assert isinstance(final["text_fr"], str)
