"""Point d'entrée FastAPI : /ws/mic (producteur authentifié), /ws/listen (lecteurs).

Pipeline : segments VAD -> file asyncio -> ASR (thread) -> diffusion `partial`.
L'étape 4 ajoutera la traduction MT et le message `final` (patch).
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from server.asr import build_asr
from server.auth import is_valid_mic_token
from server.broadcast import BroadcastHub
from server.config import Settings
from server.mt import build_mt
from server.vad import Segment, VADSegmenter

logger = logging.getLogger("lutheria")

CLOSE_UNAUTHORIZED = 4401
CLOSE_PRODUCER_BUSY = 4409
CLOSE_BAD_FRAME = 4400

SEGMENT_QUEUE_MAXSIZE = 32


def build_segmenter(settings: Settings, proba_fn=None) -> VADSegmenter:
    """Fabrique le segmenteur réel (chargement paresseux de Silero, CPU)."""
    from server.vad_model import get_silero_probas

    return VADSegmenter(
        proba_fn=proba_fn or get_silero_probas(),
        silence_ms=settings.vad_silence_ms,
        max_segment_seconds=settings.max_segment_seconds,
    )


def create_app(
    settings: Settings | None = None,
    vad_factory=build_segmenter,
    asr_factory=build_asr,
    mt_factory=build_mt,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Démarre le pipeline au boot (modèles chauds avant le premier client)."""
        from server.vad_model import get_silero_probas

        app.state.segment_queue = asyncio.Queue(maxsize=SEGMENT_QUEUE_MAXSIZE)
        app.state.segmenter = vad_factory(settings, get_silero_probas())
        app.state.asr = asr_factory(settings)
        app.state.mt = mt_factory(settings)
        task = asyncio.create_task(pipeline_worker())
        app.state._worker_task = task
        yield
        task.cancel()

    app = FastAPI(title="Lutheria", version="0.1.0", lifespan=lifespan)
    hub = BroadcastHub()
    app.state.settings = settings
    app.state.hub = hub
    app.state.producer = None

    async def enqueue_segment(segment: Segment) -> None:
        q = app.state.segment_queue
        if q.full():
            q.get_nowait()  # drop-oldest : on privilégie l'audio récent
        q.put_nowait(segment)

    async def pipeline_worker() -> None:
        """Consomme les segments VAD : ASR -> diffusion `partial`, MT -> `final`."""
        segment_id = 0
        while True:
            segment = await app.state.segment_queue.get()
            segment_id += 1
            try:
                transcription = await asyncio.to_thread(
                    app.state.asr.transcribe, segment.audio
                )
                if not transcription.text.strip():
                    continue  # segment vide (faux positif VAD) : rien à diffuser
                ts = time.time()
                text_mg = transcription.text
                await app.state.hub.publish(
                    {
                        "id": segment_id,
                        "ts": ts,
                        "state": "partial",
                        "text_mg": text_mg,
                    }
                )
                try:
                    text_fr = await asyncio.to_thread(app.state.mt.translate, text_mg)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # la traduction échoue : la ligne reste en `partial`
                    logger.exception("échec MT sur le segment %d", segment_id)
                    continue
                await app.state.hub.publish(
                    {
                        "id": segment_id,
                        "ts": ts,
                        "state": "final",
                        "text_mg": text_mg,
                        "text_fr": text_fr,
                    }
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # un échec ASR ne doit jamais tuer le pipeline ni couper le flux
                logger.exception("échec ASR sur le segment %d", segment_id)

    @app.websocket("/ws/mic")
    async def ws_mic(ws: WebSocket) -> None:
        if not is_valid_mic_token(ws.query_params.get("token"), settings.mic_token):
            await ws.close(code=CLOSE_UNAUTHORIZED)
            return
        if app.state.producer is not None:
            await ws.close(code=CLOSE_PRODUCER_BUSY)
            return
        await ws.accept()
        app.state.producer = ws
        segmenter = app.state.segmenter
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is None:
                    continue  # frames texte inattendues : ignorées
                if len(data) > settings.max_ws_frame_bytes:
                    await ws.close(code=CLOSE_BAD_FRAME)
                    break
                for segment in segmenter.process(data):
                    await enqueue_segment(segment)
                await ws.send_bytes(b"")  # ack applicatif : fenêtre(s) consommée(s)
        finally:
            final = segmenter.flush()
            if final is not None:
                await enqueue_segment(final)
            if app.state.producer is ws:
                app.state.producer = None

    @app.websocket("/ws/listen")
    async def ws_listen(ws: WebSocket) -> None:
        await ws.accept()
        hub.add_listener(ws)
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
        finally:
            hub.remove_listener(ws)

    # pages clients (mic.html / listen.html) servies après les routes WS
    client_dir = Path(__file__).resolve().parent.parent / "client"
    if client_dir.is_dir():
        app.mount("/", StaticFiles(directory=client_dir, html=True), name="client")

    return app


# Lancement : uvicorn server.main:create_app --factory --host 127.0.0.1 --port 8000
