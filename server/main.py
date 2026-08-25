"""Point d'entrée FastAPI : /ws/mic (producteur authentifié), /ws/listen (lecteurs)."""

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.auth import is_valid_mic_token
from server.broadcast import BroadcastHub
from server.config import Settings
from server.vad import Segment, VADSegmenter

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


def create_app(settings: Settings | None = None, vad_factory=build_segmenter) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="Lutheria", version="0.1.0")
    hub = BroadcastHub()
    app.state.settings = settings
    app.state.hub = hub
    app.state.producer = None
    app.state.segment_queue: asyncio.Queue[Segment] | None = None
    app.state.segmenter: VADSegmenter | None = None

    async def enqueue_segment(segment: Segment) -> None:
        q = app.state.segment_queue
        if q is None:
            return
        if q.full():
            q.get_nowait()  # drop-oldest : on privilégie l'audio récent
        q.put_nowait(segment)

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
        if app.state.segmenter is None:
            app.state.segment_queue = asyncio.Queue(maxsize=SEGMENT_QUEUE_MAXSIZE)
            app.state.segmenter = vad_factory(settings)
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

    return app


# Lancement : uvicorn server.main:create_app --factory --host 127.0.0.1 --port 8000
