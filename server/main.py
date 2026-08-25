"""Point d'entrée FastAPI : /ws/mic (producteur authentifié), /ws/listen (lecteurs)."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.auth import is_valid_mic_token
from server.broadcast import BroadcastHub
from server.config import Settings

CLOSE_UNAUTHORIZED = 4401
CLOSE_PRODUCER_BUSY = 4409
CLOSE_BAD_FRAME = 4400


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="Lutheria", version="0.1.0")
    hub = BroadcastHub()
    app.state.settings = settings
    app.state.hub = hub
    app.state.producer = None

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
                # Étape 2 : alimenter le pipeline VAD ici.
        finally:
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
