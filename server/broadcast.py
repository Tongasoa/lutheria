"""Hub de broadcast : publie les messages du pipeline vers tous les lecteurs."""


class BroadcastHub:
    def __init__(self) -> None:
        self._listeners: set = set()

    def add_listener(self, ws) -> None:
        self._listeners.add(ws)

    def remove_listener(self, ws) -> None:
        self._listeners.discard(ws)

    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    async def publish(self, message: dict) -> None:
        dead = []
        for ws in list(self._listeners):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._listeners.discard(ws)
