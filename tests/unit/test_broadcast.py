"""Tests du hub de broadcast (1 producteur -> N lecteurs)."""

import asyncio

import pytest

from server.broadcast import BroadcastHub


@pytest.fixture
def hub():
    return BroadcastHub()


class FakeWS:
    """WebSocket factice pour tests unitaires."""

    def __init__(self):
        self.sent = []

    async def send_json(self, msg):
        self.sent.append(msg)


@pytest.mark.asyncio
async def test_publish_to_no_listener_is_noop(hub):
    await hub.publish({"id": 1, "text_fr": "salut"})


@pytest.mark.asyncio
async def test_publish_reaches_all_listeners(hub):
    a, b = FakeWS(), FakeWS()
    hub.add_listener(a)
    hub.add_listener(b)
    await hub.publish({"id": 1})
    assert a.sent == [{"id": 1}]
    assert b.sent == [{"id": 1}]


@pytest.mark.asyncio
async def test_remove_listener(hub):
    a = FakeWS()
    hub.add_listener(a)
    hub.remove_listener(a)
    await hub.publish({"id": 1})
    assert a.sent == []


@pytest.mark.asyncio
async def test_dead_listener_is_evicted(hub):
    class DeadWS:
        async def send_json(self, msg):
            raise RuntimeError("connexion morte")

    dead, alive = DeadWS(), FakeWS()
    hub.add_listener(dead)
    hub.add_listener(alive)
    await hub.publish({"id": 1})
    assert alive.sent == [{"id": 1}]
    assert hub.listener_count == 1
