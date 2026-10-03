"""Thread-safe bridge from the recognition worker (plain Python thread) to
connected WebSocket clients (asyncio). The worker calls publish() from its
own thread; a broadcast task running on the FastAPI event loop drains the
queue and fans events out to every connected client.
"""
from __future__ import annotations

import asyncio
import logging
import queue
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=200)
        self._clients: set[Any] = set()

    def publish(self, event: dict[str, Any]) -> None:
        """Call from any thread. Drops the event if the queue is saturated
        (a slow/stuck consumer shouldn't be able to block the recognition
        worker)."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning("Event queue full, dropping event type=%s", event.get("type"))

    def add_client(self, ws) -> None:
        self._clients.add(ws)

    def remove_client(self, ws) -> None:
        self._clients.discard(ws)

    async def run_broadcast_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            event = await loop.run_in_executor(None, self._queue.get)
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_json(event)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)


event_bus = EventBus()
