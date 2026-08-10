"""WebSocket işbirlikçi not testleri."""

import asyncio
import json
import threading
import time

import pytest

from rag.collab_ws import _serve, websockets_available


@pytest.mark.skipif(not websockets_available(), reason="websockets yok")
def test_collab_ws_join_and_edit(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.collab_notes.METADATA_DIR", str(tmp_path))
    monkeypatch.setattr("rag.collab_crdt.METADATA_DIR", str(tmp_path))
    port = 18766
    workspace = "ws-test-room"

    def _run_server():
        asyncio.run(_serve("127.0.0.1", port))

    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()
    time.sleep(0.4)

    async def _client_flow():
        import websockets

        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(
                json.dumps(
                    {
                        "op": "join",
                        "workspace_key": workspace,
                        "username": "alice",
                    }
                )
            )
            snap = json.loads(await ws.recv())
            assert snap["op"] == "snapshot"
            assert snap["revision"] == 0

            await ws.send(
                json.dumps(
                    {
                        "op": "edit",
                        "revision": 0,
                        "content": "Canlı not",
                        "username": "alice",
                    }
                )
            )
            sync = json.loads(await ws.recv())
            assert sync["op"] == "sync"
            assert sync["content"] == "Canlı not"
            assert sync["revision"] == 1

    asyncio.run(_client_flow())
