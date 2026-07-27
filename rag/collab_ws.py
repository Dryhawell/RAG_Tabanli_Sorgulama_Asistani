"""WebSocket tabanlı gerçek zamanlı işbirlikçi not senkronu."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from rag.collab_notes import load_note, save_note

_ws_started = False
_ws_lock = threading.Lock()


def websockets_available() -> bool:
    try:
        import websockets  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class CollabRoom:
    workspace_key: str
    clients: Set[Any] = field(default_factory=set)


_rooms: Dict[str, CollabRoom] = {}


def _room(workspace_key: str) -> CollabRoom:
    key = (workspace_key or "shared").strip()
    if key not in _rooms:
        _rooms[key] = CollabRoom(workspace_key=key)
    return _rooms[key]


async def _broadcast(workspace_key: str, payload: dict, *, exclude=None) -> None:
    room = _room(workspace_key)
    dead = []
    for ws in room.clients:
        if ws is exclude:
            continue
        try:
            await ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception:
            dead.append(ws)
    for ws in dead:
        room.clients.discard(ws)


async def _handle_client(websocket) -> None:
    workspace_key: Optional[str] = None
    username: Optional[str] = None
    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(
                    json.dumps({"op": "error", "message": "Geçersiz JSON"})
                )
                continue

            op = data.get("op")
            if op == "join":
                workspace_key = str(data.get("workspace_key") or "shared")
                username = data.get("username")
                room = _room(workspace_key)
                room.clients.add(websocket)
                note = load_note(workspace_key)
                await websocket.send(
                    json.dumps(
                        {
                            "op": "snapshot",
                            "workspace_key": workspace_key,
                            "revision": note.revision,
                            "content": note.content,
                            "updated_at": note.updated_at,
                            "updated_by": note.updated_by,
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            if op == "edit":
                if not workspace_key:
                    await websocket.send(
                        json.dumps({"op": "error", "message": "Önce join gönderin"})
                    )
                    continue
                content = str(data.get("content") or "")
                expected = data.get("revision")
                user = data.get("username") or username
                try:
                    saved = save_note(
                        workspace_key,
                        content,
                        username=user,
                        expected_revision=int(expected) if expected is not None else None,
                    )
                except ValueError as exc:
                    current = load_note(workspace_key)
                    await websocket.send(
                        json.dumps(
                            {
                                "op": "conflict",
                                "message": str(exc),
                                "revision": current.revision,
                                "content": current.content,
                                "updated_by": current.updated_by,
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue

                payload = {
                    "op": "sync",
                    "workspace_key": workspace_key,
                    "revision": saved.revision,
                    "content": saved.content,
                    "updated_at": saved.updated_at,
                    "updated_by": saved.updated_by,
                }
                await websocket.send(json.dumps(payload, ensure_ascii=False))
                await _broadcast(workspace_key, payload, exclude=websocket)
                continue

            if op == "ping":
                await websocket.send(json.dumps({"op": "pong"}))
                continue

            await websocket.send(
                json.dumps({"op": "error", "message": f"Bilinmeyen op: {op}"})
            )
    finally:
        if workspace_key:
            _room(workspace_key).clients.discard(websocket)


async def _serve(host: str, port: int) -> None:
    import websockets

    async with websockets.serve(_handle_client, host, port):
        await asyncio.Future()


def _run_server_thread(host: str, port: int) -> None:
    asyncio.run(_serve(host, port))


def ensure_collab_ws_server(
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    enabled: bool = True,
) -> bool:
    """UI/CLI process içinde WebSocket sunucusunu tek sefer başlatır."""
    global _ws_started
    if not enabled:
        return False
    if not websockets_available():
        return False
    with _ws_lock:
        if _ws_started:
            return True
        thread = threading.Thread(
            target=_run_server_thread,
            args=(host, port),
            name="collab-ws",
            daemon=True,
        )
        thread.start()
        _ws_started = True
        return True


def run_collab_ws_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """CLI: bloklayarak WebSocket sunucusu çalıştır."""
    if not websockets_available():
        raise RuntimeError("websockets kurulu değil. pip install websockets")
    print(f"Collab WebSocket dinleniyor: ws://{host}:{port}")
    print("Mesaj: join → edit → sync/conflict")
    asyncio.run(_serve(host, port))
