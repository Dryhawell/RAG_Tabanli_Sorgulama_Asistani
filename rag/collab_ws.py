"""WebSocket tabanlı gerçek zamanlı işbirlikçi not senkronu."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from rag.collab_notes import load_note, save_note
from rag.collab_presence import (
    bind_client,
    presence_entry,
    presence_list,
    release_client,
    update_cursor,
)

try:
    from app.config import ENABLE_COLLAB_CRDT, ENABLE_COLLAB_RICHTEXT
except ImportError:
    ENABLE_COLLAB_CRDT = True
    ENABLE_COLLAB_RICHTEXT = True

if ENABLE_COLLAB_CRDT:
    from rag.collab_crdt import apply_text_edit, load_crdt, merge_remote_ops
    from rag.collab_undo import apply_text_edit_with_undo, redo_edit, undo_edit
    from rag.collab_ime import apply_ime_commit, apply_paste
else:
    apply_text_edit_with_undo = None  # type: ignore
    redo_edit = None  # type: ignore
    undo_edit = None  # type: ignore
    apply_paste = None  # type: ignore
    apply_ime_commit = None  # type: ignore

if ENABLE_COLLAB_RICHTEXT:
    from rag.collab_richtext import apply_rich_ops, richtext_snapshot
else:
    apply_rich_ops = None  # type: ignore
    richtext_snapshot = None  # type: ignore

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


def _attach_rich(workspace_key: str, payload: dict) -> dict:
    if ENABLE_COLLAB_RICHTEXT and richtext_snapshot is not None:
        payload["rich"] = richtext_snapshot(workspace_key)
    return payload


async def _broadcast_mentions(
    workspace_key: str,
    events: List[dict],
    *,
    exclude=None,
) -> None:
    for ev in events or []:
        await _broadcast(
            workspace_key,
            {"op": "mention_notify", "notification": ev},
            exclude=exclude,
        )


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
                meta = bind_client(websocket, workspace_key, username)
                if ENABLE_COLLAB_CRDT:
                    crdt = load_crdt(workspace_key)
                    content = crdt.materialize()
                    revision = crdt.revision
                else:
                    note = load_note(workspace_key)
                    content = note.content
                    revision = note.revision
                snap_payload = {
                    "op": "snapshot",
                    "workspace_key": workspace_key,
                    "revision": revision,
                    "content": content,
                    "crdt": ENABLE_COLLAB_CRDT,
                    "richtext": ENABLE_COLLAB_RICHTEXT,
                    "presence": presence_list(room.clients),
                }
                if ENABLE_COLLAB_RICHTEXT and richtext_snapshot is not None:
                    snap_payload["rich"] = richtext_snapshot(workspace_key)
                await websocket.send(json.dumps(snap_payload, ensure_ascii=False))
                await _broadcast(
                    workspace_key,
                    {"op": "presence_join", "user": presence_entry(meta)},
                    exclude=websocket,
                )
                continue

            if op == "cursor":
                if not workspace_key:
                    continue
                pos = int(data.get("cursor") or data.get("pos") or 0)
                sel_end = data.get("selection_end")
                updated = update_cursor(
                    websocket,
                    cursor=pos,
                    selection_end=int(sel_end) if sel_end is not None else None,
                )
                if updated:
                    await _broadcast(
                        workspace_key,
                        {"op": "presence_update", "user": presence_entry(updated)},
                        exclude=websocket,
                    )
                continue

            if op in {"undo", "redo"} and ENABLE_COLLAB_CRDT:
                if not workspace_key:
                    await websocket.send(
                        json.dumps({"op": "error", "message": "Önce join gönderin"})
                    )
                    continue
                user = data.get("username") or username
                if op == "undo":
                    saved_crdt = undo_edit(workspace_key, author=user)
                else:
                    saved_crdt = redo_edit(workspace_key, author=user)
                payload = _attach_rich(
                    workspace_key,
                    {
                        "op": "sync",
                        "workspace_key": workspace_key,
                        "revision": saved_crdt.revision,
                        "content": saved_crdt.materialize(),
                        "updated_by": saved_crdt.updated_by,
                        "crdt": True,
                        "via": op,
                    },
                )
                await websocket.send(json.dumps(payload, ensure_ascii=False))
                await _broadcast(workspace_key, payload, exclude=websocket)
                continue

            if op in {"paste", "ime_commit"} and ENABLE_COLLAB_CRDT:
                if not workspace_key:
                    await websocket.send(
                        json.dumps({"op": "error", "message": "Önce join gönderin"})
                    )
                    continue
                user = data.get("username") or username
                start = int(data.get("start") or 0)
                end = int(data.get("end") or start)
                text = str(data.get("text") or "")
                if op == "paste":
                    saved_crdt = apply_paste(
                        workspace_key, start=start, end=end, text=text, author=user
                    )
                else:
                    saved_crdt = apply_ime_commit(
                        workspace_key, start=start, end=end, text=text, author=user
                    )
                payload = _attach_rich(
                    workspace_key,
                    {
                        "op": "sync",
                        "workspace_key": workspace_key,
                        "revision": saved_crdt.revision,
                        "content": saved_crdt.materialize(),
                        "updated_by": saved_crdt.updated_by,
                        "crdt": True,
                        "via": op,
                    },
                )
                await websocket.send(json.dumps(payload, ensure_ascii=False))
                await _broadcast(workspace_key, payload, exclude=websocket)
                continue

            if op == "crdt_ops":
                if not workspace_key:
                    await websocket.send(
                        json.dumps({"op": "error", "message": "Önce join gönderin"})
                    )
                    continue
                if not ENABLE_COLLAB_CRDT:
                    await websocket.send(
                        json.dumps({"op": "error", "message": "CRDT kapalı"})
                    )
                    continue
                ops = data.get("ops") or []
                user = data.get("username") or username
                saved = merge_remote_ops(
                    workspace_key,
                    ops,
                    author=user,
                )
                payload = _attach_rich(
                    workspace_key,
                    {
                        "op": "sync",
                        "workspace_key": workspace_key,
                        "revision": saved.revision,
                        "content": saved.materialize(),
                        "updated_by": saved.updated_by,
                        "crdt": True,
                    },
                )
                await websocket.send(json.dumps(payload, ensure_ascii=False))
                await _broadcast(workspace_key, payload, exclude=websocket)
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
                if ENABLE_COLLAB_CRDT:
                    saved_crdt = apply_text_edit_with_undo(
                        workspace_key,
                        content,
                        author=user,
                    )
                    payload = _attach_rich(
                        workspace_key,
                        {
                            "op": "sync",
                            "workspace_key": workspace_key,
                            "revision": saved_crdt.revision,
                            "content": saved_crdt.materialize(),
                            "updated_by": saved_crdt.updated_by,
                            "crdt": True,
                        },
                    )
                    await websocket.send(json.dumps(payload, ensure_ascii=False))
                    await _broadcast(workspace_key, payload, exclude=websocket)
                    continue
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

            if op in {"rich_ops", "richtext"} and ENABLE_COLLAB_RICHTEXT:
                if not workspace_key:
                    await websocket.send(
                        json.dumps({"op": "error", "message": "Önce join gönderin"})
                    )
                    continue
                if apply_rich_ops is None:
                    await websocket.send(
                        json.dumps({"op": "error", "message": "Richtext kapalı"})
                    )
                    continue
                user = data.get("username") or username
                ops = data.get("ops") or []
                if not ops and data.get("type"):
                    ops = [data]
                snap = apply_rich_ops(workspace_key, ops, author=user)
                payload = {
                    "op": "rich_sync",
                    "workspace_key": workspace_key,
                    "rich": snap,
                    "updated_by": user,
                }
                await websocket.send(json.dumps(payload, ensure_ascii=False))
                await _broadcast(workspace_key, payload, exclude=websocket)
                await _broadcast_mentions(
                    workspace_key,
                    snap.get("notifications") or [],
                    exclude=websocket,
                )
                continue

            if op == "ping":
                await websocket.send(json.dumps({"op": "pong"}))
                continue

            await websocket.send(
                json.dumps({"op": "error", "message": f"Bilinmeyen op: {op}"})
            )
    finally:
        if workspace_key:
            room = _room(workspace_key)
            room.clients.discard(websocket)
            released = release_client(websocket)
            if released:
                try:
                    await _broadcast(
                        workspace_key,
                        {"op": "presence_leave", "user": presence_entry(released)},
                    )
                except Exception:
                    pass


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
