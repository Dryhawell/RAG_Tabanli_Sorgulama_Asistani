"""GitHub Actions secret güncelleme (VAPID) — GH_PAT gerekir (GITHUB_TOKEN yetmez)."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from typing import Any, Dict, Optional, Tuple


def parse_repo_slug(repo: Optional[str] = None) -> Tuple[str, str]:
    raw = (repo or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if "/" not in raw:
        raise ValueError("repo slug gerekli (owner/name) veya GITHUB_REPOSITORY")
    owner, name = raw.split("/", 1)
    return owner.strip(), name.strip()


def _gh_secret_set(name: str, value: str, *, repo: str, env_token: str) -> bool:
    """gh secret set — GH_PAT veya GITHUB_TOKEN (PAT tercih)."""
    env = os.environ.copy()
    env["GH_TOKEN"] = env_token
    env["GITHUB_TOKEN"] = env_token
    try:
        proc = subprocess.run(
            [
                "gh",
                "secret",
                "set",
                name,
                "--repo",
                repo,
                "--body",
                value,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _encrypt_secret_sodium(public_key_b64: str, secret_value: str) -> str:
    """GitHub Actions sealed-box (libsodium)."""
    try:
        from nacl import encoding, public  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyNaCl gerekli: pip install pynacl (veya gh CLI kullanın)"
        ) from exc
    pub = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder())
    sealed = public.SealedBox(pub).encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(sealed).decode("utf-8")


def put_actions_secret(
    *,
    token: str,
    owner: str,
    repo: str,
    name: str,
    value: str,
) -> bool:
    """REST: GET public-key → PUT encrypted secret."""
    try:
        import requests
    except ImportError:
        return False
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base = f"https://api.github.com/repos/{owner}/{repo}/actions/secrets"
    try:
        kr = requests.get(f"{base}/public-key", headers=headers, timeout=20)
        if kr.status_code >= 400:
            return False
        key_info = kr.json()
        key_id = key_info.get("key_id")
        key = key_info.get("key")
        if not key_id or not key:
            return False
        encrypted = _encrypt_secret_sodium(str(key), value)
        pr = requests.put(
            f"{base}/{name}",
            headers=headers,
            json={"encrypted_value": encrypted, "key_id": key_id},
            timeout=20,
        )
        return pr.status_code in {201, 204}
    except Exception:
        return False


def update_vapid_github_secrets(
    *,
    public: str,
    private: Optional[str] = None,
    subject: Optional[str] = None,
    dry_run: bool = False,
    token: Optional[str] = None,
    repo: Optional[str] = None,
    include_private: bool = False,
) -> Dict[str, Any]:
    """
    RAG_NOTIFY_PUSH_VAPID_PUBLIC (+ opsiyonel PRIVATE/SUBJECT) secret güncelle.
    Token: GH_PAT / VAPID_GH_PAT (secrets:write). Default GITHUB_TOKEN yetmez.
    """
    tok = (
        token
        or os.environ.get("GH_PAT", "").strip()
        or os.environ.get("VAPID_GH_PAT", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )
    slug = (repo or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    result: Dict[str, Any] = {
        "dry_run": dry_run,
        "repo": slug or None,
        "updated": [],
        "failed": [],
        "skipped": [],
        "method": None,
    }
    if not public.strip():
        result["failed"].append("public_empty")
        return result
    if not tok:
        result["failed"].append("no_token")
        return result
    if not slug:
        result["failed"].append("no_repo")
        return result

    pairs = [("RAG_NOTIFY_PUSH_VAPID_PUBLIC", public.strip())]
    if subject and subject.strip():
        pairs.append(("RAG_NOTIFY_PUSH_VAPID_SUBJECT", subject.strip()))
    if include_private and private and private.strip():
        pairs.append(("RAG_NOTIFY_PUSH_VAPID_PRIVATE", private.strip()))
    elif include_private:
        result["skipped"].append("private_missing")

    if dry_run:
        result["updated"] = [n for n, _ in pairs]
        result["method"] = "dry_run"
        return result

    # 1) gh CLI
    all_ok = True
    for name, value in pairs:
        if _gh_secret_set(name, value, repo=slug, env_token=tok):
            result["updated"].append(name)
            result["method"] = "gh"
        else:
            all_ok = False
            break
    if all_ok and result["updated"]:
        return result

    # 2) REST + PyNaCl
    result["updated"] = []
    result["method"] = "api"
    try:
        owner, name = parse_repo_slug(slug)
    except ValueError:
        result["failed"].append("bad_repo")
        return result
    for secret_name, value in pairs:
        ok = put_actions_secret(
            token=tok,
            owner=owner,
            repo=name,
            name=secret_name,
            value=value,
        )
        if ok:
            result["updated"].append(secret_name)
        else:
            result["failed"].append(secret_name)
    return result


def main(argv: Optional[list] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Update GitHub VAPID Actions secrets")
    p.add_argument("--public", default=os.environ.get("RAG_NOTIFY_PUSH_VAPID_PUBLIC", ""))
    p.add_argument("--private", default=os.environ.get("RAG_NOTIFY_PUSH_VAPID_PRIVATE", ""))
    p.add_argument("--subject", default=os.environ.get("RAG_NOTIFY_PUSH_VAPID_SUBJECT", ""))
    p.add_argument("--repo", default=None)
    p.add_argument("--include-private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--from-rotate-json", default=None, help="rotate CLI JSON dosyası")
    args = p.parse_args(argv)

    public = args.public
    private = args.private or None
    subject = args.subject or None
    if args.from_rotate_json:
        with open(args.from_rotate_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        public = str(data.get("public") or public)
        private = data.get("private") or private
        subject = data.get("subject") or subject

    result = update_vapid_github_secrets(
        public=public,
        private=private,
        subject=subject,
        dry_run=bool(args.dry_run),
        repo=args.repo,
        include_private=bool(args.include_private),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("failed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
