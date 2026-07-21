"""Durable Content Credentials: TrustMark watermark + Content Credentials Cloud.

Adobe's "durable" credentials survive metadata stripping: an invisible
TrustMark watermark carries a soft-binding ID that can be used to look the
manifest back up in the Content Credentials Cloud. This module:

1. decodes the TrustMark watermark (requires the optional `trustmark`
   package and its model weights, fetched from Adobe on first use), and
2. queries a manifest-recovery endpoint with the decoded ID.

Both steps degrade gracefully — every unavailability is reported as a note,
never an exception. The endpoint defaults to Adobe's public manifest
repository and can be overridden with AIDETECT_CC_CLOUD_ENDPOINT (used by
tests and by deployments with their own manifest store).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "https://cai-manifests.adobe.com/manifests/"
ENDPOINT_ENV = "AIDETECT_CC_CLOUD_ENDPOINT"

_tm_instance = None
_tm_error: str | None = None


def _get_trustmark():
    """Lazily construct a TrustMark decoder; remember why it failed."""
    global _tm_instance, _tm_error
    if _tm_instance is not None or _tm_error is not None:
        return _tm_instance
    try:
        from trustmark import TrustMark
    except ImportError:
        _tm_error = (
            "trustmark package not installed (pip install trustmark) — "
            "watermark decoding skipped."
        )
        return None
    try:
        _tm_instance = TrustMark(verbose=False, model_type="Q")
        # Model files are fetched on first use; a blocked network leaves the
        # object half-built (encoder/decoder None).
        if getattr(_tm_instance, "decoder", None) is None:
            _tm_instance = None
            raise RuntimeError("model weights unavailable")
    except Exception as exc:
        _tm_error = (
            f"TrustMark models could not be loaded ({exc}) — "
            "watermark decoding skipped."
        )
        _tm_instance = None
    return _tm_instance


def decode_trustmark(path: str) -> tuple[str | None, str | None]:
    """Return (soft_binding_id, note). ID is None when absent/undecodable."""
    tm = _get_trustmark()
    if tm is None:
        return None, _tm_error
    try:
        from PIL import Image

        with Image.open(path) as img:
            secret, present, _version = tm.decode(img.convert("RGB"))
        if present and secret:
            return str(secret).strip(), None
        return None, "No TrustMark watermark detected in the image."
    except Exception as exc:
        return None, f"TrustMark decode failed: {exc}"


def cloud_endpoint() -> str:
    return os.environ.get(ENDPOINT_ENV, DEFAULT_ENDPOINT)


def query_manifest_cloud(soft_binding_id: str,
                         endpoint: str | None = None) -> tuple[dict | None, str | None]:
    """Fetch a recovered manifest store for a soft-binding ID.

    Expects the endpoint to serve JSON (a C2PA manifest-store shaped dict)
    at <endpoint><id>. Returns (store, note).
    """
    base = endpoint or cloud_endpoint()
    if not base.endswith("/"):
        base += "/"
    url = base + urllib.request.quote(soft_binding_id)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            store = json.loads(resp.read().decode("utf-8"))
        if isinstance(store, dict) and store.get("manifests"):
            return store, None
        return None, f"Content Credentials Cloud returned no manifest for id {soft_binding_id!r}."
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, f"No recovered manifest found in the cloud for id {soft_binding_id!r}."
        return None, f"Content Credentials Cloud query failed: HTTP {exc.code}."
    except Exception as exc:
        return None, f"Content Credentials Cloud unreachable ({exc})."


def recover_credentials(path: str,
                        endpoint: str | None = None) -> tuple[dict | None, list[str]]:
    """Full durable-credential recovery: watermark -> cloud manifest store.

    Returns (manifest_store_or_None, notes).
    """
    notes: list[str] = []
    sb_id, note = decode_trustmark(path)
    if note:
        notes.append(note)
    if not sb_id:
        return None, notes
    notes.append(f"TrustMark watermark decoded: soft-binding id {sb_id!r}.")
    store, note = query_manifest_cloud(sb_id, endpoint)
    if note:
        notes.append(note)
    if store:
        notes.append(
            "Manifest recovered from Content Credentials Cloud "
            "(durable credential) despite missing embedded metadata."
        )
    return store, notes
