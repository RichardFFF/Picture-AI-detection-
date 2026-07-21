"""Wrapper around the optional c2pa-python library.

Returns the parsed manifest store as a dict (plus validation state), or None
when the library is missing or the file has no manifest. Tolerates both the
modern Reader API (>= 0.32) and the legacy read_file function.
"""
from __future__ import annotations

import json

try:
    import c2pa  # type: ignore
except Exception:  # pragma: no cover - import failure path
    c2pa = None  # type: ignore


def library_available() -> bool:
    return c2pa is not None


def read_manifest_store(path: str) -> tuple[dict | None, str | None]:
    """Return (manifest_store_dict, validation_state) or (None, None)."""
    if c2pa is None:
        return None, None

    # Modern API: Reader(path)
    if hasattr(c2pa, "Reader"):
        try:
            reader = c2pa.Reader(path)
        except Exception:
            return None, None
        try:
            store = json.loads(reader.json())
            state = None
            try:
                state = str(reader.get_validation_state())
            except Exception:
                pass
            return store, state
        except Exception:
            return None, None
        finally:
            try:
                reader.close()
            except Exception:
                pass

    # Legacy API
    if hasattr(c2pa, "read_file"):
        try:
            store = json.loads(c2pa.read_file(path, None))
            return store, None
        except Exception:
            return None, None

    return None, None
