from __future__ import annotations

import json
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any


def _get_fallback_path_for_scratch(path: Path) -> Path:
    """Get fallback path for /scratch paths.
    
    If the path is under /scratch and cannot be created, return an equivalent
    path under /tmp.
    
    Parameters
    ----------
    path:
        Original path that may be under /scratch.
    
    Returns
    -------
    Path
        Fallback path under /tmp if original is /scratch, else original.
    """
    path_str = str(path)
    if path_str.startswith("/scratch"):
        relative_part = path_str[len("/scratch"):]
        fallback = Path(tempfile.gettempdir()) / "prefadap_scratch" / relative_part.lstrip("/")
        return fallback
    return path


def _ensure_directory(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist.
    
    If the directory cannot be created due to permission errors under /scratch,
    automatically redirects to a fallback path under /tmp.
    
    Parameters
    ----------
    path:
        Directory path to create.
    
    Returns
    -------
    Path
        The actual path that was created (may be a fallback path).
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except (PermissionError, FileNotFoundError, OSError) as e:
        if "/scratch" in str(path):
            fallback = _get_fallback_path_for_scratch(path)
            warnings.warn(
                f"Cannot create directory {path} ({e}). Using fallback: {fallback}",
                RuntimeWarning,
                stacklevel=3,
            )
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        raise


def _cleanup_stale_json_tempfiles(
    path: Path | str,
    *,
    older_than_seconds: int = 3600,
) -> int:
    """Remove stale temporary files for a given JSON target.

    Stale temp files can be left behind if a process crashes in the middle of
    an atomic write. This function looks for files following our naming scheme
    and removes those older than ``older_than_seconds``.

    Parameters
    ----------
    path:
        Target JSON path whose temp files should be cleaned up.
    older_than_seconds:
        Minimum age in seconds for a temp file to be considered stale.

    Returns
    -------
    int
        Number of files removed.
    """

    p = Path(path)
    parent = p.parent
    if not parent.exists():
        return 0

    now = time.time()
    removed = 0
    prefix = f".{p.name}."
    suffix = ".tmp"

    for entry in parent.iterdir():
        try:
            if not entry.is_file():
                continue
            name = entry.name
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            # Only remove files older than the threshold
            mtime = entry.stat().st_mtime
            if now - mtime >= max(0, int(older_than_seconds)):
                entry.unlink(missing_ok=True)
                removed += 1
        except Exception:
            # Best-effort cleanup; ignore failures
            continue

    # Also handle legacy pattern where a temp file was created with
    # "<path>.tmp". Only remove a regular file (never a directory).
    legacy = p.with_suffix(p.suffix + ".tmp")
    try:
        if legacy.exists() and legacy.is_file():
            mtime = legacy.stat().st_mtime
            if now - mtime >= max(0, int(older_than_seconds)):
                legacy.unlink(missing_ok=True)
                removed += 1
    except Exception:
        pass

    return removed


def write_json_atomic(
    path: Path | str,
    data: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool | None = None,
    ensure_ascii: bool | None = None,
    cleanup_stale: bool = True,
    cleanup_older_than_seconds: int = 3600,
) -> None:
    """Atomically write JSON to ``path`` using a unique temporary file.

    This implementation avoids collisions with concurrently running processes by
    using ``tempfile.NamedTemporaryFile(delete=False)`` inside the target
    directory and then atomically replacing the destination with ``os.replace``.

    The function optionally performs best-effort cleanup of any stale temp files
    from previous crashed attempts.

    Parameters
    ----------
    path:
        Destination JSON file path.
    data:
        JSON-serialisable object to write.
    indent:
        Indentation level passed to ``json.dump`` (default: 2).
    sort_keys:
        Whether to sort keys in output JSON.
    ensure_ascii:
        Whether to escape non-ASCII characters.
    cleanup_stale:
        If True, remove stale temp files from previous runs.
    cleanup_older_than_seconds:
        Only remove temp files older than this threshold.
    """

    p = Path(path)
    actual_parent = _ensure_directory(p.parent)
    
    # If the parent was redirected to a fallback, update the target path too
    if actual_parent != p.parent:
        p = actual_parent / p.name

    if cleanup_stale:
        _cleanup_stale_json_tempfiles(p, older_than_seconds=cleanup_older_than_seconds)

    # Create a uniquely named hidden temp file in the same directory.
    # Using a dot prefix keeps it hidden and makes it easy to match for cleanup.
    prefix = f".{p.name}."
    suffix = ".tmp"

    # Use delete=False so we can close the handle before calling os.replace
    # (important for Windows compatibility, harmless on POSIX).
    tmp_name = None
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(p.parent),
        delete=False,
        prefix=prefix,
        suffix=suffix,
    ) as tmp:
        tmp_name = tmp.name
        json.dump(
            data,
            tmp,
            indent=indent,
            sort_keys=sort_keys if sort_keys is not None else False,
            ensure_ascii=ensure_ascii if ensure_ascii is not None else False,
        )
        tmp.flush()
        os.fsync(tmp.fileno())

    assert tmp_name is not None
    # Atomic replace into final destination
    os.replace(tmp_name, p)

    # Best-effort cleanup of other stale files (again) in case many runs were
    # interrupted previously. Keep this lightweight.
    if cleanup_stale:
        _cleanup_stale_json_tempfiles(p, older_than_seconds=cleanup_older_than_seconds)


__all__ = ["write_json_atomic"]
