"""The app's disk cache (``rc2016.games.base.CACHE_ROOT``, ``.cache``): its size and clearing it (Options… > Clear cache).

Everything under it is rebuilt on demand: shader translations, decoded textures, thumbnails, archive indexes and level
data. Settings live elsewhere (%APPDATA%) and are never touched. Clearing renames each entry aside first, so a reader that
opens it next sees nothing rather than a half-deleted folder, then deletes it; an entry another process holds open stays
and is reported.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

from rc2016.games.base import CACHE_ROOT

TRASH_PREFIX = '.clearing-'


def cache_usage(root: Path = CACHE_ROOT) -> tuple[int, int]:
    """(bytes, files) under ``root``; (0, 0) when it does not exist. Unreadable entries are skipped."""
    total = count = 0
    for folder, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.stat(os.path.join(folder, name)).st_size
            except OSError:
                continue
            count += 1
    return total, count


def clear_cache(root: Path = CACHE_ROOT) -> dict:
    """Delete everything under ``root`` (the folder itself stays). Returns {'bytes', 'files', 'left'}: what was removed
    and the entries that could not be (held open by another process, for example); those are tried again next time."""
    root = Path(root)
    if not root.is_dir():
        return dict(bytes=0, files=0, left=[])
    before = cache_usage(root)
    left = []
    for entry in list(root.iterdir()):
        target = entry
        if not entry.name.startswith(TRASH_PREFIX):
            target = root / f'{TRASH_PREFIX}{uuid.uuid4().hex}'
            try:
                entry.rename(target)
            except OSError:
                target = entry   # in use: delete in place, as far as it goes
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        except OSError:
            left.append(entry.name)
    after = cache_usage(root)
    return dict(bytes=before[0] - after[0], files=before[1] - after[1], left=sorted(left))


def size_text(size: int) -> str:
    """``size`` bytes for a label: 0 B, 812 KB, 3.4 GB."""
    value = float(size)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if value < 1024 or unit == 'GB':
            return f'{value:.0f} {unit}' if unit in ('B', 'KB') else f'{value:.1f} {unit}'
        value /= 1024
    return f'{value:.1f} TB'
