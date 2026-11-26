"""A content-addressed file store on disk (the Rift Apart Blender export's unit textures: nwo.games.rcra.blender_export).

Kept out of nwo.viewer.textures, whose source is part of the decoded-texture cache key."""
import hashlib
import os
import uuid

from nwo.viewer.textures import TextureDiskCache


class BlobDiskCache(TextureDiskCache):
    """Files' bytes under ``folder``, keyed by their SHA-256 (so equal files are stored once). A file whose bytes do not
    hash to its key is ignored. Writes and least-recently-used eviction as in TextureDiskCache."""

    SUFFIX = '.blob'

    def get(self, key):
        path = self._file(key)
        try:
            with open(path, 'rb') as f:
                blob = f.read()
            if hashlib.sha256(blob).hexdigest() != key:
                return None
            os.utime(path)   # recency for eviction
        except OSError:
            return None
        return blob

    def put(self, key, blob):
        path = self._file(key)
        if path.is_file():
            return
        temp = path.with_name(f'{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp, 'wb') as f:
                f.write(blob)
            os.replace(temp, path)
        except OSError:
            try:
                temp.unlink()
            except OSError:
                pass
            return
        self._grow(len(blob))
