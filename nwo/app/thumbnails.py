"""Model thumbnails: cache index, PNG writer, offscreen renderer and the batch CLI.

The renderer runs as its own process so the browser's Tk thread and 3D view never block and
a batch can be cancelled::

    python -m nwo.app.thumbnails --source <game root> --cache <game cache folder> [--ids-file F | --all] [--stdin]

Thumbnails are written to ``<cache>/thumbnails/<model id>.png`` and described by
``index.json`` in the same folder. JSON lines go to stdout: ``begin`` when a model starts, ``model``
when it is finished. With ``--stdin`` the process stays up: each stdin line ``{"ids": [...]}``
replaces the queue (the browser re-prioritises a running renderer instead of starting a new one),
``idle`` says the queue is empty, and the process ends when stdin closes. Worker processes
(``--workers``) load the next models of the queue ahead of the renderer, without GL, so their
shader translations are in the disk caches when the renderer reaches them.
The pure parts (index, PNG, downsampling, argument handling) import without Tk or OpenGL.
"""
from __future__ import annotations

import argparse
from collections import deque
import ctypes
import json
import os
from pathlib import Path
import queue
import re
import struct
import subprocess
import sys
import threading
import time
import zlib

THUMBNAIL_VERSION = 7     # bump when rendering changes: invalidates every cached thumbnail
DEFAULT_SIZE = 192
SUPERSAMPLE = 2            # render at 2x and box-filter down: the legacy pipeline has no MSAA
# Rift Apart texture reads for thumbnails: the SD chain once its top mip is at least this (else the HD stream), read at
# its largest mip no wider or taller than TEXTURE_MAX. A 384-pixel render needs no more; smaller mips upload and build
# their GL mip chain in a fraction of the time.
TEXTURE_MIN, TEXTURE_MAX = 256, 512
MAX_WORKERS = 4
INDEX_NAME = 'index.json'
STATUSES = ('ok', 'unsupported', 'error')


# Cache layout and index ------------------------------------------------------
def cache_dir(cache_root) -> Path:
    """Folder that holds one game's thumbnails: under the game's cache folder (``Game.cache_dir()``, inside the
    app's own ``.cache``), never the output root or the source folders."""
    return Path(cache_root) / 'thumbnails'


def safe_name(model_id) -> str:
    """File stem for a model id: ids are hex, anything else is reduced to a safe form."""
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(model_id)).strip('_') or 'model'


def write_atomic(path: Path, data: bytes) -> None:
    """Write beside the target, then replace it: readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class ThumbnailIndex:
    """``index.json``: model id -> {source_hash, renderer_version, size, status, reason}."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.models: dict[str, dict] = {}
        self.dirty = False
        self.last_saved = time.monotonic()

    @property
    def path(self) -> Path:
        return self.folder / INDEX_NAME

    @classmethod
    def load(cls, folder) -> 'ThumbnailIndex':
        index = cls(folder)
        try:
            document = json.loads(index.path.read_text(encoding='utf8'))
            models = document.get('models')
            if isinstance(models, dict):
                index.models = {str(k): v for k, v in models.items() if isinstance(v, dict)}
        except (OSError, ValueError, AttributeError):
            pass   # missing or damaged index: everything is simply rendered again
        return index

    def png_path(self, model_id) -> Path:
        return self.folder / f'{safe_name(model_id)}.png'

    def get(self, model_id):
        return self.models.get(model_id)

    def is_current(self, model_id, size, source_hash=None) -> bool:
        """True when the recorded outcome is still valid for this renderer version, size and source.

        ``source_hash`` None skips the source comparison (the browser does not hash archive bytes).
        A recorded success also needs its PNG on disk."""
        entry = self.models.get(model_id)
        if not entry or entry.get('renderer_version') != THUMBNAIL_VERSION or entry.get('size') != size:
            return False
        if source_hash is not None and entry.get('source_hash') != source_hash:
            return False
        if entry.get('status') == 'ok':
            return self.png_path(model_id).is_file()
        return entry.get('status') in STATUSES

    def is_unchanged(self, model_id, size, identity) -> bool:
        """Cheap freshness: current for this version and size, and recorded under the same archive identity.

        No archive bytes are read. An entry without an identity (older index) is never trusted here."""
        entry = self.models.get(model_id)
        return bool(identity) and self.is_current(model_id, size) and entry.get('identity') == identity

    def record(self, model_id, status, source_hash=None, size=DEFAULT_SIZE, reason='', **extra) -> dict:
        if status not in STATUSES:
            raise ValueError(f'Unknown thumbnail status: {status}')
        entry = dict(source_hash=source_hash, renderer_version=THUMBNAIL_VERSION, size=size,
                     status=status, reason=str(reason)[:500], **extra)
        self.models[model_id] = entry
        self.dirty = True
        return entry

    def write_png(self, model_id, data: bytes) -> Path:
        path = self.png_path(model_id)
        write_atomic(path, data)
        return path

    def save(self, force=False) -> None:
        if not (self.dirty or force):
            return
        document = dict(version=THUMBNAIL_VERSION, models=self.models)
        write_atomic(self.path, json.dumps(document, separators=(',', ':')).encode('utf8'))
        self.dirty = False
        self.last_saved = time.monotonic()

    def save_if_due(self, seconds=3.0) -> None:
        if self.dirty and time.monotonic() - self.last_saved >= seconds:
            self.save()


def missing_ids(index: ThumbnailIndex, model_ids, size=DEFAULT_SIZE):
    """Ids (in the given order) that have no valid cached outcome yet."""
    return [m for m in model_ids if not index.is_current(m, size)]


def read_ids_file(path) -> list[str]:
    """Model ids in priority order: one per line, blanks and ``#`` comments ignored, duplicates dropped."""
    seen, ids = set(), []
    for line in Path(path).read_text(encoding='utf8').splitlines():
        text = line.strip()
        if text and not text.startswith('#') and text not in seen:
            seen.add(text)
            ids.append(text)
    return ids


# PNG and downsampling --------------------------------------------------------
def encode_png_rgba(width, height, rgba_rows) -> bytes:
    """RGBA PNG from top-down 8-bit RGBA rows."""
    if len(rgba_rows) != width * height * 4:
        raise ValueError('RGBA buffer does not match the image size')
    stride = width * 4
    raw = b''.join(b'\x00' + rgba_rows[y * stride:(y + 1) * stride] for y in range(height))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def downsample_rgba(rgba_bottom_up, width, height, factor):
    """Box-filter a glReadPixels RGBA buffer (bottom row first) down by ``factor``.

    Colour is averaged weighted by alpha, so transparent background never darkens the edge.
    Returns (width // factor, height // factor, top-down RGBA bytes)."""
    if factor < 1 or width % factor or height % factor:
        raise ValueError('Downsample factor must divide the image size')
    out_w, out_h = width // factor, height // factor
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        image = np.frombuffer(rgba_bottom_up, np.uint8).reshape(height, width, 4)[::-1].astype(np.float32)
        weight = image[..., 3:4]
        premultiplied = np.concatenate([image[..., :3] * weight, weight], axis=2)
        blocks = premultiplied.reshape(out_h, factor, out_w, factor, 4).sum(axis=(1, 3))
        alpha_sum = blocks[..., 3:4]
        rgb = np.where(alpha_sum > 0, blocks[..., :3] / np.maximum(alpha_sum, 1e-6), 0.0)
        alpha = alpha_sum / (factor * factor)
        result = np.concatenate([rgb, alpha], axis=2)
        return out_w, out_h, np.clip(np.rint(result), 0, 255).astype(np.uint8).tobytes()
    out = bytearray(out_w * out_h * 4)
    stride, area = width * 4, factor * factor
    for y in range(out_h):
        for x in range(out_w):
            r = g = b = a = 0
            for dy in range(factor):
                row = (height - 1 - (y * factor + dy)) * stride
                for dx in range(factor):
                    i = row + (x * factor + dx) * 4
                    w = rgba_bottom_up[i + 3]
                    r += rgba_bottom_up[i] * w
                    g += rgba_bottom_up[i + 1] * w
                    b += rgba_bottom_up[i + 2] * w
                    a += w
            o = (y * out_w + x) * 4
            if a:
                out[o:o + 3] = (round(r / a), round(g / a), round(b / a))
            out[o + 3] = round(a / area)
    return out_w, out_h, bytes(out)


def entry_identity(archive, entry):
    """Cheap identity of a model's archive bytes without reading them, or None when it cannot be known.

    The layout identity covers layout.csv and the other metadata; the first occurrence's range and its
    archive file size and mtime catch a rewritten archive. Loose files are not covered (they are hashed)."""
    if not entry or 'loose_path' in entry:
        return None
    occurrence = next((o for o in entry.get('occurrences', ()) if o.get('available')), None)
    if occurrence is None:
        return None
    try:
        stat = os.stat(archive.root / occurrence['archive'])
    except OSError:
        return None
    return f"{archive.layout_identity}|{entry.get('order')}|{occurrence['archive']}|{occurrence['offset']}|{occurrence['size']}|{stat.st_size}|{stat.st_mtime_ns}"


# Command line ----------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(prog='python -m nwo.app.thumbnails', description=__doc__.splitlines()[0])
    parser.add_argument('--source', required=True, type=Path, help='extracted game archive root (read only)')
    parser.add_argument('--cache', required=True, type=Path, help='the game cache folder; thumbnails go to <cache>/thumbnails')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE, help='thumbnail edge in pixels (32-512)')
    parser.add_argument('--ids-file', type=Path, help='model ids to render, one per line, in priority order')
    parser.add_argument('--all', action='store_true', help='render every archive model (after any ids from --ids-file)')
    parser.add_argument('--limit', type=int, default=0, help='stop after this many models (0 = no limit)')
    parser.add_argument('--force', action='store_true', help='render again even when the cache is current')
    parser.add_argument('--no-fbo', action='store_true', help='read the window back buffer instead of an offscreen framebuffer')
    parser.add_argument('--stdin', action='store_true',
                        help='stay up and take new queues as {"ids": [...]} lines on stdin until it closes')
    parser.add_argument('--workers', type=int, default=-1,
                        help='processes that load upcoming models ahead of the renderer (-1 = automatic, 0 = none)')
    parser.add_argument('--skip', action='append', default=[], metavar='ID', help='leave this model out (repeatable)')
    return parser


def parse_arguments(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 32 <= args.size <= 512:
        parser.error('--size must be between 32 and 512')
    if not args.ids_file and not args.all and not args.stdin:
        parser.error('give --ids-file, --all and/or --stdin')
    return args


def select_ids(args, archive_models) -> list[str]:
    """Priority list: ids file first (in file order), then every other archive model sorted by path; --skip ids left out."""
    ordered = read_ids_file(args.ids_file) if args.ids_file else []
    if args.all:
        listed = set(ordered)
        ordered += [model_id for _path, model_id in sorted(archive_models) if model_id not in listed]
    skip = set(getattr(args, 'skip', ()) or ())
    ordered = [model_id for model_id in ordered if model_id not in skip]
    return ordered[:args.limit] if args.limit > 0 else ordered


def worker_count(requested, cpus=None, frozen=None) -> int:
    """Prefetch workers: the requested number (at most MAX_WORKERS), or automatically half the CPUs less one for the
    renderer, at least one. None in a frozen build, where spawned workers would start the app again."""
    frozen = getattr(sys, 'frozen', False) if frozen is None else frozen
    if frozen or requested == 0:
        return 0
    if requested > 0:
        return min(requested, MAX_WORKERS)
    if cpus is None:
        cpus = os.cpu_count() or 2
    return max(1, min(MAX_WORKERS, cpus // 2 - 1))


def parse_control(line):
    """A stdin control line: the new queue (list of ids) from {"ids": [...]}, or None when the line is not one."""
    try:
        message = json.loads(line)
    except (ValueError, TypeError):
        return None
    ids = message.get('ids') if isinstance(message, dict) else None
    if not isinstance(ids, list):
        return None
    seen, ordered = set(), []
    for model_id in ids:
        text = str(model_id).strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def configure_previews():
    """Thumbnail texture reads, in this process and its workers only (the interactive viewer is unaffected): no streamed
    HD mips for R&C 2016; for Rift Apart the SD chain from TEXTURE_MIN, read at a mip of at most TEXTURE_MAX."""
    from nwo.viewer import textures
    textures.MIN_PREVIEW_SIZE = 0
    try:
        from nwo.games.rcra import texture as rcra_texture
    except ImportError:   # a build without the Rift Apart reader
        return
    rcra_texture.MIN_PREVIEW_SIZE, rcra_texture.MAX_PREVIEW_SIZE = TEXTURE_MIN, TEXTURE_MAX


# Offscreen renderer ----------------------------------------------------------
class FramebufferError(RuntimeError):
    pass


_FBO_FUNCTIONS = {
    'GenFramebuffers': (None, [ctypes.c_int, ctypes.c_void_p]),
    'BindFramebuffer': (None, [ctypes.c_uint, ctypes.c_uint]),
    'GenRenderbuffers': (None, [ctypes.c_int, ctypes.c_void_p]),
    'BindRenderbuffer': (None, [ctypes.c_uint, ctypes.c_uint]),
    'RenderbufferStorage': (None, [ctypes.c_uint, ctypes.c_uint, ctypes.c_int, ctypes.c_int]),
    'FramebufferRenderbuffer': (None, [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]),
    'CheckFramebufferStatus': (ctypes.c_uint, [ctypes.c_uint]),
    'DeleteFramebuffers': (None, [ctypes.c_int, ctypes.c_void_p]),
    'DeleteRenderbuffers': (None, [ctypes.c_int, ctypes.c_void_p]),
}


def load_fbo_functions(gl):
    """GL 3 / ARB framebuffer entry points, else the EXT ones; raises when neither set exists."""
    if os.name != 'nt':
        raise FramebufferError('WGL only')
    gl.wglGetProcAddress.argtypes = [ctypes.c_char_p]
    gl.wglGetProcAddress.restype = ctypes.c_void_p
    for suffix in ('', 'EXT'):
        functions = {}
        for name, (restype, argtypes) in _FBO_FUNCTIONS.items():
            address = gl.wglGetProcAddress(f'gl{name}{suffix}'.encode())
            if not address or address in (1, 2, 3, ctypes.c_void_p(-1).value):
                break
            functions[name] = ctypes.WINFUNCTYPE(restype, *argtypes)(address)
        else:
            return functions
    raise FramebufferError('framebuffer objects are not available')


class _ClearProxy:
    """The viewport's GL handle, with the clear colour replaced (transparent, or opaque grey)."""

    def __init__(self, gl, clear):
        self._gl, self._clear = gl, clear

    def __getattr__(self, name):
        return getattr(self._gl, name)

    def glClearColor(self, *_ignored):
        self._gl.glClearColor(*self._clear)


class _CaptureSwap:
    """Stands in for gdi32: ``SwapBuffers`` reads the finished frame instead of presenting it."""

    def __init__(self, viewport):
        self.viewport = viewport

    def SwapBuffers(self, _hdc):
        self.viewport.capture_frame()
        return 1


def thumbnail_viewport_class():
    """ModelViewport subclass that draws into an offscreen RGBA framebuffer (imported lazily: needs Tk)."""
    from nwo.viewer.viewport import ModelViewport

    class ThumbnailViewport(ModelViewport):
        GL_FRAMEBUFFER, GL_RENDERBUFFER, GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT = 0x8D40, 0x8D41, 0x8CE0, 0x8D00
        GL_RGBA8, GL_DEPTH_COMPONENT24, GL_FRAMEBUFFER_COMPLETE = 0x8058, 0x81A6, 0x8CD5

        def __init__(self, parent, edge, use_fbo=True, **kwargs):
            super().__init__(parent, width=edge, height=edge, **kwargs)
            self.edge, self.use_fbo = edge, use_fbo
            self.fbo = None
            self.fbo_error = ''
            self.transparent = False
            self.want_capture = False
            self.frame = None

        def _create_context(self):
            super()._create_context()
            gl = self._gl
            gl.glReadBuffer.argtypes = [ctypes.c_uint]
            gl.glReadPixels.argtypes = [ctypes.c_int] * 4 + [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
            gl.glFinish.argtypes = []
            if self.use_fbo:
                try:
                    self._make_fbo()
                except Exception as exc:   # no usable framebuffers: fall back to the window back buffer
                    self.fbo, self.fbo_error = None, str(exc)
            self.transparent = self.fbo is not None
            self._gl = _ClearProxy(gl, (0.0, 0.0, 0.0, 0.0) if self.transparent else (0.14, 0.15, 0.17, 1.0))
            self._gdi32 = _CaptureSwap(self)
            self._make_current()

        def _make_fbo(self):
            gl, edge = self._gl, self.edge
            if self.winfo_width() != edge or self.winfo_height() != edge:
                raise FramebufferError(f'viewport is {self.winfo_width()}x{self.winfo_height()}, expected {edge}x{edge}')
            fn = load_fbo_functions(gl)
            frame, colour, depth = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
            fn['GenFramebuffers'](1, ctypes.byref(frame))
            fn['GenRenderbuffers'](1, ctypes.byref(colour))
            fn['GenRenderbuffers'](1, ctypes.byref(depth))
            fn['BindRenderbuffer'](self.GL_RENDERBUFFER, colour.value)
            fn['RenderbufferStorage'](self.GL_RENDERBUFFER, self.GL_RGBA8, edge, edge)
            fn['BindRenderbuffer'](self.GL_RENDERBUFFER, depth.value)
            fn['RenderbufferStorage'](self.GL_RENDERBUFFER, self.GL_DEPTH_COMPONENT24, edge, edge)
            fn['BindFramebuffer'](self.GL_FRAMEBUFFER, frame.value)
            fn['FramebufferRenderbuffer'](self.GL_FRAMEBUFFER, self.GL_COLOR_ATTACHMENT0, self.GL_RENDERBUFFER, colour.value)
            fn['FramebufferRenderbuffer'](self.GL_FRAMEBUFFER, self.GL_DEPTH_ATTACHMENT, self.GL_RENDERBUFFER, depth.value)
            status = fn['CheckFramebufferStatus'](self.GL_FRAMEBUFFER)
            if status != self.GL_FRAMEBUFFER_COMPLETE:
                fn['BindFramebuffer'](self.GL_FRAMEBUFFER, 0)
                raise FramebufferError(f'framebuffer incomplete (0x{status:X})')
            self.fbo = frame.value
            self._fbo_bind = fn['BindFramebuffer']

        def _make_current(self):
            current = super()._make_current()
            if current and self.fbo is not None:
                self._fbo_bind(self.GL_FRAMEBUFFER, self.fbo)
            return current

        def capture_frame(self):
            if not self.want_capture:
                return
            gl, edge = self._gl, self.edge
            gl.glFinish()
            gl.glReadBuffer(self.GL_COLOR_ATTACHMENT0 if self.fbo is not None else 0x0405)   # 0x0405 = GL_BACK
            gl.glPixelStorei(0x0D05, 1)
            pixels = (ctypes.c_ubyte * (edge * edge * 4))()
            gl.glReadPixels(0, 0, edge, edge, 0x1908, 0x1401, pixels)
            self.frame = bytes(pixels)

        def _draw(self):
            """Only the captured frame is drawn: set_mesh, set_mode, the camera reset, clear and Tk's idle redraws would
            each draw a frame nobody sees."""
            if self.want_capture:
                super()._draw()

        def read_frame(self):
            """Draw the current mesh once and return its RGBA bytes (bottom row first)."""
            self.want_capture, self.frame = True, None
            try:
                self._draw()
            finally:
                self.want_capture = False
            if self.frame is None:
                raise RuntimeError('the frame could not be read back')
            return self.frame

    return ThumbnailViewport


def render_model(view, payload, size, supersample):
    """Set the payload, draw it once and return (png bytes, notes). Raises ValueError when it cannot be drawn."""
    if not view.set_mesh(payload):
        raise ValueError(view._error or 'the mesh could not be displayed')
    view.set_mode('Shaded')
    frame = view.read_frame()
    width = view.edge
    out_w, out_h, rgba = downsample_rgba(frame, width, width, supersample)
    if not any(rgba[3::4]):
        raise RuntimeError('empty render: nothing was drawn')
    notes = [view.shading_error] if view.shading_error else []
    return encode_png_rgba(out_w, out_h, rgba), notes


def emit(**fields):
    print(json.dumps(fields, separators=(',', ':')), flush=True)


# Prefetch workers -------------------------------------------------------------
_WORKER = {}


def _worker_init(source, previews=True):
    """A prefetch worker's loader; ``previews`` switches it to the thumbnail texture reads (configure_previews). The
    viewer's translation workers (viewer.loader.TranslationPool) only translate and leave it off."""
    if previews:
        configure_previews()
    from nwo.viewer.loader import SourcePreviewLoader
    loader = SourcePreviewLoader()
    archive = loader._ensure_archive(source)
    archive.verify_repeats = False
    _WORKER.update(loader=loader, source=source)


def prefetch_units(archive, model_id, path):
    """The prefetch work for one model, as independent units: for Rift Apart each of its materials' translations (several
    workers then share one model: the stadium crowd has 92 materials, a dozen of them about 10 s each), otherwise the whole
    model load."""
    if hasattr(archive, 'ids_of_type'):   # Rift Apart: the material translations dominate a cold load
        try:
            references = archive.asset_references(model_id)
        except (KeyError, ValueError, OSError):
            return []
        keys = []
        for reference in references:
            if reference['path'].endswith('.material'):
                keys += [k for k in archive.resolve(reference['path']) if archive.entries[k]['type'] == 'material']
        return [('material', key) for key in dict.fromkeys(keys)]
    return [('model', (model_id, path))]


def _worker_run(kind, argument):
    """One prefetch unit (prefetch_units) in a worker, without GL: its translations land in the shared disk caches,
    written atomically, so the renderer and the other workers read whole results or none. Returns the seconds it took."""
    loader, source = _WORKER['loader'], _WORKER['source']
    started = time.perf_counter()
    try:
        if kind == 'material':
            from nwo.games.rcra.viewer_materials import translate
            translate(loader._ensure_archive(source), argument)
        elif kind == 'lighting':   # (direct or ambient, class): an engine lighting recipe every lit material shares
            from nwo.games.rcra import lighting
            archive = loader._ensure_archive(source)
            getattr(lighting, argument[0])(archive.root, argument[1], disk=archive.translation_disk_cache)
        else:
            loader.load(source, argument[0], argument[1], 'Highest', 0, None, True)
    except Exception:   # the renderer meets the same failure and records it
        pass
    finally:
        loader.identity = None
        if len(loader.texture_cache) > 400:
            loader.texture_cache.clear()
    return time.perf_counter() - started


class Prefetcher:
    """Worker processes that do the slow, cacheable part of the next models (prefetch_units) while the renderer draws."""

    def __init__(self, source, workers, units=None):
        self.pool = None
        self.futures = {}    # unit -> future
        self.units = {}      # model id -> its units
        self.unit_of = units or (lambda model_id, path: [('model', (model_id, path))])
        self.workers, self.lookahead = workers, 2 * workers
        if workers > 0:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                            initializer=_worker_init, initargs=(str(source),))

    def _units(self, model_id, path):
        if model_id not in self.units:
            self.units[model_id] = self.unit_of(model_id, path)
        return self.units[model_id]

    def plan(self, current, upcoming):
        """Submit the units of ``current`` (the model the renderer is about to load) and of ``upcoming`` ([(model id,
        archive path)], most urgent first), in that order. Queued units no longer wanted are cancelled; running ones
        finish (their results still land in the caches)."""
        if self.pool is None:
            return
        models = ([current] if current else []) + upcoming[:self.lookahead]
        ordered = [unit for model_id, path in models for unit in self._units(model_id, path)]
        wanted = set(ordered)
        for unit in [u for u in self.futures if u not in wanted]:
            self.futures.pop(unit).cancel()
        for unit in ordered:
            if unit not in self.futures:
                self.futures[unit] = self.pool.submit(_worker_run, *unit)
        keep = {model_id for model_id, _path in models}
        self.units = {m: u for m, u in self.units.items() if m in keep}

    def wait(self, model_id):
        """Before the renderer loads ``model_id``: wait for its units (queued ones run on the workers meanwhile, in parallel),
        so the load reads their results from the caches. Returns the seconds waited."""
        units = self.units.pop(model_id, ())
        futures = [self.futures.pop(unit) for unit in units if unit in self.futures]
        if not futures:
            return 0.0
        started = time.perf_counter()
        for future in futures:
            try:
                future.result()
            except Exception:   # cancelled, or a broken pool: the renderer does that part itself
                pass
        return time.perf_counter() - started

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None


def _read_controls(stream, control):
    """Stdin reader thread: each valid control line's queue goes to ``control``; None at end of input."""
    try:
        for raw in stream:
            ids = parse_control(raw.decode('utf8', 'replace') if isinstance(raw, bytes) else raw)
            if ids is not None:
                control.put(ids)
    finally:
        control.put(None)


def run(args) -> int:
    """Render the requested models; returns the process exit code."""
    import tkinter as tk
    from nwo.viewer.loader import SourcePreviewLoader

    configure_previews()
    size, supersample = args.size, SUPERSAMPLE
    folder = cache_dir(args.cache)
    folder.mkdir(parents=True, exist_ok=True)
    index = ThumbnailIndex.load(folder)
    loader = SourcePreviewLoader()
    archive = loader._ensure_archive(args.source)
    prefetch = Prefetcher(args.source, worker_count(args.workers),
                          lambda model_id, path: prefetch_units(archive, model_id, path))
    archive.verify_repeats = False   # hash the first copy only; repeats are identical in a consistent install
    if hasattr(archive, 'ids_of_type'):   # a typed index (Rift Apart): no entry dict for every asset of the install
        models = [(archive.entries[key]['path'], key) for key in archive.ids_of_type('model')]
    else:
        models = [(entry['path'], model_id) for model_id, entry in archive.entries.items() if entry['type'] == 'model']
    skip = set(args.skip)
    pending = deque(select_ids(args, models))
    total = len(pending)
    control = queue.Queue()
    if args.stdin:
        threading.Thread(target=_read_controls, args=(sys.stdin.buffer, control), daemon=True,
                         name='thumbnail-control').start()
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes('-alpha', 0.0)
    edge = size * supersample
    root.geometry(f'{edge}x{edge}+0+0')
    view = thumbnail_viewport_class()(root, edge, use_fbo=not args.no_fbo)
    view.use_game_display(args.source)   # Rift Apart: its decoded ACES display; R&C 2016: the captured one
    view.pack(fill='both', expand=True)
    root.update()
    if view._context_error or view._hrc is None:
        emit(event='fatal', reason=view._context_error or 'no OpenGL context')
        prefetch.close()
        root.destroy()
        return 2
    emit(event='start', total=total, fbo=view.fbo is not None, fbo_error=view.fbo_error, transparent=view.transparent,
         version=THUMBNAIL_VERSION, size=size, workers=prefetch.workers)

    def needs_render(model_id):
        """Cheap: False when the index already holds a current outcome under the same archive identity."""
        entry = archive.entries.get(model_id)
        if not entry or entry['type'] != 'model':
            return False
        return args.force or not index.is_unchanged(model_id, size, entry_identity(archive, entry))

    def upcoming():
        found = []
        for model_id in pending:
            if len(found) >= prefetch.lookahead:
                break
            if needs_render(model_id):
                found.append((model_id, archive.entries[model_id]['path']))
        return found

    done = number = 0
    idle = False
    try:
        while True:
            message = None
            try:
                message = control.get(timeout=0.25) if args.stdin and not pending else control.get_nowait()
            except queue.Empty:
                if not pending:
                    if not args.stdin:
                        break
                    if not idle:
                        index.save()
                        emit(event='idle', rendered=done)
                        idle = True
                    root.update_idletasks()
                    continue
            else:
                if message is None:   # stdin closed: the browser is done with this renderer
                    break
                pending = deque(model_id for model_id in message if model_id not in skip)
                total, number, idle = len(pending), 0, False
                continue
            model_id = pending.popleft()
            number += 1
            if model_id in archive.entries and needs_render(model_id):
                prefetch.plan((model_id, archive.entries[model_id]['path']), upcoming())
            started = time.perf_counter()
            entry = archive.entries.get(model_id)
            status, reason, source_hash, notes, load_s, render_s, wait_s, identity = 'ok', '', None, [], 0.0, 0.0, 0.0, None
            try:
                if not entry or entry['type'] != 'model':
                    raise KeyError('not a model in this archive')
                identity = entry_identity(archive, entry)
                if not args.force and index.is_unchanged(model_id, size, identity):
                    emit(event='model', id=model_id, status=index.get(model_id)['status'], skipped=True, n=number, total=total)
                    continue
                _data, source_hash = archive.bytes(model_id)   # only on a miss
                if not args.force and index.is_current(model_id, size, source_hash):
                    if identity:
                        index.get(model_id)['identity'] = identity   # upgrade an older entry
                        index.dirty = True
                    emit(event='model', id=model_id, status=index.get(model_id)['status'], skipped=True, n=number, total=total)
                    continue
                emit(event='begin', id=model_id, n=number, total=total)
                wait_s = prefetch.wait(model_id)
                payload = loader.load(args.source, model_id, entry['path'], 'Highest', 0, None, True)
                load_s = time.perf_counter() - started
                png, notes = render_model(view, payload, size, supersample)
                render_s = time.perf_counter() - started - load_s
                index.write_png(model_id, png)
            except (ValueError, KeyError, NotImplementedError) as error:
                status, reason = 'unsupported', f'{type(error).__name__}: {error}'
            except Exception as error:   # one failure never stops the batch
                status, reason = 'error', f'{type(error).__name__}: {error}'
            if status != 'ok':
                png_path = index.png_path(model_id)
                if png_path.exists():
                    png_path.unlink()
            index.record(model_id, status, source_hash, size, reason or '; '.join(notes), identity=identity)
            seconds = time.perf_counter() - started
            emit(event='model', id=model_id, status=status, reason=reason, seconds=round(seconds, 3),
                 load_s=round(load_s, 3), render_s=round(render_s, 3), wait_s=round(wait_s, 3), n=number, total=total)
            done += 1
            # Bound memory: only the model being drawn stays decoded, texture cache is trimmed.
            loader.identity = None
            if len(loader.texture_cache) > 400:
                loader.texture_cache.clear()
            view.clear()
            index.save_if_due()
            root.update_idletasks()
    finally:
        prefetch.close()
        index.save()
        emit(event='done', rendered=done, total=total)
        try:
            root.destroy()
        except Exception:
            pass
    return 0


class RendererProcess:
    """One owned renderer child: JSON status lines arrive on ``events``; no shell command string.

    A ``persistent`` renderer (--stdin) gets its queue through ``send`` and stays up, idle, until ``stop``."""

    def __init__(self, source, cache, ids=None, all_models=False, size=DEFAULT_SIZE, cwd=None, python=None, ids_dir=None,
                 persistent=False, skip=()):
        self.persistent, self.skip = persistent, list(skip)
        self.events = queue.Queue()
        self.process = None
        self.finished = threading.Event()
        self.lock = threading.Lock()
        self.ids = list(ids or [])
        self.command = None
        self.source, self.cache, self.size, self.all_models = source, cache, size, all_models
        self.cwd, self.python = cwd, python
        self.ids_dir = Path(ids_dir) if ids_dir else cache_dir(cache)

    def start(self):
        folder = cache_dir(self.cache)
        folder.mkdir(parents=True, exist_ok=True)
        command = [str(self.python or sys.executable), '-m', 'nwo.app.thumbnails', '--source', str(self.source),
                   '--cache', str(self.cache), '--size', str(self.size)]
        if self.persistent:
            command.append('--stdin')
        for model_id in self.skip:
            command += ['--skip', str(model_id)]
        if self.ids and not self.persistent:
            ids_file = self.ids_dir / f'queue-{os.getpid()}-{id(self)}.txt'
            ids_file.write_text('\n'.join(self.ids) + '\n', encoding='utf8')
            command += ['--ids-file', str(ids_file)]
            self._ids_file = ids_file
        if self.all_models:
            command.append('--all')
        self.command = command
        self.log = (folder / 'renderer.log').open('wb')
        self.process = subprocess.Popen(command, cwd=self.cwd, stdout=subprocess.PIPE, stderr=self.log,
                                        stdin=subprocess.PIPE if self.persistent else subprocess.DEVNULL,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        threading.Thread(target=self._read, daemon=True, name='thumbnail-reader').start()
        if self.persistent and self.ids:
            self.send(self.ids)
        return self

    def send(self, ids) -> bool:
        """Replace a persistent renderer's queue (most urgent first). False when it cannot take one (not persistent, or
        it has exited): the caller starts a new renderer."""
        if not self.persistent or not self.running():
            return False
        line = json.dumps(dict(ids=[str(model_id) for model_id in ids]), separators=(',', ':')) + '\n'
        try:
            with self.lock:
                self.process.stdin.write(line.encode('utf8'))
                self.process.stdin.flush()
        except (OSError, ValueError):   # the pipe closed under us
            return False
        self.ids = list(ids)
        return True

    def _read(self):
        try:
            for raw in self.process.stdout:
                try:
                    self.events.put(json.loads(raw.decode('utf8', 'replace')))
                except ValueError:
                    continue
        finally:
            code = self.process.wait()
            self.process.stdout.close()
            self.log.close()
            self.events.put(dict(event='exit', code=code))
            self.finished.set()

    def running(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        process = self.process
        if process is None:
            return
        if self.persistent and process.stdin:
            try:
                process.stdin.close()   # ends the renderer's loop; the kill below makes sure
            except OSError:
                pass
        if process.poll() is None:
            if os.name == 'nt':
                try:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=3, check=False,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for attribute in ('log',):
            handle = getattr(self, attribute, None)
            if handle:
                handle.close()
        ids_file = getattr(self, '_ids_file', None)
        if ids_file:
            try:
                ids_file.unlink()
            except OSError:
                pass


def main(argv=None) -> int:
    args = parse_arguments(argv)
    return run(args)


if __name__ == '__main__':
    if not getattr(sys, 'frozen', False):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
