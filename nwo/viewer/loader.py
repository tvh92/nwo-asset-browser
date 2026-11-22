"""On-selection native geometry loading. No Blender, renders, receipt writes or shaders."""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path
import threading
import numpy as np
from nwo.formats.archive import asset_references
from nwo.formats.native_model import secondary_uv
from nwo.games import RC2016, open_source as open_game_source


def presentation_basis(value):
    x,y,z=value
    return (x,-z,y)


def _basis_array(values):
    """presentation_basis over an (n, 3) array: columns (x, -z, y)."""
    values=np.asarray(values,dtype=np.float64).reshape(-1,3)
    out=np.empty_like(values)
    out[:,0]=values[:,0];out[:,1]=-values[:,2];out[:,2]=values[:,1]
    return out


def _static_raw_vectors(mesh, count):
    """Exact packed-normal-derived xyz, eligible only for a known static source mesh."""
    if mesh.get('static_vector_eligible') is False or (int(mesh.get('flags', 0)) & 0x40):
        return None
    if mesh.get('skin_batch_count', 0) or any(bool(pairs) for pairs in (mesh.get('weights') or ())):
        return None
    packed = mesh.get('packed_normals')
    if packed is None or len(packed) != count:
        return None
    vectors = mesh.get('raw_vectors')
    if vectors is None:
        from nwo.formats.native_model import _unorm_1010102_raw_vectors
        vectors = _unorm_1010102_raw_vectors(packed, 'normal')
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    if len(vectors) != count or not np.isfinite(vectors).all():
        return None
    return vectors


def _tangent_w(mesh):
    """The tangent attribute's w: Rift Apart's whole TEXCOORD3.w (``tangent_w``, the bitangent sign over a squared scale,
    whose magnitude one program reads), else the stored bitangent sign."""
    w=mesh.get('tangent_w')
    return w if w is not None else mesh.get('tangent_signs')


def _present(values):
    """A per-vertex stream that is there: a non-empty list or array (arrays cannot be tested for truth)."""
    return values is not None and len(values) > 0


def _vertex_colours(decoded, meshes):
    """{'vertex_colours': (n, 4) uint8 RGBA} for a Rift Apart model with a colour section, else {}: the stored bytes, which
    the translated materials (rcra_glsl) read as TEXCOORD4. Meshes without colours read 255 (1.0), the engine's value for a
    subset without a colour stream. R&C 2016 rules read the colour alpha only (rock_alpha)."""
    if decoded.get('game')!='rcra' or not any(_present(mesh.get('colors')) for mesh in meshes):return {}
    parts=[np.rint(np.asarray(mesh['colors'],dtype=np.float64).reshape(-1,4)*255.0) if _present(mesh.get('colors'))
           else np.full((len(mesh['positions']),4),255.0) for mesh in meshes]
    return dict(vertex_colours=np.clip(np.concatenate(parts),0,255).astype(np.uint8))


def _morph_payload(decoded, meshes):
    """{'morphs': dict(names, hashes, targets)} for a model with blend shapes on the shown meshes (Rift Apart,
    ``rcra.model.attach_shapes``), else {}. ``targets`` holds one (payload vertex indices, position deltas, normal
    deltas) per shape, in the model's shape order and the display basis; a shape on no shown mesh has empty arrays."""
    shapes=decoded.get('shapes') or []
    if not shapes or not any(mesh.get('morphs') for mesh in meshes):return {}
    parts=[([],[],[]) for _shape in shapes];base=0
    for mesh in meshes:
        for number,vertices,positions,normals in mesh.get('morphs') or ():
            parts[number][0].append(np.asarray(vertices,dtype=np.int64)+base)
            parts[number][1].append(_basis_array(positions));parts[number][2].append(_basis_array(normals))
        base+=len(mesh['positions'])
    targets=[(np.concatenate(i) if i else np.zeros(0,np.int64),
              np.concatenate(p).astype(np.float32) if p else np.zeros((0,3),np.float32),
              np.concatenate(n).astype(np.float32) if n else np.zeros((0,3),np.float32)) for i,p,n in parts]
    return dict(morphs=dict(names=[s['name'] for s in shapes],hashes=[s['hash'] for s in shapes],targets=targets))


def _motion_payload(decoded, meshes):
    """{'vertex_motion': at} for a model whose vertex programs move over shader time on the shown meshes (Rift Apart,
    ``rcra.vertex.VertexMotion``), else {}. ``at(seconds)`` gives (payload vertex indices, positions, normals) in the
    display basis, or None when nothing evaluates at that time."""
    motion=decoded.get('vertex_motion')
    if motion is None:return {}
    wanted=set(motion.meshes);starts={};base=0
    for mesh in meshes:
        if mesh['index'] in wanted:starts[mesh['index']]=base
        base+=len(mesh['positions'])
    if not starts:return {}

    def at(seconds):
        index,positions,normals=[],[],[]
        for mesh,(p,n) in motion.at(seconds,starts).items():
            if mesh in starts:
                index.append(np.arange(starts[mesh],starts[mesh]+len(p),dtype=np.int64))
                positions.append(_basis_array(p));normals.append(_basis_array(n))
        if not index:return None
        return np.concatenate(index),np.concatenate(positions).astype(np.float32),np.concatenate(normals).astype(np.float32)
    return dict(vertex_motion=at)


def _mesh_payload_arrays(decoded, meshes, selected):
    """mesh_payload for decode_model(..., as_arrays=True) meshes: vertices, normals, triangles, uvs ((n, 2)), skin
    joints and weights (flat, 8 per vertex) and rock alpha as numpy arrays, which viewport.prepare_viewport and
    set_mesh take as-is (the list path gives the same values as lists)."""
    vertices,normals,raw_vectors,triangles,uvs,batches,joints,weights,alpha,alpha_sources=[],[],[],[],[],[],[],[],[],[]
    tangents=[];uv1s=[];has_uv1=False
    raw_vector_eligible = True
    base=triangle_total=0
    for mesh in meshes:
        positions=np.asarray(mesh['positions'],dtype=np.float64).reshape(-1,3);count=len(positions)
        vertices.append(_basis_array(positions));normals.append(_basis_array(mesh['normals']))
        tangent,signs=mesh.get('tangents'),_tangent_w(mesh)
        if tangent is not None and len(tangent)==count and signs is not None and len(signs)==count:
            # xyz in the display basis; w is the stored bitangent sign (position w), not a basis value.
            tangents.append(np.column_stack((_basis_array(tangent),np.asarray(signs,dtype=np.float64))))
        raw = _static_raw_vectors(mesh, count)
        if raw is None:
            raw_vector_eligible = False
        else:
            raw_vectors.append(_basis_array(raw))
        colors=mesh.get('colors') if _present(mesh.get('colors')) else None
        alpha.append(np.asarray(colors,dtype=np.float64).reshape(-1,4)[:,3] if colors is not None else np.ones(count))
        alpha_sources.append(dict(mesh=mesh['index'], source='native_colors_alpha' if colors is not None else 'viewer_white_default_1_runtime_input_unresolved'))
        if mesh.get('skin_joint_array') is not None:
            joints.append(np.asarray(mesh['skin_joint_array'],dtype=np.float64));weights.append(np.asarray(mesh['skin_weight_array'],dtype=np.float64))
        else:
            mesh_joints=np.zeros((count,8));mesh_weights=np.zeros((count,8))
            for i,influences in enumerate(mesh.get('weights') or []):
                for slot,(j,w) in enumerate((influences or [])[:8]):
                    mesh_joints[i,slot]=float(j);mesh_weights[i,slot]=float(w)
            joints.append(mesh_joints);weights.append(mesh_weights)
        uv0=mesh.get('uv0')
        if uvs is not None and uv0 is not None and len(uv0):
            uv0=np.asarray(uv0,dtype=np.float64).reshape(-1,2)
            uvs.append(np.stack((uv0[:,0],1.0-uv0[:,1]),axis=1))
        else:uvs=None
        uv1=secondary_uv(mesh)   # the window glass rule's UV1; zeros where a mesh has no second pair
        has_uv1=has_uv1 or (uv1 is not None and len(uv1)==count)
        uv1s.append(uv1 if uv1 is not None and len(uv1)==count else np.zeros((count,2)))
        tris=np.asarray(mesh['triangles'],dtype=np.int64).reshape(-1,3)
        batches.append(((mesh.get('material') or {}).get('index'),triangle_total,len(tris)))
        triangles.append(tris+base if base else tris)
        base+=count;triangle_total+=len(tris)
    if not base or not triangle_total:raise ValueError('Selected source look/LOD has no geometry')
    return dict(**_morph_payload(decoded,meshes),**_motion_payload(decoded,meshes),vertices=np.concatenate(vertices),normals=np.concatenate(normals),
                raw_vectors=np.concatenate(raw_vectors) if raw_vector_eligible and len(raw_vectors)==len(vertices) else None,
                static_vector_eligible=bool(raw_vector_eligible and len(raw_vectors)==len(vertices)),
                tangents=np.concatenate(tangents) if len(tangents)==len(vertices) else None,
                triangles=np.concatenate(triangles),
                uvs=np.concatenate(uvs) if uvs is not None else None,batches=batches,
                **({'uv1s':np.concatenate(uv1s)} if has_uv1 else {}),
                skin_joints=np.concatenate(joints).ravel(),skin_weights=np.concatenate(weights).ravel(),
                rock_alpha=np.concatenate(alpha),rock_alpha_sources=alpha_sources,**_vertex_colours(decoded,meshes),
                source_sha256=decoded['source_sha256'],selected_mesh_indices=sorted(selected),
                source_exact_geometry=True,game_equivalence=False,material_preview='neutral geometry inspection')


def _effect_preview(archive, data, source_path=''):
    """The labelled effect preview of a model, with its sprite textures loaded: the health crate's idle effect
    (nwo.viewer.idle_effect, a model with the igLoc_health_FX locator), a holocon device's hologram
    (nwo.viewer.holocon_effect, by model name) or a holocard's glow (nwo.viewer.holocard_effect, by model name); None
    for other models or when the locator table, the effect or a
    texture cannot be read."""
    import struct
    try:
        from nwo.formats.native_model import decode_locators
        from nwo.viewer.holocard_effect import holocard_preview
        from nwo.viewer.mech_flame_effect import mech_flame_preview
        from nwo.viewer.holocon_effect import holocon_preview
        from nwo.viewer.idle_effect import effect_preview
        from nwo.viewer.textures import load_texture
        locators=decode_locators(data)
        preview=(effect_preview(locators) or holocon_preview(archive,source_path,data,locators)
                 or holocard_preview(archive,source_path,data,locators)
                 or mech_flame_preview(archive,source_path,data,locators))
        if preview:
            preview['textures']={role:load_texture(archive,path) for role,path in preview['texture_paths'].items()}
        return preview
    except (ValueError,KeyError,IndexError,OSError,struct.error):
        return None


def mesh_payload(decoded, lod='Highest', look=0):
    from nwo.formats.native_model import _selected_meshes
    selected=set(_selected_meshes(decoded['looks'],decoded['mesh_record_count'],lod,look))
    chosen=[m for m in decoded['meshes'] if m['index'] in selected]
    if chosen and all(isinstance(m['positions'],np.ndarray) for m in chosen):
        return _mesh_payload_arrays(decoded,chosen,selected)
    vertices,normals,raw_vectors,triangles,uvs,batches=[],[],[],[],[],[]
    tangents=[]
    raw_vector_eligible = True
    # Up to 8 skin influences per vertex as flat (joint, weight) lists; all-zero
    # weights mean "not skinned" and the vertex keeps its rest position.
    skin_joints,skin_weights=[],[]
    rock_alpha=[];rock_alpha_sources=[];uv1s=[];has_uv1=False
    for mesh in decoded['meshes']:
        if mesh['index'] not in selected:continue
        base=len(vertices)
        # presentation_basis inlined: these loops dominate large models.
        vertices.extend([(x,-z,y) for x,y,z in mesh['positions']])
        normals.extend([(x,-z,y) for x,y,z in mesh['normals']])
        signs=_tangent_w(mesh)
        if mesh.get('tangents') is not None and len(mesh['tangents'])==len(mesh['positions']) \
                and len(signs if signs is not None else ())==len(mesh['positions']):
            tangents.extend([(float(x),-float(z),float(y),float(h)) for (x,y,z),h in zip(mesh['tangents'],signs)])
        raw = _static_raw_vectors(mesh, len(mesh['positions']))
        if raw is None:
            raw_vector_eligible = False
        else:
            raw_vectors.extend([(float(x),-float(z),float(y)) for x,y,z in raw])
        rock_alpha.extend([float(c[3]) for c in mesh['colors']] if mesh.get('colors') else [1.0]*len(mesh['positions']))
        rock_alpha_sources.append(dict(mesh=mesh['index'], source='native_colors_alpha' if mesh.get('colors') else 'viewer_white_default_1_runtime_input_unresolved'))
        if mesh.get('weights'):
            for influences in mesh['weights']:
                pairs=(influences or [])[:8]
                skin_joints.extend([float(j) for j,_w in pairs]+[0.0]*(8-len(pairs)))
                skin_weights.extend([float(w) for _j,w in pairs]+[0.0]*(8-len(pairs)))
        else:
            skin_joints.extend([0.0]*(8*len(mesh['positions'])));skin_weights.extend([0.0]*(8*len(mesh['positions'])))
        # The decoder flips V for Blender; textures upload top row first, so flip back.
        if uvs is not None and mesh.get('uv0'):uvs.extend([(u,1.0-v) for u,v in mesh['uv0']])
        else:uvs=None  # untextured if any selected mesh lacks UV0
        uv1=secondary_uv(mesh)   # the window glass rule's UV1; zeros where a mesh has no second pair
        count=len(mesh['positions'])
        has_uv1=has_uv1 or (uv1 is not None and len(uv1)==count)
        uv1s.extend(uv1.tolist() if uv1 is not None and len(uv1)==count else [(0.0,0.0)]*count)
        # (material slot, first triangle, triangle count) for per-material drawing.
        batches.append(((mesh.get('material') or {}).get('index'),len(triangles),len(mesh['triangles'])))
        triangles.extend([(a+base,b+base,c+base) for a,b,c in mesh['triangles']] if base else [tuple(t) for t in mesh['triangles']])
    if not vertices or not triangles:raise ValueError('Selected source look/LOD has no geometry')
    eligible = bool(raw_vector_eligible and len(raw_vectors)==len(vertices))
    return dict(vertices=vertices,normals=normals,raw_vectors=raw_vectors if eligible else None,
                tangents=tangents if len(tangents)==len(vertices) else None,
                static_vector_eligible=eligible,triangles=triangles,uvs=uvs,batches=batches,**({'uv1s':uv1s} if has_uv1 else {}),
                skin_joints=skin_joints,skin_weights=skin_weights,rock_alpha=rock_alpha,rock_alpha_sources=rock_alpha_sources,
                **_vertex_colours(decoded,[m for m in decoded['meshes'] if m['index'] in selected]),
                source_sha256=decoded['source_sha256'],selected_mesh_indices=sorted(selected),
                source_exact_geometry=True,game_equivalence=False,material_preview='neutral geometry inspection')


def material_rules(archive, materials, translations=None):
    """{material path: {'graphs', 'rule_kind', 'rule'}}: which viewer shading rule each material gets
    (exact rule, selected native translation, family fallback or generic reading).
    ``translations`` ({path: selected translation or None}, from material_textures) skips selecting it again."""
    from nwo.formats.links import rule_coverage
    rules={}
    for material in materials:
        path=material.get('path')
        if not path or path in rules:continue
        try:
            ids=[k for k in archive.resolve(path) if archive.entries[k]['type']=='material']
            if len(ids)!=1:continue
            graphs=[r['path'] for r in asset_references(archive,ids[0])
                    if not r.get('resolution_error') and r['path'].endswith('.materialgraph')]
        except (ValueError,KeyError,OSError):
            continue
        from nwo.viewer.native_glsl import selected_translation
        if translations is not None and path in translations: native = translations[path]
        else:
            try: native = selected_translation(archive, path)
            except (ValueError, KeyError, IndexError, OSError): native = None
        forward=False
        if native is None:   # a gated forward rule counts as the exact reading only for the materials it applies to
            try:
                from nwo.viewer.textures import material_texture_paths
                found=material_texture_paths(archive,path)
                forward=bool(found.get('forward')) or (found.get('flat_surface') or {}).get('model')=='color_light'
            except (ValueError,KeyError,IndexError,OSError):
                forward=False
        kind,rule=rule_coverage(graphs, native, forward)
        rules[path]=dict(graphs=graphs,rule_kind=kind,rule=rule,
                         native_limitations=native.get('limitations', []) if native else [])
    return rules


def translation_workers(cpus=None, frozen=None):
    """Worker processes for TranslationPool: the CPUs less two (the Tk thread and the loader), at most 8; 0 in a frozen
    build, where spawned workers would start the app again."""
    import os
    import sys
    if getattr(sys, 'frozen', False) if frozen is None else frozen:
        return 0
    cpus = cpus or os.cpu_count() or 2
    return max(1, min(8, cpus - 2))


class TranslationPool:
    """Spawned worker processes that translate Rift Apart materials into the shared translation disk cache
    (viewer_materials.translate, in the thumbnail renderer's prefetch workers), so a model's cold materials translate in
    parallel and the loader then reads each result from the cache. ``run`` waits for a model's materials; ``warm`` queues
    background work (the pre-warm), which a ``run`` puts behind its own by cancelling what has not started."""

    def __init__(self, source_root, workers):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from nwo.app.thumbnails import _worker_init
        self.workers = workers
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=_worker_init, initargs=(str(source_root), False))
        self.lock = threading.Lock()
        self.background = OrderedDict()   # material key -> future, the pre-warm's

    def _submit(self, unit):
        from nwo.app.thumbnails import _worker_run
        return self.pool.submit(_worker_run, *unit)

    def run(self, keys):
        """Run the units ``keys`` (('material', key) or ('lighting', (kind, class)), loader.cold_units) on the workers and
        wait for them (failures are left to the caller's own translation)."""
        with self.lock:
            for key in [k for k, f in self.background.items() if f.cancel()]:
                del self.background[key]   # queued pre-warm work waits until this model's is done
            futures = [self.background.pop(key, None) or self._submit(key) for key in keys]
        for future in futures:
            try:
                future.result()
            except Exception:   # a broken pool or a cancelled unit: the loader translates it itself
                pass

    def warm(self, keys):
        """Queue ``keys`` behind anything already queued; returns how many were queued."""
        with self.lock:
            for key in [k for k, f in self.background.items() if f.done()]:
                del self.background[key]
            fresh = [key for key in keys if key not in self.background]
            for key in fresh:
                self.background[key] = self._submit(key)
            return len(fresh)

    def busy(self):
        with self.lock:
            return any(not f.done() for f in self.background.values())

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class SourcePreviewLoader:
    """One archive metadata instance and one selected decoded model; no corpus model decode."""
    game=RC2016   # the open archive's game (nwo.games); set by _ensure_archive
    def __init__(self):
        self.archive=None
        self.actor_graph=None   # the archive's actor reference graph, read once for borrowed animsets
        self.graph_lock=threading.Lock()
        self.source_root=None
        self.current=None
        self.identity=None
        self.texture_cache={}
        self.preview_cache={}   # Textures tab: SD previews, kept apart from the 3D view's full-size cache
        self.animation=None
        # Recently viewed models: identity -> decoded model + its animation library, so
        # switching back (or changing look/LOD) skips the decode. Bounded by count and by
        # total vertices. Assigning identity=None from outside empties it (forces a decode).
        self._recent=OrderedDict()
        self._recent_identity=None
        # Model loads and clip decodes run on different worker threads.
        self.lock=threading.Lock()
        # Texture reads from the list, the grid's thumbnails and the preview share the archive's file handle.
        self.texture_lock=threading.Lock()
        self.translation_pool=None   # TranslationPool of the open source, started on the first model with cold materials
        # Opt-in (the browser): the thumbnail renderer and its workers have their own prefetch processes. It also keeps the
        # recently viewed models (RECENT_FILE) and pre-warms their translations in the background (request_prewarm).
        self.parallel_translation=False
        self._prewarm_thread=None
        self._prewarm_again=False
        self.pool_lock=threading.Lock()

    def pose(self, clip_index, root_motion=False, base=None):
        """The clip's pose (``AnimationLibrary.pose``); ``base`` a clip index an additive layer plays over."""
        with self.lock:
            if self.animation is None:raise ValueError('No animation library is loaded for this model')
            if base is None:return self.animation.pose(clip_index,root_motion)
            return self.animation.pose(clip_index,root_motion,base)

    def layer_partners(self, clip_index):
        """Clips the additive clip plays over in the game, best first (``AnimationLibrary.layer_partners``)."""
        with self.lock:
            if self.animation is None:return []
            return self.animation.layer_partners(clip_index)

    def _ensure_archive(self, source_root):
        source_root=Path(source_root).expanduser().resolve()
        if self.source_root!=source_root:
            # The folder's game is detected once per open (nwo.games); its caches live under
            # game.cache_dir(), which for R&C 2016 is .cache itself, as before the game seam.
            # Archive initialization reads metadata and prior source_mapping only.
            # This path is never checkpointed or decoded using Archive.decode().
            game,self.archive=open_game_source(source_root,'source_preview',lambda *_a,**_k:None)
            self.game=game;cache=game.cache_dir()
            self.actor_graph=None
            self.archive.verify_repeats=False   # preview only: read the first of an asset's repeated copies
            from nwo.viewer.textures import TextureDiskCache
            self.archive.texture_disk_cache=TextureDiskCache(cache/'texture_preview')
            from nwo.viewer.native_glsl import TranslationDiskCache
            self.archive.translation_disk_cache=TranslationDiskCache(cache/'native_translation')
            self.archive.cinematic_disk_cache=cache/'cinematic_directory'
            self.source_root=source_root
            self.close_translation_pool()
            self.identity=None;self.current=None;self.texture_cache={};self.preview_cache={}
            self._recent.clear();self._recent_identity=None
        return self.archive

    def archive_textures(self, source_root):
        """Every texture in the archive, model or not: [{'path', 'folder'}] sorted by path."""
        with self.lock:
            archive=self._ensure_archive(source_root);game=self.game
        from nwo.viewer.textures import _suffix
        return [dict(path=p,suffix=_suffix(p),materials=[],via_graph=False) for p in game.texture_paths(archive)]

    # Flash UI (.gfx/.swf) ---------------------------------------------------
    def ui_files(self, source_root):
        """Every Scaleform/Flash UI file in the archive, sorted by path."""
        with self.lock:
            archive=self._ensure_archive(source_root)
        if hasattr(archive,'ids_of_type'):   # a typed index (Rift Apart): no entry dict for every asset of the install
            return sorted({archive.entries[k]['path'] for kind in ('gfx','swf') for k in archive.ids_of_type(kind)})
        return sorted({e['path'] for e in archive.entries.values() if e['type'] in ('gfx','swf')})

    def _gfx_bytes(self, path):
        ids=self.archive.resolve(path) if self.archive else []
        if not ids:return None
        data=self.archive.bytes(ids[0])[0]
        return data

    def _gfx_import(self, url):
        # ImportAssets URLs are bare file names; match them to archive UI files by name.
        name=url.replace('\\','/').rsplit('/',1)[-1].lower()
        stem=name.rsplit('.',1)[0]
        index=getattr(self,'_gfx_names',None)
        if index is None or index[0] is not self.archive:
            # First UI file (entry order) per lower-case stem: a file-name match implies a stem match.
            names={}
            for entry in self.archive.entries.values():
                if entry['type'] in ('gfx','swf'):
                    names.setdefault(entry['path'].rsplit('/',1)[-1].lower().rsplit('.',1)[0],entry['path'])
            index=self._gfx_names=(self.archive,names)
        path=index[1].get(stem)
        return self._gfx_bytes(path) if path is not None else None

    def gfx_library(self, path):
        from nwo.flash.gfx import Library, parse
        cache=getattr(self,'_gfx_cache',None)
        if cache is None or cache[0]!=self.archive:
            cache=self._gfx_cache=(self.archive,{})
        if path not in cache[1]:
            data=self._gfx_bytes(path)
            if data is None:raise ValueError('UI file not found in the archive: '+path)
            cache[1][path]=Library(parse(data),self._gfx_import)
        return cache[1][path]

    def gfx_symbols(self, path):
        from nwo.flash.gfx import symbols
        library=self.gfx_library(path)
        movie=library.movie
        return dict(path=path,symbols=symbols(movie),version=movie.version,signature=movie.signature,
                    imports=sorted({url for url,_name in movie.imports.values()}),
                    errors=list(movie.tag_counts.get('errors',[])))

    def gfx_image(self, path, char_id, frame, size, background=None):
        from nwo.flash.gfx import collect, render
        library=self.gfx_library(path)
        return render(collect(library,char_id,frame),size=size,background=background)

    def gfx_export(self, path, folder, char_id, frames, size, name, include_source=False, progress=None):
        """PNG per (frame, label) into folder; optionally the original .gfx bytes too."""
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        written=[]
        if include_source:
            target=folder/Path(path).name
            target.write_bytes(self._gfx_bytes(path));written.append(str(target))
        for number,(frame,label) in enumerate(frames,1):
            image=self.gfx_image(path,char_id,frame,size)
            suffix='' if len(frames)==1 else '_'+(label or f'frame{frame+1:03d}')
            safe=''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in name+suffix)
            target=folder/f'{safe}.png'
            image.save(target);written.append(str(target))
            if progress:progress(number,len(frames),target.name)
        return written

    def texture_image(self, texture_path, index=0):
        """(Pillow image, facts) for the Textures tab; reuses the 3D view's decoded textures."""
        from nwo.viewer.textures import FORMAT_NAMES, to_image
        archive,game=self.archive,self.game
        if archive is None:raise ValueError('No source archive is loaded')
        texture=self.preview_cache.get(texture_path)
        if texture is None:
            with self.texture_lock:texture=game.load_texture(archive,texture_path)
            self.preview_cache[texture_path]=texture
        index=min(max(0,index),texture.get('array_size',1)-1)
        fmt=texture.get('stored_format',texture['format'])   # a game's float texture shown through its 8-bit conversion
        facts=dict(path=texture_path,format=fmt,format_name=FORMAT_NAMES.get(fmt,str(fmt)),
                   preview_size=(texture['width'],texture['height']),full_size=texture['full_size'],srgb=texture['srgb'],
                   array_size=texture.get('array_size',1),slice=index)
        return to_image(texture,index),facts

    def texture_thumbnail(self, texture_path, size=96):
        """RGB Pillow thumbnail (at most size px) of a texture's top mip, for the texture grid."""
        from PIL import Image
        from nwo.viewer.textures import to_image
        archive,game=self.archive,self.game
        if archive is None:raise ValueError('No source archive is loaded')
        with self.texture_lock:
            image=to_image(game.load_texture(archive,texture_path),0)
        image=image.convert('RGB')
        image.thumbnail((size,size),Image.Resampling.LANCZOS)
        return image

    def export_textures(self, paths, folder, formats, flip=False, progress=None):
        if self.archive is None:raise ValueError('No source archive is loaded')
        return self.game.export_textures(self.archive,paths,folder,formats,flip,progress)

    RECENT_MODELS = 6
    RECENT_VERTEX_BUDGET = 1_500_000

    PURPOSES = ('view', 'export')

    def load(self, source_root, model_id, source_path, lod='Highest', look=0, output_root=None, defer_animation=False, purpose='view',
             borrowed=False, defer_materials=False):
        """Payload for the viewport. With ``defer_animation`` a skinned model returns at once
        with ``animation_pending=True`` and no clips; call ``load_animation`` (another worker)
        to build its clip library. With ``defer_materials`` a model whose materials are not loaded yet returns its geometry
        with ``materials_pending=True`` and no textures (the viewport draws its neutral preview); call ``load_materials``
        (another worker) for them.

        ``purpose='export'`` decodes geometry, skeleton and material paths only (list-valued
        meshes, as export.models.mesh_arrays reads them): no viewer textures, texture list,
        headers, material rules, native translation or viewport arrays, and the returned
        payload carries no geometry. Read ``current``/``animation`` afterwards as before.
        """
        if purpose not in self.PURPOSES:raise ValueError('Unknown load purpose: '+str(purpose))
        return self._load(source_root, model_id, source_path, lod, look, output_root, defer_animation, purpose, borrowed,
                          defer_materials)

    def load_animation(self, source_root, model_id, source_path, output_root=None, borrowed=False):
        """Build (or reuse) the selected model's clip library: {model_id, source_path, clips, clip_notes, animsets,
        clip_seconds, animation_note}; ``clip_notes`` is ``AnimationLibrary.clip_notes`` (copies, additive layers, unplayable clips).

        ``borrowed`` also offers the animsets the game pairs with another model of the same rest rig (opt-in)."""
        with self.lock:
            self._ensure_archive(source_root)
            identity=(model_id,source_path,str(output_root) if output_root else None)
            cached=self._recent.get(identity)
            if cached is None:
                raise ValueError('The model is no longer loaded; select it again')
        # Built outside self.lock, so selecting another model is not blocked behind it.
        self._build_animation(cached,output_root,borrowed,model_id)
        library=cached['animation']
        notes=library.clip_notes() if library else []   # decodes each clip's first frame once, off the Tk thread
        with self.lock:
            if self.identity==identity:self.current,self.animation=cached['decoded'],library
            decoded=cached['decoded']
            return dict(model_id=model_id,source_path=source_path,animsets=decoded['animsets'],
                        clips=library.labels() if library else [],clip_notes=notes,
                        clip_seconds=library.clip_seconds() if library else [],animation_note=decoded['animation_note'])

    def _build_animation(self, cached, output_root, borrowed=False, model_id=None):
        with cached['build_lock']:
            self._build_animation_locked(cached,output_root,borrowed,model_id)

    def _graph(self):
        with self.graph_lock:
            if self.actor_graph is None:
                from nwo.viewer.animation import actor_graph
                self.actor_graph=actor_graph(self.archive)
            return self.actor_graph

    def _build_animation_locked(self, cached, output_root, borrowed=False, model_id=None):
        decoded=cached['decoded']
        if cached['animation_built']==borrowed:return   # None (not built yet) never equals a bool
        cached['animation_built']=borrowed
        cached['animation']=None;decoded['animation_note']='';decoded['animsets']=[]
        if not decoded['joint_count']:return
        if self.game.has_animation_library():   # the game finds its own clips (Rift Apart: through its dag)
            try:
                library=self.game.animation_library(self.archive,cached['data'],decoded['_archive_path'])
                if cached['purpose']=='view':   # the browser also lists cutscene shots and face performances
                    library=library.with_cinematic(self.archive,decoded['_archive_path']).with_performances()
                cached['animation']=library
                decoded['animsets']=list(library.animset_data)
                decoded['animation_note']='' if library.clips else 'No animation set or cutscene clip found for this model.'
            except Exception as error:  # the model still displays without clips
                cached['animation']=None;decoded['animation_note']=f'Animations unavailable: {error}'
            return
        try:
            from nwo.viewer.animation import AnimationLibrary, model_animsets
            animsets=model_animsets(self.archive,output_root,decoded['_archive_path']) if output_root else []
            if borrowed:
                from nwo.viewer.animation import borrowable_animsets
                animsets=animsets+borrowable_animsets(self.archive,self._graph(),model_id,cached['data'],exclude=animsets)
            library=AnimationLibrary(self.archive,cached['data'],animsets)
            # The browser also offers the model's real-time cutscene shots (the only
            # motion some NPCs have); exports opt in through ExportOptions instead.
            if cached['purpose']=='view':library=library.with_cinematic(self.archive,decoded['_archive_path'])
            cached['animation']=library
            decoded['animsets']=list(library.animset_data)
            if not library.clips:
                if not output_root:decoded['animation_note']="Set an output root: animation sets are found through the scan's reference index there."
                else:
                    from nwo.formats.index import open_asset_index
                    try:open_asset_index(output_root).close()
                    except Exception:decoded['animation_note']='No asset index in the output root: run a scan there so animation sets can be found.'
                    else:decoded['animation_note']='No animation set or cutscene clip found for this model.'
        except Exception as error:  # the model still displays without clips
            cached['animation']=None;decoded['animation_note']=f'Animations unavailable: {error}'

    def _remember(self, identity, cached):
        self._recent[identity]=cached;self._recent.move_to_end(identity)
        while len(self._recent)>1 and (len(self._recent)>self.RECENT_MODELS or
                sum(c['vertices'] for c in self._recent.values())>self.RECENT_VERTEX_BUDGET):
            self._recent.popitem(last=False)

    # Translation pool ------------------------------------------------------
    def _translation_pool(self):
        """The open source's TranslationPool, started on first use; None unless ``parallel_translation``, for R&C 2016
        (its graphs compile in nwo.shaders.bindings' own pool) or in a frozen build."""
        with self.pool_lock:
            if (self.translation_pool is None and self.parallel_translation and hasattr(self.archive,'ids_of_type')
                    and self.source_root is not None):
                workers=translation_workers()
                if workers:
                    self.translation_pool=TranslationPool(self.source_root,workers)
            return self.translation_pool

    def close_translation_pool(self):
        with self.pool_lock:
            pool,self.translation_pool=self.translation_pool,None
        if pool is not None:
            pool.close()

    def cold_materials(self, archive, model_id, source_path):
        """The material keys of a Rift Apart model whose translation is not in the disk cache ([] for R&C 2016)."""
        if not hasattr(archive,'ids_of_type'):
            return []
        from nwo.app.thumbnails import prefetch_units
        from nwo.games.rcra.viewer_materials import is_cached
        return [key for kind,key in prefetch_units(archive,model_id,source_path) if kind=='material' and not is_cached(archive,key)]

    def cold_units(self, archive, model_id, source_path):
        """The pool units a model's materials wait for: [('lighting', (direct or ambient, class))] for the engine lighting
        recipes not in the disk cache (shared, slow, first), then [('material', key)] for its cold materials."""
        units=[('material',key) for key in self.cold_materials(archive,model_id,source_path)]
        if units or hasattr(archive,'ids_of_type'):
            from nwo.games.rcra import lighting
            try:
                units=[('lighting',pair) for pair in lighting.cold_classes(archive.root,archive.translation_disk_cache)]+units
            except (OSError,ValueError,AttributeError):
                pass
        return units

    # Recently viewed models and the pre-warm -------------------------------------
    RECENT_FILE = 'recent_models.json'   # in the game's cache folder: [[model id, archive path], ...], newest first
    RECENT_KEEP = 24

    def _recent_path(self):
        return self.game.cache_dir()/self.RECENT_FILE

    def recent_models(self):
        """[(model id, archive path)] of the models last viewed with this game, newest first ([] when unreadable)."""
        try:
            rows=json.loads(self._recent_path().read_text(encoding='utf-8'))
            return [(str(m),str(p)) for m,p in rows][:self.RECENT_KEEP]
        except (OSError,ValueError,TypeError):
            return []

    def _remember_viewed(self, model_id, source_path):
        rows=[(model_id,source_path)]+[r for r in self.recent_models() if r!=(model_id,source_path)]
        path=self._recent_path()
        temp=path.with_name(path.name+'.%d.tmp'%threading.get_ident())
        try:
            path.parent.mkdir(parents=True,exist_ok=True)
            temp.write_text(json.dumps([list(r) for r in rows[:self.RECENT_KEEP]]),encoding='utf-8')
            temp.replace(path)
        except OSError:
            try:temp.unlink()
            except OSError:pass

    def request_prewarm(self):
        """Translate the recently viewed models' cold materials (and the engine lighting recipes) on the pool in the
        background, newest model first, so they open warm after a code change. Work for a model being opened goes ahead of
        it (TranslationPool.run). A request while one runs runs it again afterwards."""
        if not self.parallel_translation:
            return
        with self.pool_lock:
            if self._prewarm_thread is not None and self._prewarm_thread.is_alive():
                self._prewarm_again=True
                return
            self._prewarm_again=False
            self._prewarm_thread=threading.Thread(target=self._prewarm,args=(self.archive,),name='translation-prewarm',daemon=True)
            self._prewarm_thread.start()

    def _prewarm(self, archive):
        while True:
            for model_id,source_path in self.recent_models():
                if self.archive is not archive:
                    return   # another source opened
                try:
                    units=self.cold_units(archive,model_id,source_path)
                except (ValueError,KeyError,OSError):
                    continue
                pool=self._translation_pool() if units else None
                if pool is not None:
                    pool.warm(units)
            with self.pool_lock:
                if not self._prewarm_again:
                    self._prewarm_thread=None
                    return
                self._prewarm_again=False

    def _translate_ahead(self, archive, model_id, source_path):
        """Translate a model's cold materials in parallel on the pool, so material_textures reads them from the cache."""
        try:
            cold=self.cold_units(archive,model_id,source_path)
        except (ValueError,KeyError,OSError):
            return
        if len(cold)<2:
            return   # one translates as fast here
        pool=self._translation_pool()
        if pool is not None:
            pool.run(cold)

    def _decode(self, archive, model_id, source_path, purpose, texture_cache, game=None, defer_materials=False):
        """Decode one model into a _recent entry. Runs without self.lock. ``game`` (nwo.games) is the
        archive's game; the decode and texture reads dispatch through it once per model. With ``defer_materials`` a view
        decode leaves textures, material rules and the effect preview to _decode_materials (``materials_pending``)."""
        game=game or self.game
        decoded,data=game.decode_model(archive,model_id,source_path,purpose=='view')
        decoded.update(textures={},texture_list=[],material_rules={},effect_preview=None,materials_pending=False)
        decoded['_archive_path']=source_path;decoded['animation_note']='';decoded['animsets']=[]
        cached=dict(decoded=decoded,data=data,purpose=purpose,animation=None,animation_built=None,build_lock=threading.Lock(),
                    materials_lock=threading.Lock(),vertices=sum(len(m['positions']) for m in decoded['meshes']))
        if purpose=='view':
            decoded['materials_pending']=True
            if not defer_materials:
                self._decode_materials(archive,model_id,cached,texture_cache,game)
        return cached

    def _decode_materials(self, archive, model_id, cached, texture_cache, game):
        """Textures, texture list, material rules and effect preview of a view entry (once; later calls return at once)."""
        with cached['materials_lock']:
            decoded,data=cached['decoded'],cached['data']
            if not decoded['materials_pending']:
                return
            source_path=decoded['_archive_path']
            self._translate_ahead(archive,model_id,source_path)
            from nwo.viewer.textures import FORMAT_NAMES
            translations={}
            decoded['textures']=game.material_textures(archive,decoded['materials'],texture_cache,translations,hd=True)
            decoded['texture_list']=game.model_texture_list(archive,decoded['materials'])
            for row in decoded['texture_list']:
                try:
                    header=game.texture_header(archive,row['path'])
                    full=(header['hd_width'],header['hd_height']) if header['hd_width'] else (header['sd_width'],header['sd_height'])
                    row.update(format=header['format'],format_name=FORMAT_NAMES.get(header['format'],str(header['format'])),width=full[0],height=full[1])
                except (ValueError,KeyError,OSError) as error:
                    row.update(format_name='unreadable',error=str(error))
            decoded['material_rules']=game.material_rules(archive,decoded['materials'],translations)
            for path,notes in (decoded.get('material_notes') or {}).items():   # decoder notes per material (Rift Apart vertex programs)
                rule=decoded['material_rules'].get(path)
                if rule is not None:rule['native_limitations']=list(rule.get('native_limitations') or [])+list(notes)
            decoded['effect_preview']=game.effect_preview(archive,data,source_path)
            decoded['materials_pending']=False

    @staticmethod
    def _material_rows(current):
        return [dict(index=m.get('index',i),name=m.get('name'),path=m.get('path'),**current['material_rules'].get(m.get('path'),{}))
                for i,m in enumerate(current['materials'])]

    def load_materials(self, source_root, model_id, source_path, output_root=None):
        """After ``load(..., defer_materials=True)``: the model's materials, {model_id, source_path, textures, texture_list,
        effect_preview, materials} (the payload keys a deferred load left empty). Kept with the decoded model, so
        switching back to it, or changing look or LOD, does not load them again."""
        with self.lock:
            archive=self._ensure_archive(source_root)
            cached=self._recent.get((model_id,source_path,str(output_root) if output_root else None))
            texture_cache,game=self.texture_cache,self.game
        if cached is None:
            raise ValueError('The model is no longer loaded; select it again')
        self._decode_materials(archive,model_id,cached,texture_cache,game)
        self.request_prewarm()   # the queued pre-warm waited behind this model's translations
        current=cached['decoded']
        return dict(model_id=model_id,source_path=source_path,textures=current['textures'],texture_list=current['texture_list'],
                    effect_preview=current.get('effect_preview'),materials=self._material_rows(current))

    def _load(self, source_root, model_id, source_path, lod, look, output_root, defer_animation=False, purpose='view',
              borrowed=False, defer_materials=False):
        # self.lock covers loader state only; the decode, texture reads and viewport arrays run
        # outside it so clip, pose and texture-list calls are not queued behind a model decode.
        with self.lock:
            archive=self._ensure_archive(source_root)
            entry=archive.entries.get(model_id)
            if not entry or entry['type']!='model' or entry['path']!=source_path:
                raise ValueError('Selected asset ID and archive path do not match the canonical source index')
            if self.identity!=self._recent_identity:
                self._recent.clear()   # identity was reset from outside: decode afresh
            # Animation sets are found through the output root, so it is part of the identity.
            identity=(model_id,source_path,str(output_root) if output_root else None)+(('export',) if purpose=='export' else ())
            cached=self._recent.get(identity)
            if cached is not None:
                self._remember(identity,cached)
            texture_cache,game=self.texture_cache,self.game
        if cached is None:
            cached=self._decode(archive,model_id,source_path,purpose,texture_cache,game,defer_materials)
            with self.lock:
                if self.archive is not archive:
                    raise ValueError('The source archive changed while the model loaded; select it again')
                cached=self._recent.get(identity) or cached
                self._remember(identity,cached)
        if purpose=='view' and self.parallel_translation:
            self._remember_viewed(model_id,source_path)
        if purpose=='view' and not defer_materials:
            self._decode_materials(archive,model_id,cached,texture_cache,game)
        pending=bool(defer_animation and cached['decoded']['joint_count'] and cached['animation_built'] is None)
        if not pending:
            self._build_animation(cached,output_root,borrowed,model_id)
        with self.lock:
            self.current,self.animation=cached['decoded'],cached['animation']
            self.identity=self._recent_identity=identity
        current,animation=cached['decoded'],cached['animation']
        info=dict(model_id=model_id,source_path=source_path,archive_root=str(archive.root),
            source_packed_sha256=current['packed_sha256'],codec=current['codec'],
            kind='skinned' if current['joint_count'] else 'static',joint_count=current['joint_count'],
            looks=current['looks'],lod=lod,look=look,mesh_record_count=current['mesh_record_count'],
            locators=[dict(m,joint_name=current['joints'][m['joint']]['name'] if m.get('joint') is not None else None)
                      for m in current.get('locators') or []],unread_sections=current.get('unread_sections') or {},
            animsets=current['animsets'],clips=animation.labels() if animation else [],
            clip_notes=animation.clip_notes() if animation and purpose=='view' else [],
            clip_seconds=animation.clip_seconds() if animation else [],
            animation_note='Loading animations…' if pending else current['animation_note'],animation_pending=pending)
        if purpose=='export':
            info.update(purpose='export',materials=[dict(index=m.get('index',i),name=m.get('name'),path=m.get('path'))
                                                    for i,m in enumerate(current['materials'])])
            return info
        result=mesh_payload(current,lod,look)
        result.update(info,textures=current['textures'],texture_list=current['texture_list'],
            effect_preview=current.get('effect_preview'),materials=self._material_rows(current),
            materials_pending=current['materials_pending'],
            limitation='Native bind/rest geometry only. Neutral viewer shading; no runtime material, animation, attachment or game-equivalence claim.')
        from nwo.viewer.viewport import prepare_viewport
        result["_prepared_viewport"]=prepare_viewport(result)
        return result
