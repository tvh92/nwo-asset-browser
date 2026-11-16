# Rift Apart (PC)

The Rift Apart reader lives in `rc2016/games/rcra/`. Rift Apart is a PS5 game; the reader takes its Steam PC release, not a PS5 dump, and reads the install directly: nothing is extracted, and nothing is written to the game folder. This page covers the archive layer, model geometry, textures, the material texture slots behind the viewer's labelled preview, and the translation of the games' own shader programs.

The layouts below were taken from two references and then checked against the installed files:

- ALERT `dat1lib` (GPL-3.0, vendored in `third_party/ALERT`)
- RCRA-Forge (GPL-3.0), for mesh, joint and texture header layouts

## How the browser picks the game

The folder chosen as the game folder picks the game, and the browser shows only that game's assets. `rc2016/games/` holds one `Game` record per game. `games.detect(folder)` tries them in this order:

| Game | Recognised by |
| --- | --- |
| Rift Apart | A `toc` whose first u32 is `0x34E89035`, a `dag` file and a `d/` folder |
| R&C 2016 | Exactly one extracted archive root under the folder (`layout.csv`, `toc`, `dag`, `chunkmap.txt` and segment files) |

A folder that matches neither gets an error naming both games. With the Transform only backend, a folder that matches neither is read as R&C 2016 loose files.

Once the scan recognises the game:

- The window title becomes "R&C Asset Browser — <game>".
- Settings keep a folder and an output root per game (`game.<id>.source`, `game.<id>.output`) beside the last-used `source` and `output`.
- An installed game, under Program Files or in a Steam library, never receives output. Its default output root is `Documents\R&C Asset Browser\<game> Output`.

Each `Game` record declares its capabilities and supplies hooks: model decode, material textures and rules, texture header, texture listing, loading and export, export material maps and shadow-only flags. Dispatch happens once per source open and once per model decode or export, so no hot loop sees the seam. R&C 2016's record names the functions it used before the seam, and its caches stay directly in `.cache/`. Other games cache under `.cache/<game id>/`.

### What Rift Apart supports

Rift Apart's capabilities are `models`, `textures`, `export` and `animation`:

- **Supported:**
  - Browsing.
  - The 3D model view, with skeletal [animation](#animation) playback. The Shaded mode draws each material's own translated program where it translates ([In the viewer](#in-the-viewer)), and the labelled preview elsewhere.
  - The Textures tab.
  - Model export to GLB, glTF and OBJ (FBX, USD and `.blend` through Blender), with skin weights and animation clips. Blender exports carry the game's own materials where the G-buffer writer or, for blended materials, the forward program translates ([Blender materials](#blender-materials)).
- **Disabled:** the Level and Flash UI types, the backend choice, the Blender scene and the shading filter. Each is greyed out with the tip "Not yet available for Rift Apart".
- **Clips:** found through the dag by `Game.animation_library`, without a scan's reference index.
- **Model list:** read in-process through `Game.list_models`, without the pipeline subprocess.

The export writes only what is decoded, through `rc2016/games/rcra_export.py`:

- Each material's base colour and its normal map, at full resolution.
- Every material is opaque, single-sided, non-metallic and fully rough, and its notes say so.
- A texture stored in a float or 16-bit data format is left out with a note, because its 8-bit conversion is not decoded. One example is the 4×4 `R32G32_FLOAT` `default_pink_shader_n` placeholder.

## Install layout

| File | Contents |
| --- | --- |
| `toc` | Asset table: the id, archive, offset and size of every asset, plus optional 36-byte asset headers |
| `dag` | The path, type and dependency list of every named asset |
| `d/*` | DSAR block-compressed archives; the `movie*`, `soundbank`, `soundbank.us` and `wem*` files are stored without DSAR |
| `dstoragecore.dll` | Microsoft DirectStorage 1.2. Its CPU GDeflate codec is the one we use |

## toc

- **Header:** u32 magic `0x34E89035`, then u32 DAT1 size, then the DAT1 itself (type `0x4D7CF320`). The installed file has a few padding bytes after the DAT1.
- **Rows:** each row is one copy of an asset. An asset id is the CRC-64 of its normalized path (`levels.zone_format.path_crc64`).

| Section | Contents |
| --- | --- |
| `0x506D7B8A` | u64 asset id per row |
| `0x65BCF461` | Per row: u32 size, u32 archive index, u32 offset into the archive's decompressed stream, i32 offset of a 36-byte header in `0x654BDED9` (-1: none) |
| `0x654BDED9` | 36-byte asset headers. When a row has one, it is prepended to the archive bytes; a model then has its DAT1 at +36 |
| `0xEDE8ADA9` | 256 spans of (u32 first row, u32 count). Ids are sorted within a span. Locale spans hold the language copies |
| `0x398ABFF0` | 66-byte archive records, with a NUL-terminated name such as `d\model` in the first 40 bytes |
| `0x36A6C8CC` | u64 ids of streamed textures |
| `0xC9FB9DDA` | One 72-byte record per streamed texture id |

**One asset, several rows.** An asset id can appear in more than one row:

- **Streamed textures** have two rows: the headered `.texture` (SD mips) and a headerless stream that holds the top mips.
- **Localized assets** have one row per language. A row's language is its archive's `.xx` suffix, or else the locale of the span it sits in.

`RcraSource.bytes()` returns the headered row in the base language. `texture_stream()` returns the HD stream.

## dag

- **Header:** u32 magic `0xB8EF3955`, u32 DAT1 size, then u32 0 and the DAT1 itself (type `0x2A077A51`).
- **Order:** one entry per named asset, sorted by path CRC-64.
- **The `0x933C0D32` field is not the asset id.** Ids are computed from the path strings instead.

| Section | Contents |
| --- | --- |
| `0xD101A6CC` | u32 offset of the path string in the DAT1 string blob (backslash separators) |
| `0x7A0266BC` | u8 asset type |
| `0xF958372E` | i32 start of the asset's dependency list in `0xBC91D1CC` (-1: none) |
| `0xBC91D1CC` | i32 dag indices; each list ends with -1 |

**What gets a name.** The dag names every model, texture, material and other content asset. The assets it does not name are Wwise audio, movies and some UI files. The browser lists those as `_unnamed/<archive>/<id>`.

**Dependency lists** are what `asset_references()` returns. For example, a model's dependencies are its materials and textures.

## DSAR archives

- **Header:** b`DSAR`, u32 version `0x10003`, u32 block count, u32 offset of the first data byte, u64 total decompressed size, u64 unused.
- **Block records** follow the header, 32 bytes each: u64 decompressed offset, u64 compressed offset, u32 decompressed size, u32 compressed size, u8 mode, then 7 unused bytes.
- **Block size:** usually 256 KiB, up to 4 MiB in texture archives.
- **Compression modes:**
  - **Mode 3, LZ4 block.** Decoded with the `lz4` package, or the pure-Python fallback in `formats.archive`.
  - **Mode 2, GDeflate.** Decoded by DirectStorage's `DStorageCreateCompressionCodecCore` in the install's `dstoragecore.dll`. The front-end `dstorage.dll` export returns `E_NOTIMPL` for the same request.
- **When the DLL is missing,** GDeflate assets fail with a message that names it, and LZ4 assets still read.

`BlockReader` covers a range read in three steps:

1. It bisects the block table to find the covering blocks.
2. It decompresses each block once. Decompressed blocks are kept in a shared 256 MB LRU, because neighbouring assets share blocks.
3. It joins the pieces.

## Source and caches

`RcraSource` offers the same surface the app uses on `formats.archive.Archive`: `entries`, `by_path`, `resolve`, `bytes`, `asset_data`, `asset_references`, `decode`, `close` and `checkpoint`.

- **Entry keys** are the asset id as 16 lower-case hex digits.
- **`entries` is built on request.** It is a read-only mapping that builds an entry dict only when one is asked for. `ids_of_type()` lists one type without building any entries.

**Index cache.** Hashing every dag path is the slow part of opening a source.

- The result is cached in `.cache/rcra/index_<key>.pickle`.
- The cache key covers the toc and dag hashes (through the shared `stat_file_hash` sidecar) and the reader's own source.
- A warm open takes about half a second.

## Models

`rc2016/games/rcra/model.py` decodes a `.model` asset into the same dict schema as `formats.native_model.decode_model` (`schema='rcra-pc-model-v1'`). Because the schema matches, `viewer.loader.mesh_payload`, `_selected_meshes` and the glTF exporter use the result unchanged. `decode_for_viewer` is the `Game.decode_model` hook.

### Asset and DAT1 layout

- **Asset header:** the 36-byte toc header comes first: u32 magic `0x9D2C0FA9`, u32 streamed-section offset, u32 streamed-section size, then 24 zero bytes.
- **DAT1:** starts at +36, with type `0x9D2C0FA9`.
- **String pointers** are offsets from the DAT1 start.

| Section | Contents |
| --- | --- |
| `0x283D0383` built | f32 position offset at +0x1C; f32 position scale at +0x2C (1/4096 in every checked model); u32 at +0x30 whose low 4 bits give the UV scale `(1 << n) / 16384` |
| `0xA98BE69B` | 16-byte vertex: int16 x, y, z, w; u32 packed normal; int16 u, v. The tangent is split over w and the normal word ([Vertex decoding](#vertex-decoding)) |
| `0x0859863D` | u16 indices. Mesh flag `0x10` makes them relative to the mesh's first vertex; without it they are global |
| `0x78D9CBDE` | 64-byte mesh record (fields below) |
| `0x06EB7EFC` | Looks: 8 (u16 first mesh, u16 count) LOD slots per look. The 2016 game uses 4 |
| `0x3250BB80` | Materials: the first half of the section holds (u64 path, u64 name) string pointers |
| `0x6B855EED` | int16 UV1 per vertex, at the same UV scale |
| `0x5CBA9DE9` | u8 RGBA per vertex |
| `0x15DF9D3B` | Joints: i16 parent, u16 index, u16, u16 flags (bit 0: segment scale compensation, as in 2016), u32 name hash (game CRC-32 of the name), u32 name pointer |
| `0xDCC88A19` | Joint rest S/Q/T, 48 bytes each, then the inverse binds from the next 64-byte boundary (the 2016 layout) |
| `0xC61B1FF5` | Skin batches, 16 bytes: u32 offset into `0xDCA379A2`, u32 remap, u16 palette size, u16 (not read), u16 vertex count, u16 first vertex in the mesh |
| `0xDCA379A2` | Skin data: groups of 16 vertices per batch (see [Skin weights](#skin-weights)) |
| `0x5240C82B` | Skin joint palettes: u16 skeleton joints, one palette per batch that names a size |

`0x6B855EED` and `0x5CBA9DE9` may be zero-padded to a 16-byte multiple.

**Mesh record fields (`0x78D9CBDE`):**
- +4: u64 mesh id
- +20: four u32: first vertex, first index, index count, vertex count
- +36: four u16: flags, material slot, first skin batch, skin batch count word (the mesh owns the low byte's count of batches; the high byte is not read)
- +56: u32 first weight index

### Vertex decoding

- **Positions** are `int16 * scale + offset`.
- **Normals** use ALERT's two-angle form: x and y come from two 10-bit fields, z from the remainder, and z's sign is bit 31. Across the corpus, decoded normals agree with face normals at a median dot product of 0.9995.
- **UVs** are `int16 * uv scale`. V is flipped for the Blender convention, as the 2016 decoder does.
- **Tangents** are a second two-angle pair, decoded as the engine model vertex program decodes `TEXCOORD3` (`model.unpack_tangents`):

  | Field | Bits | Use |
  | --- | --- | --- |
  | a | `\|w\|` bits 0-9 | `0.0027648 a − 1.4142`, the tangent's y angle |
  | b | normal word bits 20-29 | `0.0027648 b − 1.4142`, the tangent's x angle |
  | z sign | normal word bit 30 | set: z positive |
  | e | `\|w\|` bits 10-14 | scale of `TEXCOORD3.w` |
  | sign | sign of w | the bitangent sign; w = 0 counts as positive |

  With q = a² + b² and s = √(1 − q/4), the tangent is `(b s, a s, ±|1 − q/2|)`, a unit vector. `TEXCOORD3.w = sign(w) / (e/31)²`; draw render flag `0x10000` negates it. The decoder returns `tangents`, `tangent_signs` (±1) and `tangent_w` (the whole `TEXCOORD3.w`).
- **Where the tangent decode comes from:** the engine model vertex programs in `RiftApart.exe`. These are the vertex programs that pull `g_MainVBNrmTanView`, write `TEXCOORD` to `TEXCOORD7` and bind only the model streams and per-draw views (two programs). Translated with `sm5.py` and run on Ratchet's vertex streams, their `TEXCOORD3` equals the decoder within 2 × 10⁻⁶ (`tests/test_rcra_model.py`). The material graphs' own vertex programs decode it the same way.
- **How the pixel programs read it:** 530 G-buffer and 604 forward programs (of 1,474 each) read `TEXCOORD3.w`. All but one test only its sign (`w < 0`). One forward program (`world_space_uv_scrolling_sss`) also reads `|w|`. e = 0 would make w infinite; it does not occur in 4.6 million checked vertices, where e runs from 1 to 31.
- **Checks on 300 models (6.9 million vertices):**
  - The tangent follows UV0: its cosine with the area-weighted dP/du has a median of 0.999997 and is above 0.9 for 99.7 % of vertices.
  - `h (N × T)` runs along −dP/dv (game v down) for 99.6 % of vertices, the orientation the UV-derivative frame used.
  - The tangent is not orthogonalised against the stored normal: |T·N| > 0.05 on 22 % of vertices, and 10 % against the area-weighted face normal. The programs receive it as stored.
- **Vertex programs** move some meshes at rest. Those meshes are drawn moved, and they keep their stored vertices ([Vertex programs](#vertex-programs)).

### Skin weights

Each skin batch covers `vertex count` vertices from `first vertex` of its mesh. Its data is a run of groups of 16
vertices; the last group may be shorter. Each group is:

1. A byte n.
2. For n = 0: one u8 joint id per vertex, with weight 1.
3. Otherwise: n + 1 (u8 joint id, u8 weight) pairs per vertex, at most 8 influences. Weights sum to 255 or 256 and are
   normalized.

**Joint ids.** A batch with a palette size reads the u8 id as an index into that many u16 skeleton joints, at byte
offset `remap` of `0x5240C82B`. A batch without one adds `remap` to the id. These are the R&C 2016 rules, used by the
rigs of more than 256 joints. Remapping brings the median joint-to-vertex distance on those rigs from about 2.0 to
0.28 units.

**The weight records in `0xCCBAFF15` are not read.** They hold the same weights as fixed 8-byte records (4 u8
joints, 4 u8 weights) for the meshes with flag `0x100`, from the mesh's first weight index. The batches cover every
skinned mesh:

- Across the install's 959 skinned models (41,818 meshes), every batch parses and covers its mesh.
- On the 77,722 batches that both forms hold, the raw batch ids and weights equal the records exactly.
- So the records keep palette-local ids, and only the remapped id is a skeleton joint. On a remapped rig, the record
  ids put the Thug Brawler's pistol on `sword_RT_unequip_chunk`; the remap puts it on `gun_unequip_chunk`.

`joint_count` is the skeleton size when the model has `0xDCC88A19`. The skeleton is still listed in `joints`.

### Not decoded yet

These sections are listed in `unsupported_sections` and are not interpreted.

**Models without geometry:** 24 models have no vertex or mesh section, only looks, locators or joints. Decoding one raises `UnsupportedModelError` with a message saying so.

## Animation

Rift Apart clips play in the 3D view and export with the model, like R&C 2016 clips. The decoder is
`rc2016/games/rcra/animation.py`; `rc2016/games/rcra/clip_library.py` finds a model's clips and is the
`Game.animation_library` hook.

### Clip assets

Each clip is its own `.animclip` asset: the 36-byte toc header (type `0x0F64FFE8`) and a DAT1 of the same type. Its
sections have the R&C 2016 clip ids ([animation](animation.md#sample-layout)), and the base records, base scale
tail, 48-byte destination layout and frame-major bit packing are the 2016 version-15 layout. So
`animation_transforms.sample_transforms` turns a decoded clip into scale, rotation and translation per joint
unchanged. Joint hashes are the game CRC-32 of the model's joint names.

Four things differ from 2016:

| Section | Rift Apart |
| --- | --- |
| `0x9DF23F77` built | +0 u32 is the name's string offset, not a format version. +4 name hash; +8 flags (`0x40` additive, `0x10000` frame table); +12 f32 fps; +16 f32 seconds; +32 u16 joints, frames, elements, 0; +40 u32 bytes per stored sample; +44 u16 scale-tail blocks, u16 count of 4-byte records after the tail; +48 u16 stored samples |
| `0xD070D358` elements | u32 destination per element (not u16), then u8 bit width minus one each, zero-padded to 4 bytes |
| `0x3A7B4855` samples | With flag `0x10000`: `round(seconds * fps) + 1` u16 stored-sample indices, padded to 16 bytes, then the stored samples. A clip that repeats a pose stores it once. Without the flag, frame i is stored sample i. One clip (`rivet_phonemes`) also stores the closing pose, equal to the first, without counting it in the header; it is not played |
| `0x2BB5BC8F` base | Joint records and scale tail as in 2016; one additive clip adds the 4-byte (u16 joint, u16 value) records the built header counts, which are not read |

A clip with only custom float tracks (`0x14014CB6`/`0x116EB684`: camera and property curves) is not skeletal and is
refused. Root motion (`0x3976E44C`, `0xE08AA35F`), triggers, morph and face sections are not read.

**Scale sentinel.** In 10 additive clips (bladeball turns, horned-toad aim and look layers, the emergency-raft
wobble) a joint's packed scale is `0x8000` on all three axes, which would be −8. `packed_scales` reads it as one,
the identity of the additive preview's scale product. This is inferred, not traced in the runtime: every such joint
has an identity base rotation, no scale channel and a translation under 0.01. The same value in a clip without the
additive flag, and any other high-bit scale, is still refused (as in 2016, whose clips carry no additive flag).

**Coverage.** Of the 23,317 installed clips, 21,377 are skeletal and all of them decode. The other 1,940 are not
skeletal: custom-track clips and one camera-shake clip. No decoded clip uses the collapsed-rotation sentinel.

**Check.** On `amb_fish_01`, frame 0 of `Swimming_Idle` matches the model's rest S/Q/T with a median translation
difference of 1.1e-4.

### Skinning

Each skinned mesh's weights are in skin batches (`0xC61B1FF5`, `0xDCA379A2`). The layout is in [Models](#models).
Rest S/Q/T and inverse binds (`0xDCC88A19`) have the 2016 layout, read by `formats.skeleton.inspect_model`.
Both 2016 inheritance rules reproduce the stored inverse binds:

| Model | Max inverse-bind error |
| --- | --- |
| Fish | 4.5e-7 |
| Ratchet | 1.8e-5 |

### Finding a model's clips

Clips are found through the dag, never by joint coverage alone:

1. **Referenced animsets.** An actor or config that depends on the model, and the configs it depends on, name the
   animsets. Each animset depends on its clips.
2. **Same-folder animsets.** These are the animsets in the model's own folder, or whose first clip is in that folder
   or in its `animations` or `animation` subfolder, whose first clip names at least half of the model's joints. The
   runtime assembles some models, such as the Ratchet outfits, and no actor names them. Zones place others, such as
   the thug's sitting crowd set in `enm_thug_brawler/animations/`. The subfolder rule also gives about 50 models
   their gallery (topiary showcase) sets and sets shared between rigs, such as the bladeball's with the swarmer's. A
   set whose clip assets are all already offered is left out: 29 such sets only repeat clips, among them the
   standing crowd sets (twins of the sitting ones) and the shark's `enm_pool_shark_underwater`.
3. **Cutscene shots** (3D view, or export with Include cutscenes). These are clips under `cinematics/` whose name
   ends in `_shot<N>_<performer>`. The performer must be the model's file stem or one of its folder names holding an
   underscore, as written or without a `_master<N>` or copy-number suffix. The clip's `.cinematic2` is its source.

Each clip is still classified on its first frame, as in 2016. The clip list hides one that does not drive the
skeleton. For example, `hero_ratchet.model` gets:

- 1,380 animset clips from five animsets, among them `hero_rivet.animset`, which an actor pairs with the model.
- 532 cutscene shots.
- Of those 1,912 clips, 212 are additive layers, 38 are unplayable and 20 are copies.

The clip notes take about 8 s to compute, off the Tk thread.

### Not decoded

- Root-motion paths.
- Clip blending and additive partners. Additive clips are previewed on the rest pose, as in 2016.
- Face performance clips (`.performanceclip`) and the phoneme and viseme pose clips their performance sets use.
- Vertex-animation textures over time: the viewer and exports draw the pose at shader time 0 ([Vertex programs](#vertex-programs)).

## Textures

`rc2016/games/rcra/texture.py` reads `.texture` assets. It returns the same dicts as the 2016 viewer's texture functions:

- `load_texture`, `load_texture_full` and `read_top_mip` return the loaded top mip.
- `texture_header` returns the header facts.
- `export_textures` also writes the asset and its HD stream.
- `archive_textures` lists every texture in the install.

### Layout

- **Asset header:** the 36-byte toc header comes first: u32 type `0x8F53A199`, u32 DAT1 size, u32 SD byte count.
- **DAT1:** a small DAT1 follows. Its one section, `0x4EDE3593`, is the 44-byte texture header.
- **SD mip chain:** follows the DAT1.
- **HD stream:** a streamed texture's headerless toc row holds the larger top mips the SD chain lacks.

**Header fields.** The 2016 game uses the same layout.

| Offset | Field |
| --- | --- |
| +0 | u32 SD bytes |
| +4 | u32 HD bytes |
| +8 | u16 HD width, u16 HD height |
| +12 | u16 SD width, u16 SD height |
| +16 | u16 array size |
| +18 | u8 Insomniac format, u8 planes |
| +20 | u16 DXGI format |
| +30 | u8 SD mip count |
| +32 | u8 HD mip count |

### Storage on PC

**Mips** are stored linearly in DXGI order. There is no tiling.

**Arrays and cube maps** store one whole chain per slice, back to back.
- A cube map has an SD payload exactly six chains long (108 sky and probe textures). Its six faces are read as six slices.

**Format coverage.** All 12,502 headers parse, and their SD and HD sizes match the computed chains exactly.
- **Previewable:** BC1–BC7, BC6H, RGBA8, R8 and R8G8.
- **Sized but not previewed:** the float data formats (RG32F, RGBA16F, RGBA32F, RG16F, RG16, R16F) used by vertex-animation and lookup textures.

## Materials and the viewer preview

`rc2016/games/rcra/material.py` reads the texture slots of a `.material`. It also provides the `Game` hooks `material_textures`, `model_texture_list`, `material_rules` and `effect_preview`.

### Layout

- **Asset header:** a material asset is the 36-byte toc header (type `0x88730155`) and a DAT1 of the same type.
- **Section `0xF5260180`** begins with ten u32:
  1. section size
  2. parameter count
  3. unknown
  4. unknown
  5. parameter batch end
  6. texture count
  7. texture batch start
  8. texture batch end
  9. unknown
  10. unknown
- **Texture records:** each is (u32 string offset from the batch end, u32 slot key). The strings follow the batch.
- **Section `0xE1275683`** is the 40-byte flags block. Its third u32 is the material flags word ([Flags word](#flags-word)).
- **Fur materials** have no `0xF5260180` section, so they bind no textures.

### Flags word

The engine's material loader copies u32 2 of section `0xE1275683` into the material object (`material.flags_word`). These bits are traced in `RiftApart.exe`:

| Bits | Meaning | Where the engine reads it |
| --- | --- | --- |
| `0x80`, `0x100`, `0x200` | Blend mode: additive, blended (alpha "Lerp"), modulate. At most one is set. | The mesh sort sends a material down the blended path when any of the three is set and bit `0x8` is clear. The engine's blend enum names the modes and describes them as rendering "to Alpha layer". |
| `0x8` | Keeps a material on the opaque path (2 materials) | Same mesh sort |
| `0x800` | Lit: the draw's render flag `0x400` | Draw setup, `(word >> 11 & 1) << 10`. Programs with the flag clear use a lighting term of 1 (unlit). |
| `0x4000` | Fur | The loader reads the fur section |
| `0x800000` | Water | The loader reads the water section |

`material_blend` returns `opaque`, `additive`, `blended` or `modulate`; `draw_flags` returns the lit bit. Over 5,721 materials: 3,974 opaque, 1,309 blended, 427 additive and 11 modulate. The blend factors are pipeline state outside the files.

### Slot keys

Slot keys are the low 32 bits of the 2016 game's `0x49A7866B........` slot hashes. Across 5,721 materials they bind the same texture suffixes:

| Slot key | Texture | Count |
| --- | --- | --- |
| `ABDAD780` | Base colour (`_c`) | 2,600 of 2,862 |
| `7AE34E7A` | Normal (`_n`) | 2,591 of 2,647 |
| `1576D848` | `_g` | — |
| `E68930AC` | `_m` mask | — |

### The labelled preview

The viewer draws a labelled **Rift Apart preview** (`rule_kind 'rift apart preview'`) in the Textured mode, and in the Shaded mode for materials whose program does not translate or while the native translation is off ([In the viewer](#in-the-viewer)).

**What the preview uses:**
- **Base colour:** from the material's `ABDAD780` slot.
- **Normal:** from the material's `7AE34E7A` slot.
- **Everything else:** neutral parameters.

**What it leaves out:**
- **`_g` and `_m` maps:** their channel meanings are not decoded.
- **Material graphs:** the preview translates no graph shader. Architecture glass keeps its hand decode (next section) as the fallback of the translated program. 1,474 graphs exist; 1,386 are bound by materials and 29 materials bind none.
- **Gloss, metal, emission and blending:** these are review values, not game data.
- **Fur shells:** they have no texture slots, so they draw in the neutral colour.

### Architecture glass (`gbl_arch_glass`)

The one decoded graph. 44 materials use it, among them the Nefarious trooper and swarmer domes. Every graph embeds its shaders (about 15 DXBC containers each, SM5 and DXIL, with named reflection), so the rule is traced to the graph's forward pixel shader (SM5 `ps_5_0`, disassembled with `D3DDisassemble`), not to a capture. The viewer draws these materials alpha-over through the forward glass program (`uKind 26`; GLSL comment in `rc2016/viewer/shading.py`), with `rule_kind 'rift apart glass'`.

**Graph layout** (`graph_layout`). Three sections of the graph, then the material's own values (`material_parameters`):

| Section | Content |
| --- | --- |
| `0x1CAFE804` (graph) | Slot table, rows of (u32 default-texture string offset, u16, u16 unit, u64 slot hash). Unit `u` is texture register `t(5 + u)`. |
| `0x45C4F4C0` (graph) | Parameter records (u16 byte offset, u16 byte size, u32 key); the offset is the dword position in `cb2`. |
| `0xA59F667B` (graph) | The float defaults of `cb2`. |
| `0xF5260180` (material) | The same record layout from byte 40, its values in a block that starts at header word 2. |

`cb2` is the graph's defaults overwritten by the material's values at the graph's offsets. An unbound texture unit takes the graph's default texture.

**Texture units**

| Unit | Slot | Role | Use |
| --- | --- | --- | --- |
| `t5` | `A6C3B1D5` | `_pick` | One scalar, added to the alpha. The default is a mid-grey. |
| `t6` | `ABDAD780` | `_window_c` | Colour. |
| `t7` | `7AE34E7A` | `_n` | Normal x, y. |
| `t8` | `6376D191` | `_window_g` | F0 in rgb, gloss in alpha. |

**Parameters** (dword in `cb2`; the trooper dome value in brackets, then the graph default)

| Dword | Key | Name | Use |
| --- | --- | --- | --- |
| 0 | `FD3F0F1D` | `soft_flag` | Scene-depth fade on the colour and alpha (1, default 1). |
| 1 | `0C0A6B30` | `emissive_adapt` | Non-zero: the emission is scaled by 4 `m_AdaptedLuminance` (1, default 0). |
| 2 | `24F4CA4C` | `gloss_scale` | Scale of the `t8` gloss (0.9, default 1). |
| 3 | `70137DA7` | `rim_width` | Width of the grazing rim (0.2, default 0.25). |
| 4 | `74979B4A` | `colour_k` | Colour factor at full alpha (0.5, default 5). |
| 5 | `8E7EC8A1` | `g_mode` | At or above 0.5 `t8` is F0 rgb and gloss alpha; below, F0 is `t8.g` and gloss is `sqrt(t8.r)` (1, default 0). |
| 6 | `06F19114` | `emission` | Self-lit amount (15, default 0.05). |
| 7 | `CAF31501` | `alpha_floor` | Lowest alpha before the rim and `t5` (0.4, default 0.4). |

**Equations.** `N' = normalize((1 - x² - y²) N + x T + y B)`, `(x, y) = 2 n.rg - 1`.
- **Rim:** `rim = sat(((1 - N'·V)⁴ - w/2) / (w/2))`.
- **Alpha:** `pre = max(rim, floor) + t5.r`, `alpha = sat(pre)`, times the vertex alpha (taken as 1).
- **Colour:** `c' = c (1 + sat(pre + soft_flag · depth_fade) (k - 1))` with `depth_fade` 0 here, emission `E = emission (1 - rim/2)`, times `4 L` with the adapt flag.
- **Specular:** GGX alpha `(0.9725 - 0.7514 gloss)⁴`, Schlick Fresnel on `F0`.
- **Result:** `rgb = c' (E + Ld) + specular`, then the fog.

**Not decoded.** These are viewer stand-ins, and the preview says so (`GLASS_LIMITATION`):
- **Lighting:** the light list, shadows, probe and screen-space reflections and fog are the viewer rig. The specular lobe is the shared glass lobe, not the program's own visibility term (its Fresnel has F90 = 1).
- **Frame:** the soft depth fade is 0 (no scene depth); the vertex alpha, the alpha override and the dissolve discard are 1, 0 and off; the normal frame comes from screen derivatives, not the vertex tangent.
- **Exposure:** `m_AdaptedLuminance` is not known. The preview uses 1.0 (`RCRA_ADAPTED_LUMINANCE`), so a material with a large `emission` (the trooper and swarmer domes: 15) draws far brighter than a lit surface.
- **Blender:** there is no Blender rule; the Blender scene is not available for Rift Apart.

**Translated program.** The automatic path (see [Material graph shaders](#material-graph-shaders)) reproduces these equations: on the trooper dome, with the real material values and textures, its alpha and emissive colour match them to float precision (`tests/test_rcra_shaders.py`). The translation also shows the terms the preview leaves out:
- the material and vertex alpha (`TEXCOORD.w`) and the per-draw alpha override (`MatDrawCB.m_AlphaOverrides`)
- the custom clip plane and the dissolve test on `TEXCOORD.z`, which set the alpha to 0
- a fog factor on the alpha, gated by render flag `0x4000000`
- the unlit path: with render flag `0x400` clear, the lighting term is exactly 1
- the normal frame: the vertex tangent and its bitangent sign, not screen derivatives

## Material graph shaders

Every `.materialgraph` carries its compiled programs, and the engine's own passes are in `RiftApart.exe`. Both are read in place; nothing is copied out of the game folder.

| Module | Role |
| --- | --- |
| `rc2016/shaders/dxbc.py` | DXBC container: chunks, RDEF reflection (constant buffers with struct members, bindings), signatures, and the SM4/SM5 token stream. DXIL containers are identified only. |
| `rc2016/shaders/sm5.py` | SM5 program to a typed expression DAG, with a numeric evaluator. |
| `rc2016/games/rcra/shaders.py` | Pass table, pass programs, the generic material binding and the neutral engine profile. |

### Programs in a graph

- **Shader section `0xBBFC8900`:** entries on 16-byte boundaries, with zero bytes between some of them. An entry is a 12-byte header (u32 magic `IGSH`, u32 SM5 size, u32 DXIL size), the SM5 container, then the DXIL container. Raytracing libraries use magic `IGRS` and are DXIL only (395 graphs).
- **Pass table `0x8C049CCA`:** 18 records (1,281 graphs) or 36 (193 graphs) of (u32 flags, u32 hash, u32 entry offset from the section start, u32 entry size). The record index is the pass slot; an entry's size is 12 + SM5 + DXIL.
- **Raytracing record `0xF9C35F30`:** one record of the same shape, pointing at the `IGRS` entry.

The slot meanings below come from each program's stage, outputs and bindings across all 1,474 graphs:

| Slot | Graphs | Program |
| --- | --- | --- |
| 1 | 1,474 | Forward pixel program, one target. It does its own lighting: light grid, probes, sun shadows, SSR, GI, fog. |
| 4 | 1,474 | G-buffer pixel program, 4 targets (5 in 54 graphs): packed normal and specular, albedo and occlusion or emission, view depth, velocity. |
| 5 | 1,474 | The same writer without velocity (3 targets). |
| 8 | 1,474 | Depth-only pixel program (alpha test). |
| 11 | 1,474 | Two-target pixel program. |
| 15, 16 | 1,474 | GI voxelization pixel program (UAV stores) and its geometry program. |
| 3, 7, 10, 13, 17 | 208 | Vertex programs of vertex-deforming graphs ([Vertex programs](#vertex-programs)). |
| 0, 14, 18 to 31 | fewer | Other pixel and vertex programs, mostly in the 36-record tables. |

The engine vertex program that feeds slots 1, 4 and 5 is in the exe (vertex pulling from `g_MainVBPosView`, `g_MainVBNrmTanView`, `g_MainVBUV0View` and others). Its translation gives the interpolants:

| Interpolant | Content |
| --- | --- |
| `TEXCOORD.xy` | UV0 (int16 × the subset's UV scale / 16384) |
| `TEXCOORD.zw` | Material and object values (dissolve, alpha) |
| `TEXCOORD1.xy` | UV1 |
| `TEXCOORD2.xyz`, `.w` | World normal; instance index + 0.25 |
| `TEXCOORD3.xyz`, `.w` | World tangent; `sign(w) / (e/31)²`, the bitangent sign over a scale ([Vertex decoding](#vertex-decoding)) |
| `TEXCOORD4` | Vertex colour or paint, squared ([In the viewer](#in-the-viewer)) |
| `TEXCOORD5` | Previous clip position (velocity) |

### Translation

`dxbc.program` decodes the token stream: opcodes, operands with swizzles, masks, modifiers and relative indices, declarations and the immediate constant buffer. Over all 7,899 distinct SM5 programs in the graphs, its instruction list matches `D3DDisassemble`, mnemonic by mnemonic and operand count by operand count. The one naming difference is the disassembler's `resinfo_uint`.

`sm5.translate` turns a program into a hash-consed DAG:

- **Types:** every node is a float, an integer bit pattern or a comparison result. Bitcasts fold for constants; folding uses float32 arithmetic.
- **Branches:** each `if`, `else` and `switch` case runs as its own region and joins with one `select` per register.
- **Loops:** a loop is unrolled while its exit condition folds. A data-dependent loop (a light list, for example) poisons only the registers it writes, with an `unknown` node, so outputs that do not depend on it still translate. Branch, break, return and discard conditions act only on active lanes, so they are folded under what a lane's active predicate implies. A counter is then exactly the iteration count, and a loop with an early break still ends at its count. Facts carry over iterations: a lane active now passed the previous iteration's exit test, so its decided parts were false. An exit test that one select condition decides is split on it. A parallax counter that jumps past its step count on a hit (n = trunc(32 − 24 sat(x)), so 8 to 32 steps) then reads "hit, or step ≥ n", and the loop ends at n's bound. Value ranges (`Graph.bounds`) decide comparisons, and an integer test of `ftoi(x)` becomes the exact float test of x. Extra predicated iterations are allowed only up to the bound, at most 64 steps plus the exit test. When the folded exit test no longer shows the counter (16 to 64 portal steps: the test is decided false while the step stays below 16), the budget comes from the unfolded test, from the counter's low bound. Ray marches and parallax decals end this way. The outputs are bit-identical to the unfolded translation wherever both translate.
- **Discard and return:** both become predicates. A discarded pixel's registers are don't-care, so a discard does not predicate later register writes. This keeps a register one type when a program reuses it, a mask before the discard and a float after it, as the holo graphs do. UAV stores stay gated by the discard. On the materials both versions translate, the outputs are bit-identical on every kept pixel.
- **Texture sizes:** `resinfo` on a texture whose size the bindings know (`Bindings.texture_size`) folds to SM5's values for its dimension, mip level and return type. `surface` and `forward` take `texture_sizes`, which is `texture.texture_size`: the full texture with every streamed mip resident (labelled). Array sizes and depths do not depend on streaming.
- **Names:** constant-buffer reads become inputs named from the reflection, such as `GlobalWorldCBuffer.m_KeyLightVec+4` or `MultiCustomMaterialCB.g_CustomMaterial_Buffer+12`. Interpolants become inputs named by semantic, such as `TEXCOORD2.x`. Texture reads become `sample` and `load` nodes carrying the binding name.
- **Compute programs:** UAV stores are kept with their predicate (`Translation.stores`).

A `Bindings` object can replace any input during translation.

**Material binding.** The binding is the same for every graph:
- `cb2` (`MultiCustomMaterialCB`) holds the graph's float defaults, overwritten by the material's values at the graph's offsets (`material_constants`).
- Texture unit `t(5 + u)` (`textureUnit<5+u>`) is the graph's slot-table row `u`: the material's texture for that slot, else the row's default (`material_units`).
- A graph without parameters has no parameter or default section, and some have only the slot table; `graph_layout` reads the parts that are there.

**Neutral engine profile** (`NEUTRAL_INTEGERS`, `NEUTRAL_FLOATS`): no local lights or probes (lookup word counts 0), no debug view (`m_MiscFlags` 0), no optional features (`m_FeatureFlags`, `m_RTFlags` 0) and 8 sun-shadow taps. These are viewer choices, not game data. Every other engine input stays a named input.

### Coverage

With the neutral profile, over all programs (`ready` means no output depends on an unknown value):

| Slots | Ready | Material uses covered |
| --- | --- | --- |
| G-buffer 4 and 5 | 1,449 of 1,474 programs | 5,611 of 5,692 |
| Forward 1 | 356 fully, 1,073 in part, 45 not at all | 552 of 5,692 fully |
| Vertex 3, 7, 10, 13, 17 | 194 of 208 | 526 of 583 |

With the vertex bindings of [Vertex programs](#vertex-programs), which bind the vertex streams and leave the force-field bitset empty, all five vertex slots translate for the 207 graphs that materials use (583 of 583 uses); the 208th, `gbl_displacement_singularity`, has no material.

The forward programs still in part keep further engine loops, so their outputs need more engine inputs bound. Not translated:
- groupshared memory
- tessellation control-point inputs (3 graphs)
- UAV atomics
- the DXIL-only raytracing libraries

### G-buffer and deferred lighting

The G-buffer writer of `basic_normal_gloss` (666 materials) translates fully. The engine's deferred apply programs are compute programs in the exe that read `g_GBufNormalSpec`, `g_GBufDiffuseEmis` and `g_GBufExtra`. Their translations decode the targets:

| Target | Content |
| --- | --- |
| NormalSpec x, y | Normal: 12 bits each, `n = v × 0.00069070 − 1.41421` (octahedral-style, z sign in bit 15 of y); class in bits 13 to 15 of x (bit 12 set means class 0) |
| NormalSpec z, w | Bytes: F0 red, F0 green (z high, low), F0 blue (w high), gloss (w low). F0 is stored as `sqrt(F0) × 255` and read as `(byte / 255)²`. |
| DiffuseEmis rgb | Albedo (`basic_normal_gloss`: `t5.rgb × tint`, `cb2` dwords 0 to 2) |
| DiffuseEmis a | Occlusion when NormalSpec y bit 13 is set; otherwise emission `2^(5a) − 1` |

The class-0 apply program (exe offset 59122268) gives the deferred BRDF. It is not the R&C 2016 one ([shading](shading.md#opaque-deferred-brdf)):

| Term | Rift Apart |
| --- | --- |
| GGX | `α = (0.9725 − 0.7514 gloss)²` |
| Visibility | Smith joint approximation, `0.5 / (NL (NV (1 − α) + α) + NV (NL (1 − α) + α))` |
| Fresnel | Schlick, F90 = 1 |
| Diffuse | Lambert `sat(N·L)`, no 1/π, times the light colour |

### Forward profile

Blended materials are drawn by their forward program (slot 1), which lights itself. For export, `rc2016/games/rcra/shaders.py` defines one labelled set of engine inputs. Its rule is that every light source is zero, and the renderer lights the material's G-buffer surface instead:
- **Lights:** key light, light grid and lists, shadow atlas, ray-traced shadows, gobos, probes, GI, SSR, AO and the BRDF lookup read 0 (`FORWARD_LIGHTING`).
- **Draw flags:** only the material's lit bit (`material.draw_flags`).
- **Object:** the neutral object of [Per-object values](#per-object-values).
- **Depth:** the stored scene depth sits at the far clip, so there is no soft depth fade.
- **Scene G-buffer:** the scene the far-clip depth stands for is empty, so reads of `g_GBufNormalSpec` (screen effects, eye normal smoothing, decals reading around the pixel) return a cleared target, every word 0, at the review viewport size (`SCENE_GBUFFER`, `SCENE_SIZE`). G-buffer writers that read these engine textures, or the default probe in their emission, get the same values.
- **Clip plane:** the custom clip plane clips nothing.
- **View:** the programs rebuild the view position from `SV_POSITION` and `m_ViewToWorldMat`. With `FORWARD_VIEW_FLOATS`, `SV_POSITION.xyw` is the camera-to-point offset in world axes, so a renderer supplies it as `VIEW.xyz` and the matrix translation as the camera position `CAMERA.xyz`.

`surface.forward` returns the target under this profile: colour (HDR), alpha and the discard. On the trooper dome it matches the hand decode of `gbl_arch_glass` with lighting zero: alpha within 4 × 10⁻⁷, colour within 7 × 10⁻⁷ relative. Two translator folds make the profile's dead code disappear: a comparison `x < 0` of a value that is never negative is false, and a shifted zero is zero.

**The scene behind.** A program that samples the scene colour behind it (`g_FrameBuffer`: heat haze, shockwaves, water splats) binds the sample as symbolic `BACKGROUND.rgb`. `surface._affine_split` keeps it only when each colour channel is affine in its own background channel, colour = A + B × background (through add, negate, a multiply or divide by a free value, or a select on a free condition), and the alpha and discard do not read it. It then returns A as the colour and B as the `background` terms. Over the 45 materials it splits, A + B × background matches the unsplit program within 1.4 × 10⁻⁷ relative at random inputs. The sample's screen-space offset (the distortion) is dropped and labelled.

A lit material's background coefficient could carry the forward lighting the profile zeroes. `surface._lit_background` translates such a program again with every value the profile sets to 0 left symbolic (`_EngineFreeBindings`: the unbound engine constants, such as the key light, ambient fill, GI and cloud shadows, and samples of the lighting textures; the light-list counts stay 0 and the view matrix keeps the forward view binding). Then:

- **B reads none of them:** the scene passes through unlit whatever the lights, and the profile's B is the game's: the shockwaves, ground waves and portal energy (10 materials, whose split matches the unsplit program within 8 × 10⁻⁸ relative).
- **B = B0 + Y × L:** the program lights the scene behind as albedo. L is one engine factor per channel that also lights the program's own colour and reads no specular input (`SPECULAR_INPUTS`): the forward diffuse light, key light × shadows (cascades, cloud shadows, gobo) × N·L plus (1 − the BRDF lookup's diffuse term) × the ambient fill or light grid, distant GI and AO. In `gbl_heat_distortion_01`, B is 0 with every light zero and 1 with only the ambient fill at 1. `surface._factor_split` finds L; B0 and Y are free of every symbolic value and become the terms `background` and `lit_background`, and a renderer supplies its own diffuse light for L (26 materials: heat distortion, goo and water splat decals, ripples, bubbles, liquids). Over them, A + (B0 + Y × L) × background matches the program within float rounding (3.8 × 10⁻⁵ relative at random engine values).
- **Otherwise** it stays unsupported: three goo decals (`fx_goo_decal_simple_alt`) also pass the background through a specular factor (the BRDF lookup's scale term and the gloss reflection constants).

**Scene mode.** A program that reads the scene behind other than as an affine tint keeps it: the screen blend curves of the holograms and camera panes (`bg < 0.5 ? 2ab : …`), a vignette's contrast power, the absolute value or saturate of a distorted scene, and particles that read one background channel into all three. `surface.forward` then returns the whole target with `BACKGROUND.rgba` as inputs and a background tint of 0 (`reads_scene`), labelled (`SCENE_NOTE`). A lit one qualifies only when its target reads the scene and the light sources in separate summands (`surface._separable` over the symbolic translation), so the scene passes through unlit. The symbolic values are the light sources only: the unbound `GlobalWorldCBuffer` constants and the lighting textures; view and draw settings (clip plane, depth of field, temporal jitter, alpha overrides) keep their profile values. The viewer samples the inputs from a copy of its frame taken before the batch; Blender has no node for the scene behind, so these materials keep their G-buffer surface there.

**Texture kinds.** Material units are 2D arrays, sampled at a slice. 3D textures (flipbook volumes) are sampled trilinearly with a normalised depth ([Textures](#textures)). Cube maps (`TextureCube`, and a `TextureCubeArray` read at a constant cube index) are sampled along the direction the program computes (the first three address lanes, in game space). The explicit mip level, bias or gradients of a cube lookup are not applied: the top mip is read, and the result says so. An array read at a run-time index, or a cube read by `gather` or `ld`, is unsupported. A cube texture's slices are its faces, six per cube in the order +X, -X, +Y, -Y, +Z, -Z, top row first. A constant array index that is NaN (a per-object seed of 0 divided by 0) reads slice 0, as D3D converts NaN to the integer 0 (`blender_export._slice_index`), and one past the array reads its last slice, as D3D clamps an array index (`slice_index`). A texture a material binds that the install does not hold (about 20 materials) is replaced by the graph's default for that unit when the install has it, else by a texture that reads 0 in every channel, as an unbound D3D resource does (`texture.UNBOUND`). The game's own fallback is not known; the limitations name each texture and its stand-in (`blender_export.replace_missing`).

**Coverage:** 1,744 of 1,747 blended materials (the 77 cube-map materials translate; the cube arrays among them are all read at index 0), 27 of them in scene mode, which Blender does not draw. Not translated: the three goo decals that light the scene behind them through a specular factor.

### Per-object values

Programs read the engine's per-object record, `g_SceneObjectView` (element `$Element`, 128 bytes; names and offsets from the reflection). The engine's model vertex program shows its layout:
- **Transform:** row-major 3 × 4, rows at bytes 0, 16 and 32, translation in dword 3. The world normal is the object normal dotted with each row.
- **Values:** alpha, an fp16 object colour (`m_ColorRGFp16`, `m_ColorBMiscFp16`), custom XY, Z and random halves, creation time, LOD override.
- **Interpolants:** the vertex program passes the custom values on as `TEXCOORD6 = (1, Misc, Rand, LOD factor)` and `TEXCOORD7 = (custom X, custom Y, custom Z, FX lighting index)`.

The game sets these per object at run time. `shaders.NEUTRAL_OBJECT` is a plain object: identity transform, alpha 1, white colour, and every custom value, time and bias 0. Both the G-buffer and the forward translations bind it, and a material that reads it says so in its limitations. Effects driven by the custom values, such as the particle quads `exp_quad` and `smk_quad`, come out at their zero state (often transparent).

### Vertex programs

208 graphs (583 material uses) replace the engine model vertex program with their own, in pass slots 3, 7, 10, 13 and 17: vertex-animation textures (crowds, prison-cell idles, the Zurkon pub band, swarmer eyes and dials), wind sway, flags, beams and shockwaves, rotating debris, and offsets along the normal. `rc2016/games/rcra/vertex.py` translates them with the SM5 translator and places each mesh's vertices where its material's program puts them at rest.

**Vertex pulling.** The programs read the vertex themselves from `SV_VertexID`. The streams are typed views, and each read lands on the vertex's own element of the 16-byte vertex ([Vertex decoding](#vertex-decoding)), which the translation checks for every stream:

| View | Element | Read as |
| --- | --- | --- |
| `g_MainVBPosView` | 2v | int16 x, y, z, w, scaled by the subset's `m_MetersPerUnit` (the model's position scale) |
| `g_MainVBNrmTanView` | 4v + 2 | The normal word as four UNORM channels, 10, 10, 10 and 2 bits. The programs rebuild the normal from them as `model.unpack_normals` does. |
| `g_MainVBUV0View` | 4v + 3 | int16 u, v, times the UV scale of `m_GeomLodAndUVScales` bits 0 to 3 (UV1: bits 4 to 7) |
| `g_MainVBUV1View`, `g_MainColorVBView` | v | UV1 and the colour section |

The subset record (`g_ModelSubsetView`) gives the stream offsets, the scale and the UV-scale word; the paint and skinned streams are not read for a static subset.

**Bindings** (`vertex.translate`; review values, labelled `REST_NOTE`):
- **Draw:** `MatDrawCB.m_RenderFlags` holds 0x1, 0x8000 and the material's lit bit. With both low bits clear the programs write no geometry, and their meaning is not decoded; 0x8000 reads the model's colour section, as the viewer does.
- **Object:** the neutral object of [Per-object values](#per-object-values), and the same transform as its previous one.
- **Engine:** shader time 0 and every other engine constant 0: no wind, no ambient animation, no force fields (the per-tile force-field bitset is empty).
- **View:** the camera at the origin with world axes and an identity world-to-clip matrix, so `SV_POSITION.xyz` is the moved position in model units and `TEXCOORD2.xyz` the world normal.

**View check.** A second translation moves the camera and uses another affine world-to-clip matrix (`CHECK_VIEW`). When the world positions or normals it gives differ from the review view's, the placement depends on the view (screen-space glitches, camera-facing geometry) and the stored vertices are drawn, labelled (`VIEW_NOTE`). A program that writes a clip w other than 1 counts as view-dependent too.

**Survey** (shader time 0, the neutral object, 256 random vertices per material):

| At rest | Material uses | Graphs | Examples |
| --- | --- | --- | --- |
| Moves | 434 | 148 | Vertex-animation textures (190 uses, 52 graphs, read with `ld`; the normal comes from the texture too), wind sway, flags, beams, the holo offset |
| Stored vertices | 89 | 28 | Interactive foliage, floating hexes and bubbles: their offsets are 0 at rest |
| View-dependent | 60 | 33 | The dimensional and klemming glitches (an offset along the normal that depends on the view), a grass shader, four HUD programs that write screen positions |

All 583 translate. Slots 3, 10, 13 and 17 place every vertex where slot 7 does, and no program changes UV0, UV1 or the vertex colour at rest, so the pixel programs read the same interpolants. Of the moving uses, 397 move further over shader time; the at-rest pose is the one at time 0.

**Application.** `vertex.deform_model` runs from the `decode_model` hook (`model.decode_for_viewer`), so the viewer, OBJ, glTF and the Blender formats all draw the moved vertices. The meshes of one material run as one batch:
- **Evaluation:** the translation runs on the CPU over the model's vertex streams (`vertex.evaluate`, float32 numpy, bit for bit the translator's operations, except which signed zero a min or max of two zeros returns). The vertex-animation textures are read with integer texel loads and half-float unpacking (`ld`, `f16tof32`, `ubfe`), which the viewer's GLSL 1.20 cannot express. At rest the result is static, so the CPU run is the same arithmetic. `rcra_glsl.vertex_source` stays the engine program's colour square.
- **Textures:** `ld` reads the stored texel (0 outside the texture). A filtered read (`sample_l`) is bilinear with wrap at the top mip; the sampler state and mip level are not applied, and the note says so. A texture the install does not hold gets the pixel side's stand-in (`blender_export.replace_missing`).
- **Results:** a mesh moves when a vertex changes by more than 10⁻⁵ model units or a normal by more than 10⁻⁴. A moved mesh keeps its stored vertices and normals (`stored_positions`, `stored_normals`); the glTF export writes them as a morph target at weight 0, named `stored vertices`, which Blender imports as a shape key.
- **Notes:** each material's note reaches the viewer's material list (`decoded['material_notes']`) and `export.json`. `decoded['vertex_programs']` holds one row per mesh: state, note and largest offset.
- **Caching:** both translations are kept in the source's translation disk cache, keyed by the material, the graph, the model's stream layout and `vertex.CODE_SHA256`. A cold load of the stadium crowd (550 meshes, 92 materials) takes about 5 s.

**Limitations:**
- Motion over time is not drawn.
- On a skinned mesh, the at-rest offset is added to the bind pose, so its joints carry it; the game adds it after skinning, in object space.
- A moved mesh takes the program's tangent output (`TEXCOORD3`) with its normal. Where that output does not translate, the stored tangent stays, and the note says so.

### Fur materials

A fur material (19: Ratchet's and Rivet's head, ears and limbs, Daxter, the sheep, a carpet, a towel, a cotton canopy and a cloud) has no graph. In place of the serialized parameters it carries a fur record, DAT1 section 0xD9B12454: 13 u32, an integer (32 on Ratchet's head), eight floats, and four offsets into the material's DAT1 of its texture paths, in the order base colour, normal, gloss and fur control (`fur.fur_record`; the viewer's preview takes the first three as its slots).

The executable holds the fur programs, found by their reflection names (`fur.fur_writer`): the fur G-buffer writer, a ps_5_0 with constant buffer `MatrialFurCBuffer` (`m_Density`, `m_Length`, `m_GlossScale`, `m_SpecularScale`, `m_TransmittanceScale`, `m_Wetness`) and textures `g_BaseMap` (t5), `g_FurLayerMap` (t7, 3D), `g_FurGlossMap` (t8) and `g_FurControlMap` (t9), writing the four engine targets; a five-target variant; and the shell vertex programs (`ModelFurCBuffer`). `surface.surface` takes the writer as its `program` (`fur.fur_program`), so the material gets the same terms and packing as a graph's writer: albedo from the base map, F0 and gloss from the gloss map, occlusion from the control map, and the vertex normal (the writer binds no normal map).

What is not decoded, and labelled (`fur.FUR_NOTE`):
- How the engine fills the fur constants from the record. The surface terms read `m_GlossScale`, `m_SpecularScale` and `m_Wetness`, bound to review values 1, 1 and 0 (`FUR_REVIEW`).
- The shells: layers extruded along the normal and cut by the layer map, which the engine builds at run time (no texture of it ships). The writer's discard, sat(layer term + TEXCOORD3.w) < 0.5, is that cut and is not applied: the material is its base surface.

## Blender materials

A Blender export (`.blend`, FBX or USD, through `rc2016/export/blender_convert.py`) replaces each imported glTF material with the game's own material. The work splits into three modules:

| Module | Role |
| --- | --- |
| `rc2016/games/rcra/surface.py` | Translates the G-buffer writer (slot 4) and unpacks the engine's packing into surface terms: a recipe. For blended materials it also translates the forward program (slot 1) into colour and alpha. |
| `rc2016/games/rcra/blender_export.py` | The `Game.blender_materials` hook: recipes and unit textures, written as `rcra_materials.json` beside the export |
| `rc2016/export/rcra_blender.py` | Runs in Blender: the "RC RA Deferred BRDF" group, and one Math node per recipe operation. A division by a constant 0 (a neutral profile value) is x × Exponent(10000), which is x × inf as in D3D; Blender's Divide alone returns 0. A `lookup` (below) is a one-row float32 image read with Closest. Saturate is Maximum then Minimum, which drop NaN as D3D does |

### Surface terms

The packing is the same engine code in every graph (see [G-buffer and deferred lighting](#g-buffer-and-deferred-lighting)). `surface` translates the writer with the material bound and unpacks:

- albedo, F0 rgb, gloss and the world normal
- occlusion, or emission `E` (self-lit `albedo × E`)
- the writer's discard, which becomes alpha clip

F0 and gloss keep the game's 8-bit rounding. A material passes only when every term is a float expression of the mesh interpolants (UV0, UV1, normal, tangent and sign, vertex colour, front face) and its own textures and constants. Integer arithmetic on constants and at most four boolean conditions, such as a triplanar writer's dominant-axis index, is evaluated in every case and becomes a select of constants. This is exact. Other integer arithmetic of float values (cell indices, hashes, a remainder of case constants) is carried in float on the same integers: casts round toward zero, shifts and masks become floored division and modulo, and a remainder of a select of constants folds per case. It is exact while every value stays within ±2²⁴, and the result says so. A word the program packs from fields (an or of shifted values, a `bfi`) and reads back with a mask, `ubfe` or a shift gives each field its own value, which holds while every field fits below the next one, as the packing's scales are set for; that is labelled too. Over the materials these rewrites gain, the terms are bit-identical to the unrewritten translation at random inputs. A normal word chosen by a select between two packed words is unpacked per case. The snow writer does this: its snow arm writes another lighting class, but it is gated by the deformable-snow inputs, which hold review values of 0.

Values a material does not own are bound and listed in its limitations:
- **Neutral draw:** no dissolve, opaque, no render flags (lighting class 0), shader time 0.
- **Volume flags:** the per-tile bitset of engine flag volumes (`g_VolumeFlagBitsetView`, whose bits index the boxes in `GlobalVolumeFlagBuffer`) is empty in every pass, so no flag volume covers the material. Rain on glass and the rain, snow and puddle writers read it to find shelters.
- **Material LOD distance:** the per-material and per-subset records (`g_MaterialView.m_LodDist`, `g_ModelSubsetView.m_MaterialLodDist`, `shaders.NEUTRAL_RECORDS`) hold no distance, like the neutral object's override. Writers then take no LOD fade, so they draw at full detail. Other members of these records are not bound. The material record's alpha (`g_MaterialView.m_Alpha`) is 1, opaque, as the neutral object's.
- **Review values:** screen position for dither, LOD fade and adapted luminance 1.
- **View direction:** under the review screen position the camera sits at the shaded point, so a writer that normalises the direction to the camera (tangent-space parallax in the holo and dimensional glass, fresnel, the emissive sequence, the two-layer snow) gets a normalised zero vector, NaN. When a term depends on NaN, `surface` translates the writer again with the forward profile's view (`_ViewBindings`: `VIEW`, `CAMERA`, see [Forward profile](#forward-profile)) and uses that when no NaN is left, with a limitation saying so (449 materials). The snow writers then read the deformable-snow map, `g_DeformablesView`, bound empty (0) as its constants are.
- **Undefined terms:** 77 writers still divide by a per-object value the neutral object holds at 0: the stadium crowd's vertex-animation frame (0/0) and a butterfly wing's random seed. Their limitations say which terms are undefined (`NAN_NOTE`); renderers resolve the NaN by D3D rules.

**Coverage:** 5,711 of 5,721 materials, the 19 fur materials among them ([Fur materials](#fur-materials)). Not translated: eight materials name a graph by file name that the install does not contain (`material.no_graph_reason`), one has no graph at all (`flowmap_water_custom`), and `jn_glass_01_opaque` views a loop flag's boolean mask as a float (`bits2f`, NaN when set) in arms that only correlated loop conditions rule out.

Some materials name their graph by file name only, which the dag does not resolve; `material._graphs` then takes the one materialgraph asset of that name. Writer forms the unpacking also reads: two constant normal words (`surface.constant_normal`, the exact inverse of the packing: |n.z| = 1 − 8|p|², z sign in y bit 15), and a byte whose factor the compiler folded into the 255 scale, ftoi(x k + 0.5), read as x k / 255.

The emission alpha is decoded from the engine encoding `0.2 log2(min(E, 31) + 1)`, and also from the form the compiler folds it into (`0.2 log2 Y`, with the clamp dropped or the 1 merged into other constants). That form decodes as `E = clamp(Y, 1, 32) − 1`, which is what the deferred apply reads back from the target's [0, 1] alpha.

**Lookup tables.** Some programs index a table the compiler unrolled into select chains: `select(eq(X, k), table[k], …)` over every key. The noise in the god rays, search lights and wind quads reads a 512-entry permutation table (151, 160, 137, …) and its 12 gradients, 24 to 28 chains of about 511 links. That is about 21,000 recipe operations. Blender's node creation grows with the square of a tree's size (3,000 nodes in one tree take about 35 s), so a tree that size is out of reach. For the Blender export, `surface.lookup_tables` turns each chain of at least 16 integer keys over one X into a `lookup` node (`low`, `table`, `present`, args X and the rest of the chain), so those recipes have 366 to 2,858 operations. `rcra_blender` emits it as a one-row float32 image (R the value, G whether the key is present), packed as EXR, read at the texel centre with Closest. A key that is outside the table, not an integer or absent falls through to the rest of the chain, as the selects do. `evaluate_recipe` expands lookups back into the chain.

**NaN constants.** A neutral profile value can leave a NaN constant in a recipe (a normalised zero vector in `cpz_spotlight_godray`'s clip). D3D carries it: arithmetic on NaN is NaN, saturate of NaN is 0, min and max return the other operand, and comparisons are false. Cycles runs with fast math and does not keep NaN, so `surface.fold_nan` resolves constant NaN by those rules before the Blender export builds the recipe (`surface.blender_recipe` is both rewrites). The folded recipe evaluates the same as the original. Recipes still over 10,000 operations (`blender_export.MAX_RECIPE_NODES`) are exported as untranslated, with the count as the reason; none are left.

**Part groups.** A recipe with more than 400 operations (`PART_ABOVE`) is emitted into chained part groups of at most 150 nodes (`PART_SIZE`), named after the material's group with `part 1`, `part 2` and so on. A value used in a later part leaves its part through the Group Output and enters the later one through its Group Input. Parts only feed later parts. The largest recipe (3,710 operations) builds in seconds instead of minutes, and Cycles flattens the groups, so the bakes are unchanged.

### Textures

Each unit texture is written as stored. Missing channels follow D3D: green and blue read 0, alpha reads 1.

| Texture | Written as |
| --- | --- |
| 8-bit, `Non-Color` | RGBA PNG |
| 8-bit, sRGB | 16-bit RGBA PNG of the decoded linear values, loaded as `Non-Color` (alpha as stored). The GPU decodes each texel before it filters; Cycles filters an sRGB image, 8 or 16 bit, and decodes afterwards, which darkens lookups between contrasting texels (0.216 instead of 0.401 between 0.1 and 0.9). Stored values are within 7.6 × 10⁻⁶ of the decode, and every byte value stays distinct |
| Float | Raw float32 RGBA (`.f32`), loaded into a packed float image |
| Array, slice chosen at run time | Every slice; the material picks one by `round(slice)`, clamped |
| Cube map | One equirectangular image, 4 × face size by 2 × face size, read by an Environment Texture node with the game direction as its Vector (Blender's `u = 0.5 − atan2(y, x) / 2π`, `v = 0.5 + atan2(z, hypot(x, y)) / π`). Each pixel holds the D3D cube lookup of its direction (major axis picks the face, `s = (sc / \|ma\| + 1) / 2`, bilinear within the face); sRGB colour is filtered in linear light. A cube whose alpha is not 1 gets a second image of its alpha. Filtering differs at face seams and poles (labelled) |
| 3D (flipbook volumes) | One atlas of every slice, each tile with a one-texel border copied from its opposite edge; the material samples the two slices around `depth × z − 0.5` bilinearly and mixes them, wrapping in depth (trilinear, as D3D filters) |

The colourspace follows the DXGI format: sRGB formats are `sRGB` (decoded when written, as above), everything else `Non-Color`. Some material and graph strings double their path separators; they are normalised.

### In Blender

The emitter reads its inputs from the mesh:

- **UVs:** the first and second UV maps, with game `v = 1 − v_blender`; every texture lookup flips back.
- **Normal:** Geometry Normal, un-flipped on back faces.
- **Tangent frame:** the decoded tangent and `TEXCOORD3.w`.
  - **Export:** the glTF export writes them as one custom point attribute, `_RCRA_TANGENT` (native xyz, w). Blender's importer swaps the data of two custom attributes on one primitive when its set order differs from its list order, so there is only one. The importer makes a VEC4 a float colour attribute.
  - **Conversion:** `rcra_blender.store_tangent_frame` turns the attribute into the tangent's coordinates (a, b, c) at each corner, `T = a t + b (n × t) + c n`, where t is Blender's UV0 tangent (MikkTSpace) and n is the corner normal. They are stored as the corner attribute `rcra_tangent_frame`, w as the point attribute `rcra_tangent_w`, and the colour attribute is removed. The coordinates are solved in that basis, so they also hold where Blender's tangent is not orthogonal to the normal (degenerate UV triangles), and they keep c, the part of the game's tangent along the normal.
  - **Material:** rebuilds T from the Normal Map node's tangent and the geometry normal, so T turns with the armature as both do. On a back face, Cycles and EEVEE negate both the geometry normal and the Normal Map node's output. The material un-flips both, as the game's interpolants are not turned. The UV0-frame fallback does the same.
  - **Without the attribute:** a mesh gets Blender's UV0 frame, with the bitangent sign on Blender's +v (the game's −v), labelled (`NO_TANGENT_NOTE`).
  - **Draw state:** render flag 0x10000 is clear (unmirrored draw).
- **Vertex colour:** the first colour attribute, else 1.
- **Vertices:** where the material's vertex program places them at rest, with the stored vertices as the shape key `stored vertices` ([Vertex programs](#vertex-programs)).
- **Vectors:** game `(x, y, z)` is Blender `(x, −z, y)`.

The BRDF group follows the deferred apply program:

| Term | Node |
| --- | --- |
| Specular | Metallic BSDF, GGX, roughness `0.9725 − 0.7514 gloss`, F82 Fresnel with a white edge tint (Schlick with F90 = 1) |
| Diffuse | Diffuse BSDF, roughness 0 |
| Self-lit | Emission: albedo at strength `E` |

Light strength is π × the game's light colour, as for R&C 2016. Cycles uses the exact Smith term rather than the game's joint approximation. The game applies occlusion to ambient light only, which Cycles cannot separate; the group's "Occlusion on diffuse" input (default 0) applies it to the diffuse colour instead.

### Blended materials

A blended material's entry carries `blend` and a `forward` recipe ([Forward profile](#forward-profile)), and keeps its G-buffer `recipe` when that translates. The Blender material composes them over a Transparent BSDF:

| Mode | Shader |
| --- | --- |
| Blended | Transparent(mix(white, B, alpha)) + alpha × (BRDF + Emission(colour)); with B = 0 this is mix(Transparent, BRDF + Emission(colour), alpha). The BRDF's own emission input is left unlinked, because the forward colour already holds it. |

A material that lights the scene behind it (`lit_background`, [The scene behind](#forward-profile)) needs the diffuse light reaching the point, which Cycles has no node for. `surface.unit_irradiance` folds it at a unit white light, B = background + lit_background, and the material's limitations say so (`UNIT_IRRADIANCE_NOTE`).
| Additive | Transparent(1 + alpha B) + Emission(colour, strength alpha) |
| Modulate | Transparent tinted by mix(white, colour, alpha) |

B is the background tint of a program that passes the scene behind it through ([Forward profile](#forward-profile)); a modulate material that does so is not linear in it and keeps its opaque surface.

`VIEW` and `CAMERA` come from Geometry Incoming and Camera Data View Distance. EEVEE materials use the blended render method. The engine names the modes, but the blend factors are the standard ones for those names and are labelled as such. A blended material whose forward program does not translate keeps its opaque surface, with the reason in its limitations.

### Checks

Export check: each export builds the `.blend` as a candidate, reopens it in a second Blender (`blender_convert.py --validate`), then moves it into place.

`python -m rc2016.games.rcra.blender_check --game <install> --blender <blender.exe> --output-root <folder> <material> ...` bakes every surface output of the emitted node tree in Cycles. It compares the bake, texel by texel, with the recipe evaluated by the translator on the same textures:
- **Window:** a UV window small enough that the bake's sub-pixel jitter stays below a 32nd of a texel, so a material can tile its UV up to 8 times. With the old quarter-texel window, a triplanar rock that tiles ×2 differed by 0.11 at sharp texels.
- **Undefined pixels:** a pixel whose recipe samples a texture at a non-finite coordinate is left out and counted as `undefined`. This happens when a neutral profile value ends up as a divisor, such as a custom object value of 0 dividing a distortion offset in the waterspray material. Neither renderer defines that lookup.
- **Results:** on one material from each of the ten most-used graphs, F0 and alpha are exact. Albedo agrees within 1.1 × 10⁻⁶ and the normal within 3 × 10⁻⁵, and gloss, occlusion and emission within 8-bit rounding ties. The snow writer, a triplanar rock and a layered robot material agree within 3 × 10⁻⁵ too. Before the sRGB units were stored linear, they differed by up to 0.11, all from filtering between contrasting sRGB texels.
- **Forward:** a blended material's forward colour, alpha and background tint are baked too, with the view held at fixed values (`FORWARD_VIEW`). On the trooper dome and the wine glass, colour agrees within 2 × 10⁻⁶ and alpha within 6 × 10⁻⁸; a 128-slice noise volume within 5 × 10⁻⁶. A cube map is compared with the exact D3D lookup of its stored faces, so the difference includes the equirectangular resampling: on a laser-tube material the colour (values near 31) differs by at most 0.1, and on a synthetic cube with six different faces 93 % of the baked pixels agree within 0.02 and the rest sit on face seams. The 64-step convergence-tear portal march and the rain torso glass bake exactly.

### In the viewer

The model viewer's Shaded mode draws the translated programs (`rc2016/games/rcra/viewer_materials.py` builds each material's receipt; `rc2016/viewer/rcra_glsl.py` emits its GLSL, one statement per recipe node). The receipt follows the export's choices: the G-buffer surface for opaque materials; the forward program, plus the lit surface for blended ones. The viewer's native translation toggle switches back to the preview.

| Part | Viewer |
| --- | --- |
| Vertices | Where the material's vertex program places them at rest, evaluated when the model is decoded ([Vertex programs](#vertex-programs)) |
| Inputs | UV0 and UV1, the world normal, the decoded tangent and the whole `TEXCOORD3.w` (attribute `aTangent`, skinned with the vertex, interpolated as the game's interpolant; render flag 0x10000 clear), the vertex colour, front face, and the view from `vTintViewOffset`. A mesh without a tangent stream gets the UV-derivative frame (`native_glsl.FRAME_HELPERS`). |
| Textures | A sampled array slice as stored. Arrays with a run-time slice and 3D textures as the padded atlas of the Blender export, without mipmaps. A cube map as a GL cube map of its six stored faces (face k to `GL_TEXTURE_CUBE_MAP_POSITIVE_X + k`, rows top first), top mip, linear, seamless where the driver has it: D3D and GL share the face table, so `textureCube` along the game direction returns the D3D texel. Float data formats as `RGBA32F`. |
| Lighting | The review rig with the Rift Apart BRDF: GGX α = (0.9725 − 0.7514 gloss)², Smith joint approximation, Schlick F90 = 1, Lambert, occlusion on ambient light only |
| Blending | Forward materials draw in the float scene target, tested but not writing depth. A multiply pass scales what is behind by the background factor (blended: mix(1, B, α); additive: 1 + αB; modulate: the tint). B adds lit_background × the rig's diffuse light at the surface normal (`rcraDiffuse` with albedo 1 and no occlusion) for a material that lights the scene behind it. Then a colour pass adds the premultiplied colour, coverage blended over. A program in scene mode reads `BACKGROUND` from a copy of the frame taken before its batch (`_scene_copies`): scene-linear in the float scene target, display-referred otherwise. |
| Texture sizes | Programs that query a texture's size () get it from the full-resolution texture, as in the Blender export. Without it 143 materials failed in the viewer; with it 31 do (29 have no graph, 2 fail the G-buffer writer). |
| Caching | The translation (`viewer_materials.translate`: recipes, blend, limitations, bindings, or the reason it fails) is kept across sessions in the source's translation disk cache, keyed by the material and graph bytes and `CODE_SHA256` (the Rift Apart reader, `sm5`, `dxbc` and `rcra_glsl`). Infinite and NaN constants are stored tagged. Atlases are kept per source (least recently used out past 256 MB), since materials share their texture arrays, and 8-bit atlases are built in bytes. |
| Non-finite constants | Receipts resolve constant NaN by D3D rules (`surface.fold_nan`, as the Blender export does), since GLSL leaves min, max and clamp of NaN undefined. The inf and NaN constants that remain (a division by a zero profile value, a NaN in a select arm) are made at run time from a uniform that is never set, `rcraInf = 1 / uRcraZero` and `rcraNan = uRcraZero / uRcraZero`: GLSL 1.20 has no literal for them, and the compiler cannot fold a uniform. |

**Vertex colour.** The engine model vertex program loads a colour, squares it channel by channel and passes the square on as `TEXCOORD4`, so the square is interpolated. Which colour it loads depends on the draw:
- **Model colour:** when the subset's colour offset (`g_ModelSubsetView` dword 1) is not negative, `g_MainColorVBView` at half that offset plus the vertex index with render flag 0x8000, else `g_MainPaintVBView` at the offset plus the vertex index.
- **Object paint:** when the subset's paint offset (dword 3) plus the object's (`g_SceneObjectView` byte 0x4C) is positive, `g_MainPaintVBView` there, plus the vertex index, replaces it.
- **No colour stream:** 1, or 0 with render flag 0x800.

The viewer reads the model's colour section (`0x5CBA9DE9`) as normalized bytes in R, G, B, A order. The translated materials' vertex program (`rcra_glsl.vertex_source`, attribute `aRcraColour`) squares it. The game sets the views' formats at run time, so this channel order is the reader's, not read from the game. A model without the section reads 1. Object paint and the paint stream are not drawn: the viewer takes render flag 0x8000 as set and 0x800 as clear.

`tests/test_rcra_viewer.py` draws the swarmer through the model viewport (Windows, `RCRA_GAME`) and checks that the translated programs link and draw. It also draws a pressure button whose material reads `TEXCOORD4.x`, and checks that its colour bytes reach the program: the frame changes when they are replaced by 1. The swarmer's materials read `TEXCOORD3`: its decoded tangent is uploaded, and the frame changes when the tangent is turned a quarter turn about the normal.

**Tangent checks.** On the swarmer, the decoded frame and the UV-derivative frame draw the same picture, apart from silhouette edges and the grille, where the per-pixel derivative frame is noisy. On the helper bot's `.blend`:
- the stored coordinates rebuild the decoded tangent at every corner within 2.4 × 10⁻⁷;
- a Cycles bake of the material's own `TEXCOORD3` agrees with it at a median of 4 × 10⁻⁴, and 99.9 % of corners are within 0.01;
- 13 corners (0.02 %) differ by more than 0.1:
  - 12 lie on four zero-area triangles, which the bake leaves black and which nothing draws.
  - One lies on a triangle of 7.5 × 10⁻⁶ square units whose corner tangents differ. The bake samples inside that triangle, not at its corner.
- Before the back-face fix, four more corners were flipped. They lie on two folded triangles, whose corner normals oppose their winding, so Cycles shades them as back faces.

### Not translated

- **Blended materials whose forward program does not translate** (3 of 1,747), or that read the scene behind in scene mode (27): Blender gets their G-buffer surface, opaque.
- **Forward lighting:** the forward program's own light loop is not used. Blender's lights shade the G-buffer surface instead, which assumes that the forward program lights the same terms as the deferred apply.
- **Lighting classes 1 to 5:** skin, hair and the other special classes need their own apply programs. They are lit with class 0.

## Evidence tool

```bash
python -m rc2016.games.rcra.inspect --game <install> --output-root <folder> --summary --list model --verify
```

`--verify` decompresses every block of every installed archive once, and checks that every toc row lies inside its archive. `--extract <asset path>` writes an asset's raw bytes, plus a `.stream` file for a streamed texture.

## Tests

`tests/test_rcra_archive.py`, `tests/test_rcra_model.py`, `tests/test_rcra_texture.py`, `tests/test_rcra_glass.py` and `tests/test_rcra_animation.py` each have two parts:

- **Synthetic tests** build a small toc, dag and set of DSAR archives, a two-mesh model, texture and material assets, clips (plain and with a frame table) or skin batches (with a palette and a joint base).
- **Real-install tests** read `RCRA_GAME` (the install folder) and skip when it is not set. They also compare bytes against ALERT's own TOC2 reader.

`tests/test_rcra_model.py` also checks the tangent decode. Without game files it covers a round trip of the packed fields, the viewer payload (display basis, whole `TEXCOORD3.w`) and the glTF attribute. With `RCRA_GAME` it finds the engine model vertex programs in the exe and translates them, and checks their `TEXCOORD3` against the decoder on Ratchet, with and without render flag 0x10000. It also checks that the tangent follows Ratchet's UV frame.

`tests/test_rcra_vertex.py` checks the vertex-program path. Without game files, it checks the vectorised evaluator against the translator's operations bit for bit, the view check, the stored vertices as a glTF morph target, and the notes in the loader. With `RCRA_GAME` it checks that the swarmer's vertex-animation texture moves every mesh, with each vertex where the translator's own evaluator puts it. It also checks that a dimensional glitch is view-dependent and that a balloon keeps its stored vertices.

`tests/test_rcra_cube.py` checks the cube-map path without game files: the D3D cube lookup at texel centres, the equirectangular image against that lookup and Blender's mapping, the viewer's `samplerCube` emission, and (Windows OpenGL) that the viewport's GL cube map returns the same texels for the game direction. `tests/test_rcra_surface.py` checks the surface terms: packed again with the engine's formulas they give the writer's own G-buffer bits, on four real graphs. It also checks the texture writer and the export wiring, and, with Blender installed (`RC2016_BLENDER` or the default path), bakes an emitted material against its recipe. `tests/test_sm5.py` compiles small HLSL programs with Windows' `d3dcompiler_47` at test time, checks the decoder against `D3DDisassemble` and the translator against known results (branches, loops, switch, bit packing, discard, reflection names, UAV stores). It skips without the DLL. `tests/test_rcra_shaders.py` (`RCRA_GAME`) checks the pass table, the glass forward program against the hand decode on the trooper dome, and the `basic_normal_gloss` G-buffer writer.
