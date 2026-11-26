# Rift Apart (PC)

The Rift Apart reader lives in `nwo/games/rcra/`. Rift Apart is a PS5 game; the reader takes its Steam PC release, not a PS5 dump, and reads the install directly: nothing is extracted, and nothing is written to the game folder. This page covers the archive layer, model geometry, textures, the material texture slots behind the viewer's labelled preview, and the translation of the games' own shader programs.

The layouts below were taken from two references and then checked against the installed files:

- ALERT `dat1lib` (GPL-3.0, vendored in `third_party/ALERT`)
- RCRA-Forge (GPL-3.0), for mesh, joint and texture header layouts

## How the browser picks the game

The folder chosen as the game folder picks the game, and the browser shows only that game's assets. `nwo/games/` holds one `Game` record per game. `games.detect(folder)` tries them in this order:

| Game | Recognised by |
| --- | --- |
| Rift Apart | A `toc` whose first u32 is `0x34E89035`, a `dag` file and a `d/` folder |
| R&C 2016 | Exactly one extracted archive root under the folder (`layout.csv`, `toc`, `dag`, `chunkmap.txt` and segment files) |

A folder that matches neither gets an error naming both games. With the Transform only backend, a folder that matches neither is read as R&C 2016 loose files.

Once the scan recognises the game:

- The window title becomes "NWO Asset Browser — <game>".
- Settings keep a folder and an output root per game (`game.<id>.source`, `game.<id>.output`) beside the last-used `source` and `output`.
- An installed game, under Program Files or in a Steam library, never receives output. Its default output root is `Documents\NWO Asset Browser\<game> Output`.

Each `Game` record declares its capabilities and supplies hooks: model decode, material textures and rules, texture header, texture listing, loading and export, export material maps and shadow-only flags. Dispatch happens once per source open and once per model decode or export, so no hot loop sees the seam. R&C 2016's record names the functions it used before the seam, and its caches stay directly in `.cache/`. Other games cache under `.cache/<game id>/`.

### What Rift Apart supports

Rift Apart's capabilities are `models`, `textures`, `export`, `animation`, `root_motion` and `face_performances`:

- **Supported:**
  - Browsing.
  - The 3D model view, with skeletal [animation](#animation) playback, optionally with the clip's [root motion](#root-motion), and the voice lines' [face performances](#face-performances). The Shaded mode draws each material's own translated program where it translates ([In the viewer](#in-the-viewer)), and the labelled preview elsewhere.
  - The Textures tab.
  - Model export to GLB, glTF and OBJ (FBX, USD and `.blend` through Blender), with skin weights and animation clips (optionally with their root motion, and with face performances), and blend shapes as morph targets. Blender exports carry the game's own materials where the G-buffer writer or, for blended materials, the forward program translates ([Blender materials](#blender-materials)). Options -> Blender Scene (.blend) writes the `.blend` alone as a render-ready scene: the shared review lighting and HDRI, a framing camera and the Render Region ([viewer](viewer.md#portable-export-extract-selected)). The lighting is review lighting, or a level's atmosphere while the 3D view shows Game atmosphere ([Atmosphere lighting](#atmosphere-lighting)).
- **Disabled:** the Level and Flash UI types, the backend choice, the Blender scene's group `.blend` and game tone curve (R&C 2016 captures), and the shading filter. Each is greyed out with the tip "Not yet available for Rift Apart". Rift Apart has no Flash UI to show there ([User interface](#user-interface)).
- **Visual effects:** the model view draws the quad and model nodes of the effects a model's actors start, with a still of each emitter's particles replayed from the game's spawn and update code, and camera-facing nodes turned toward the view ([Effects in the viewer](#effects-in-the-viewer)). Lights are decoded and listed, not drawn. Trails, decals, lens flares and child effects, and the exports, leave effects out.
- **Clips:** found through the dag by `Game.animation_library`, without a scan's reference index.
- **Model list:** read in-process through `Game.list_models`, without the pipeline subprocess.

The portable export (GLB, glTF, OBJ) writes only what is decoded, through `nwo/games/rcra_export.py`. Blender formats replace these materials with the translated ones ([Blender materials](#blender-materials)). Each material gets its normal map at full resolution. Its other maps come from its own translated programs, per group of terms: albedo; F0 and gloss; occlusion; emission; and, for an alpha-over material, the forward alpha. Each group is taken from the first of these that applies:

1. **Baked over UV0** (`nwo/games/rcra/bake.py`):
   - **When:** every operation of the group's terms reaches no input but UV0 (`TEXCOORD.xy`) and the front-face flag, and samples only 1D or 2D textures or one array slice per texel (`bake.blocker`; the test is structural).
   - **Zero weights:** branches the program weights by a constant 0 are taken out first (`bake.fold`): x × 0 becomes 0 where x reaches no operation that can make NaN or infinity from finite values, and x + 0 becomes x. Many enemy writers mix a UV1 damage mask in at weight 0, and the mask then no longer blocks the bake. On 40 freed materials the folded bake equals the unfolded program at random UV1 and other inputs.
   - **How:** the terms are evaluated at the texel centres of a grid in UV0 space, with every tint, palette, mask, detail map and blend the program applies there. The grid is the largest texture they sample (at most 4096 per side); a texture read at a scaled UV is resampled at that grid.
   - **Sampling:** bilinear with wrap at the top mip, as the Blender export samples, with sRGB decoded before filtering. A sample at an explicit mip, bias or gradient reads the top mip, with a note.
   - **Evaluation:** the SM5 graph is run over arrays (`bake.evaluate`, the vertex programs' vectorised operations). On the install it equals the scalar evaluator at every checked texel, and where a writer reads plain `_g` channels the baked F0 and gloss equal them within the G-buffer's 8-bit rounding.
   - **Cost:** a 2048² material takes about 2 to 3 seconds.
2. **Plain channels** (`rcra_export.decoded_maps`, for the few writers whose terms reach UV1 or the normal on a path that does not change them). The surface terms are evaluated at six random sets of inputs and texels; a form counts only when it holds at every probe, sampled at UV0 and array slice 0:
   - F0 and gloss equal a [`_g` layout](#_g-and-_m-channels)'s channels within the G-buffer's 8-bit rounding.
   - Occlusion equals `_m` blue.
   - Emission E is a constant K or K times one channel of `_c`, `_g` or `_m`, with a plain `_c` albedo (`decoded_emission`).
   - Forward alpha is K times `_c` alpha or a constant K below 1 (`decoded_alpha`).
3. **Stand-in:** the `_c` map as the base colour, non-metallic and fully rough, no occlusion, no emission, opaque. The notes give the reason, such as the term that reads UV1, the normal or the view.

How each group reaches glTF:

- **F0 and gloss** are converted to metal/roughness with the Khronos spec-gloss conversion (as for R&C 2016). Roughness is (`0.9725 − 0.7514 gloss`)², so glTF's GGX alpha (roughness²) is the game's r⁴ ([G-buffer and deferred lighting](#g-buffer-and-deferred-lighting)). The game's Lambert diffuse without glTF's energy split is not reproduced.
- **Occlusion** is the red channel of the metal/rough image, which glTF's `occlusionTexture` reads. A baked occlusion of 1 everywhere is left out. OBJ has no occlusion map.
- **Emission** is the self-lit term, albedo × E. The emissive map is albedo × E divided by the largest E, at that largest E as strength (`KHR_materials_emissive_strength`).
- **Alpha:** an alpha-over material (flags blend "blended") is glTF `BLEND`, with the baked alpha in the base map's alpha, or a constant factor. A forward alpha of 1 stays opaque.
- **Sides:** a material whose flags word sets DoubleSided ([Flags word](#flags-word)) is glTF `doubleSided`; the others are single-sided. On a double-sided material a term that reads the front-face flag does not bake, as its back faces differ.
- **Locators:** each locator ([Locators](#locators)) is an empty node named after it, a child of its joint with the stored transform.
- **Not carried:** additive and modulate materials have no glTF blend and stay opaque. Terms that depend on the view, the normal, UV1 or the vertex colour are not carried.

Over the install (5,274 materials with a G-buffer surface, 1,306 alpha-over):

| Group | Carried | Of which baked | Mostly blocked by |
| --- | --- | --- | --- |
| Albedo | 3,787 baked (the others keep `_c`) | 3,787 | UV1, the normal and view |
| F0 and gloss | 4,166 | 4,164 (1,554 constants) | UV1, a volume or cube map, the normal |
| Occlusion | about 1,245 | | UV1, the normal |
| Emission | about 916 | 911 | an albedo or E that reads UV1 or the view (about 444 left) |
| Alpha-over alpha | about 357 | 350 | the view (soft-particle and distance fades of effects), the normal |

The zero-weight fold accounts for 91 of the baked albedos and 7 of the F0 and gloss groups; it also frees 148 occlusion and 150 emission groups, which the occlusion and emission counts above do not yet include (a freed group is carried only when its baked value is not 1 or 0 everywhere).

Additive (427) and modulate (11) materials stay opaque, and 139 alpha-over materials have a forward alpha of 1. Each has only a forward program, and its colour and alpha both bake over UV0 for 39 additive materials and no modulate one: most read the view and the normal (rims and fades) or the vertex colour.

A texture stored in a float or 16-bit data format is used as its sampled values rounded to 8 bits when they all lie in 0..1 ([Float and 16-bit textures](#float-and-16-bit-textures)), with a note. This carries the flat 4×4 `R32G32_FLOAT` `default_pink_shader_n` placeholder of 386 materials and the `R16G16` normal of the escape ship. A texture with values outside 0..1 is left out with its range. The bake samples float textures as stored.

## User interface

Rift Apart has no Flash (Scaleform GFX) UI. Its menus and HUD are Coherent Gameface pages: HTML, CSS and JavaScript run by V8 (`cohtml.WindowsDesktop.dll`, `RenoirCore.WindowsDesktop.dll`, `v8*.dll`), loaded from `coui://ui/loaded/...` addresses. No asset in the install carries a SWF or GFX header, and the dag names no `.gfx`, `.swf` or `.html` path.

| Assets | Where |
| --- | --- |
| UI images | 2,099 named `ui/**.texture` assets (the Textures tab) |
| Pages, scripts, styles, SVG art, fonts | 567 assets in `d/userinterface` that the dag does not name (`_unnamed/d_userinterface/<id>`): 461 SVG, about 60 JavaScript, 21 CSS, 11 HTML, 10 TrueType/OpenType. A path's CRC-64 is its id, so the paths the executable and the pages mention name some of them (`ui/loaded/exported/<View>/<View>.html`, `ui/loaded/authored/<View>/js|css/...`) |
| Level maps | Flat 3D models, `models/ui/map_<location>.model` (14), drawn with `material/ui/levelmap_base`, `MapDarkBase` and `Map_01`: a backdrop, the playable area and highlighted landmarks. They open as models; map icons are separate UI images |

The browser does not render these pages.

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

`nwo/games/rcra/model.py` decodes a `.model` asset into the same dict schema as `formats.native_model.decode_model` (`schema='rcra-pc-model-v1'`). Because the schema matches, `viewer.loader.mesh_payload`, `_selected_meshes` and the glTF exporter use the result unchanged. `decode_for_viewer` is the `Game.decode_model` hook.

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
| `0x811902D7` | Look names: one 80-byte record per look. The first u32 of the section is the records' byte size; each record ends with u32 count, u32 name hash, u32 lowercase-name hash and u32 name pointer (+64). The 16 u32 before them and the data after the records are not read |
| `0x9F614FAB` | [Locators](#locators), 64 bytes each |
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

### Looks and locators

**Look names** (`looks[i]['name']`): every model names its looks. All 11,387 have `Default` as look 0. The others name alternative mesh sets: `base`, `topiary`, `pristine` and `damaged` pairs, `coll` (collision geometry), `skeleton`, and the `_chunk` pieces of breakable models. The model view's Look box shows `index: name`.

### Locators

`decode_model` returns `locators`: the named attach points of section `0x9F614FAB` (580 models, 10,126 locators, every name `igLoc_…`).

| Bytes | Field |
| --- | --- |
| 0 | u32 name hash |
| 4 | u32 name pointer |
| 8 | i32 joint; −1: on the model (74 locators) |
| 12 | u32 0 |
| 16 | 3 × 4 f32 transform, stored column by column: the three axis columns, then the translation |

- **Space:** the transform is relative to the joint. On Ratchet, `igLoc_belt_buckle` is 0.068 in front of the pelvis; read in model space it would sit at his feet. A locator without a joint is in model space.
- **Axes:** the axes are orthonormal (within 3 × 10⁻⁷ on 300 models). Reading them as columns follows the inverse binds' storage; no runtime use of a locator's axes is traced.
- **Lookups:** `0x731CBC2E` is the sorted (name hash, locator) table of `0x9F614FAB`. `0xEE31971C` is the sorted (name hash, joint) table of the joints, closed by a (`0xFFFFFFFF`, `0xFFFFFFFF`) row. Both hold only what the tables they index hold.
- **Use:** the Details tab lists them, and the model export writes each as an empty node under its joint.

### Sections not read

`decode_model` returns `unread_sections`: each section it does not read, with what is known of it (`model.SECTION_ROLES`, `model.DERIVED`). `unsupported_sections` lists those with no known meaning. Names in quotes come from ALERT `dat1lib`; counts are models of 11,387. None of these is needed to draw the geometry.

| Section | Models | What is known |
| --- | --- | --- |
| `0xEFD92E68` | 4,907 | Havok physics data: the collision shapes ("Model Physics Data") |
| `0xCCBAFF15` | 954 | The skin weights again, as fixed records ([Skin weights](#skin-weights)) |
| `0xC5354B60` | 1,123 | Joint mirror ids ("Model Mirror Ids") |
| `0x3C9DABDF` | 32 | Spline subsets ("Model Spline Subsets"). Only in Ratchet's and Rivet's bangle models, always with `0x27CA5246`, `0xB25B3163` and `0xBB7303D5` |
| `0x7CA37DA0` | 11 | Ambient shadow primitives ("Ambient Shadow Prims"), 48 bytes |
| `0x707F1B58` | 9,216 | Unknown: u32 16, u32 data size, counts, then float matrices |
| `0x4CCEA4AD` | 11,387 | Unknown: holds look-name hashes and look indices |
| `0xDF9FDF12` | 11,104 | Unknown: 16-byte records |
| `0x90CDB60C`, `0xB7380E8C` | 1,123 | Unknown, in every rigged model: 80 bytes; u16 values |
| `0x0AD3A708` | 958 | Unknown: a count and 20-byte records, in skinned models |
| `0x665DA362` | 371 | Unknown: 8 or 16 bytes |
| `0x27CA5246`, `0xB25B3163`, `0xBB7303D5` | 32 | Unknown: with the spline subsets |
| Seven more | 1 to 22 | Unknown: `0xBCE86B01`, `0xADD1CBD3`, `0x9A434B29`, `0x5A39FAB7`, `0x855275D7`, `0x8A84E4D6`, `0xCD903318`, `0x00823787` |

The morph sections are read by `attach_shapes` ([Blend shapes](#blend-shapes)), and the joint rest transforms `0xDCC88A19` by the skeleton.

### Models without geometry

24 models have no vertex or mesh section:

- **22 river and lake models** under `levels/i29/…/houdinigenerateddata/` (Savali rivers, the catacombs water). Each holds only its bounds, an empty look and section `0x707F1B58`, about 2 KB.
- **`garden_planter_bush_x2_stump`:** one joint, locators and a Havok collision shape.
- **`…_finalcrystal`** (a Blizar grind rail): a Havok collision shape only.

Their 36-byte header names no streamed sections: its streamed size is 0. In the install that is true of exactly these 24 models. So `Game.list_models` marks them (`PipelineModelCandidate.no_geometry`) from the toc alone, and the browser's Variants menu hides them by default ([viewer](viewer.md)). Opening one raises `UnsupportedModelError` with a message naming what it holds (`model.geometry_free_note`).

## Animation

Rift Apart clips play in the 3D view and export with the model, like R&C 2016 clips. The decoder is
`nwo/games/rcra/animation.py`; `nwo/games/rcra/clip_library.py` finds a model's clips and is the
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
refused. The root motion section `0xE08AA35F` is read by `motion` ([Root motion](#root-motion)); the morph sections
are read by `rcra.morph` ([Blend shapes](#blend-shapes)); triggers are not read.

**Path section** `0x3976E44C` (in 23,295 clips, always 128 bytes): two 4x4 f32 matrices, row-vector layout (three
basis rows, then the translation row; the last column is always 0, 0, 0, 1). In 22,873 clips both are a rotation
times a uniform scale; in 11,683 the two are equal and in 2,454 both are the identity. The game does not read it:
its id occurs nowhere in `RiftApart.exe` (code or data), while the clip loader names the root motion section
(`0x140F27724`). It is not root motion, a joint's transform, or the root motion start: in Ratchet's clips the
matrices sit about 1.2 units up, with a 9° pitch and, in some, a 1.5-unit sideways offset, and match no joint's world
transform. Neither the viewer nor the export uses it.

**Scale sentinel.** In 10 additive clips (bladeball turns, horned-toad aim and look layers, the emergency-raft
wobble) a joint's packed scale is `0x8000` on all three axes, which would be −8 or −16 (exponent 12 or 11).
`packed_scales` reads it as one, the identity of a layer's scale factor. The corpus supports a marker rather than a
value: over all skeletal clips the value occurs only in these additive clips (41 records), never with another value on
the same joint, and every other scale of an additive clip is a factor near one (5,420 records exactly one, the rest
0.97 to 0.999), so no arithmetic reading of `0x8000` (±8, ±16) fits beside them. Every such joint also has an identity
base rotation, no scale channel and a translation under 0.01. It is still not traced in the runtime: the clip loader
(`RiftApart.exe` 0x140F27100) keeps the base-state pointer at +0x88 and the scale-tail block count at +0x7A of its clip
object, but the code that reads the tail was not found, and the animation code holds no scalar compare against
`0x8000` or −32768. The same value in a clip without the additive flag, and any other high-bit scale, is still
refused (as in 2016, whose clips carry no additive flag).

**Coverage.** Of the 23,317 installed clips, 21,377 are skeletal and all of them decode. The other 1,940 are not
skeletal: custom-track clips and one camera-shake clip. No decoded clip uses the collapsed-rotation sentinel.

**Check.** On `amb_fish_01`, frame 0 of `Swimming_Idle` matches the model's rest S/Q/T with a median translation
difference of 1.1e-4.

### Root motion

The joint tracks play in place: a walk, a dash or a cutscene shot's travel through the scene is not in the body
joints but in the clip's root motion section `0xE08AA35F`, which every clip carries. `rcra.animation.motion` reads
it:

| Offset | Field |
| --- | --- |
| +0 | u8 record format (0 or 1), u8 kind (0: no records, 2, 3), u16 record count |
| +4 | f32 start rotation xyzw |
| +32 | f32 1.0, then f32 start translation xyz |
| +60 | u32 record offset from +124 (0 in every clip) |
| +64 | u32 bytes from +124 (the records, zero-padded so the section is a multiple of 16), u16 frame count, u8 count of 80-byte entries at that offset (0 in every clip) |
| +76, +100 | f32 minimum and maximum xyz of the root translation over the clip |
| +124 | the records |

A record is the root's rotation q and translation t relative to the start: the root sits at rotation `start_q * q`
and translation `start_t + start_q * t`. There is one record per frame, or one more (a closing record, as with
the frame table). Record 0 has the identity rotation.

- **Format 0** (16 bytes; animset clips): i16 rotation xyzw with unit length at 32767, i16 translation xyz, u8
  exponent e (the translation is the integer times 2^-e, 12 in most records), and the i8 foot phase.
- **Format 1** (24 bytes; mostly cutscene shots): a u64 smallest-three rotation, f32 translation xyz, and a u32 whose
  low byte is the i8 foot phase. The rotation word holds three signed 20-bit fields from bit 0, each
  times sqrt(1/2) / (2^19 - 1). They are the components other than the largest, in xyzw order, and bits 60-61 give
  the largest one's index. That component is sqrt(1 - the others' squares), so it is positive.

**Foot phase.** Kind-3 (locomotion) clips set it in every record: (2 × i8 + 1) / 255, so −128 is −1 where the left
foot lands and 127 is +1 where the right foot lands, linear in between. The game's root-motion sampler
(`RiftApart.exe` 0x140F29C00, 0x140F2A100) computes it this way, and reads the byte only when bit 0 of the kind byte
is set (kind 3). In Ratchet's `neutral_strafe_walk_n` it is −1 at frame 0 with the left
foot planted and +1 at frame 14 as the right foot lands, and his strafe run is the same over 20 frames. Of the 1,108
format-0 kind-3 clips, 845 vary and 263 hold one value (−91 in `walk_forward_wrench`); 164 format-1 kind-3 cutscene
clips carry the same triangle in the u32 (all below 256). Kind-2 clips leave it 0. In 7 kind-2 clips (the spitter's
distortion projectile and six fixer-scaffold props of one cutscene) the u32's high 16 bits hold a smooth per-frame
curve (as a half float, 1.45 to 2.26 and 169 to 195); the sampler reads only the low byte, so the game does not use
it. `motion` returns the phase as `phase` (None unless kind 3); playback does not use it. How the runtime uses it (for example to match the step of blended locomotion clips) is not traced.

**Coverage.** All 23,317 installed clips read. 10,464 have records (3,700 in format 0, 6,764 in format 1). In each
of them the stored minimum and maximum equal those of `start_t + start_q * t` within quantization: format 0's
exponent 8 is a step of 1/256. The other clips have no records, so their root stays at the start. Every format-0
rotation has unit length at 32767. Across a change of the largest component, the format-1 rotation is continuous
up to sign.

**Checks.** In Ratchet's `walk_forward_wrench` the `sync` joint's world translation is minus the root's (start plus record),
while `pelvis` does not move. Ratchet's 90° idle turns, walks and runs keep a planted foot in place only with the
root motion added. Without it, the foot slides at walking speed. Speed of the planted foot, median over the clip,
in units per second (`tests/test_rcra_animation.py`):

| Clip | In place | With root motion |
| --- | --- | --- |
| `walk_forward_wrench_hero_ratchet` | 1.01 | 0.10 |
| `idle_turn_left_90_right_foot_hero_ratchet` | 0.47 | 0.005 |

In 637 of the 734 moving clips offered to Rivet's model the pelvis stays in place while the root travels. Among them are 229 of 247
cutscene shots, whose roots travel up to 1,945 units through the scene. The others move the body inside the clip
as well, such as jump arcs and attack lunges.

**Playback.** `RcraAnimationLibrary.root_displacement` gives the root's motion since the clip's first frame
(record 0 inverted, times record f). A closing record past the last frame is not played. With `root_motion`,
`AnimationLibrary.clip_frames` applies it to the joints without a parent: their local T R S becomes D T R S. This
drives the viewer's **Root motion** box ([viewer](viewer.md#animation-playback)) and the export's *Root motion* option.
The scene placement of the start transform is not used.

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
4. **Face performances** (3D view, or export with Face performances): the voice lines of the speaker whose face pose
   clip the model plays ([Face performances](#face-performances)).

Each clip is still classified on its first frame, as in 2016. The clip list hides one that does not drive the
skeleton. For example, `hero_ratchet.model` gets:

- 1,380 animset clips from five animsets, among them `hero_rivet.animset`, which an actor pairs with the model.
- 532 cutscene shots.
- Of those 1,912 clips, 212 are additive layers, 38 are unplayable and 20 are copies.

The clip notes take about 8 s to compute, off the Tk thread.

### Face performances

Voice lines carry their lip sync as `.performanceclip` assets, one per line, under
`localization/<speaker>/` (20,885 paths, 14,408 in the install). Each speaker's
`localization/<speaker>/<speaker>.performanceset` depends on the speaker's face pose clip. The speaker's lines
belong to its `<speaker>_<group>.animset` files. 67 of the 200 sets name one of 21 pose clips; the others (credits, menus,
generic voices) do not, and their 2,080 lines have nothing to drive.

**Performance clip** (`rcra.animation.performance_clip`). A built header with no joints (fps, seconds and frame
count as in a clip) and the samples section. That section holds one 16-byte row per frame, zero-padded to a
multiple of 8 rows: eight u8 performance indices, then eight u8 weights out of 255. A frame's weights sum to 255
(within one, rounding) in 98.7% of frames; an index with weight 0 is unused. 31 parade and crowd lines keep
weighted rows past the header's frame count; they are not played. 28 lines have no built section and do not read.

**Pose clip** (`pose_table`). An additive skeletal clip of 25 frames (39 for Kit and Clank) on the jaw, teeth and
tongue joints (7 for Ratchet; up to 46 for Skidd). Its section `0x1D4BD9FA` is a table of 40 u16 pose frames, one
per performance index; in all 21 tables every value is an even frame. The performance indices are the 40 phonemes
of the game's `kPhoneme` enum (named in `RiftApart.exe`'s reflection data; `animation.PHONEMES`): `x` (silence), `IY`,
`IH`, `EH`, `AE`, `AH`, `UW`, `UH`, `AA`, `AO`, `EY`, `AY`, `OY`, `AW`, `OW`, `l`, `r`, `y`, `w`, `ER`, `m`, `n`, `NG`,
`CH`, `j`, `DH`, `b`, `d`, `g`, `p`, `t`, `k`, `z`, `ZH`, `v`, `f`, `TH`, `s`, `SH`, `h`.

Section `0x495BA079` names the pose frames: (u32 name hash, u32 frame) pairs sorted by hash (`pose_names`). The clips
hold only the hashes; all 30 names of the install's 22 pose clips were recovered by their game CRC-32. Every clip has
the 13 visemes `vis_xxx`, `vis_aaa`, `vis_eee`, `vis_iii`, `vis_ooo`, `vis_uuu`, `vis_ow`, `vis_www`, `vis_lll`,
`vis_rrr`, `vis_fv`, `vis_bmp` and `vis_nnn`; Kit and Clank add eye and mouth poses (`Eyes_Open`, `Left_Eye_Closed`,
`Mouth_Open` …) and Skidd eye, brow, lid and mouth poses (`eyes_up`, `brows_down`, `lids_closed`, `Smile` …). The
phoneme table agrees with the names: `x` plays `vis_xxx`, `v` and `f` play `vis_fv`, `m`, `b` and `p` play `vis_bmp`,
`l` plays `vis_lll`, `r` and `ER` play `vis_rrr`, and 11 consonants (`n`, `NG`, `d`, `t`, `k`, `s` …) share `vis_nnn`. Every one of the 12,269 lines whose speaker has a pose clip stays inside its
table. Index 0 maps Ratchet to frame 20, whose jaw is closed.

**Blend** (`performance_frames`). Per frame, the weighted blend of the indexed pose frames: translation and scale are
weighted sums, and the rotation is the normalised weighted sum of the quaternions in the identity's hemisphere. A
weight the frame leaves unused counts as the identity delta. This blend is a reading of the data, not traced in the
runtime. The result plays on the rest pose like an additive layer, without root motion.

**Finding them** (`clip_library.performance_sources`). A set is the model's when its pose clip lies in the model's
folder (or its animation subfolder) or, only when none does, is one of the model's offered clips, and the model has
at least half of the pose clip's joints. Rivet's actor also offers Ratchet's clips, so the folder comes first.

| Model | Lines | Pose clip |
| --- | --- | --- |
| `hero_ratchet` | 1,535 | `ratchet_phonemes` |
| `hero_rivet` | 1,659 | `rivet_phonemes` |
| `hero_clank_ps4` | 1,077 | `clank_phonemes_2` |
| `hero_kit` | 724 | `visemes_hero_kit` |
| `npc_civ_average` | 207 | `phonemes_npc_civ_average` (19 of its 24 joints) |

The Blizar miner's model has 2 of its pose clip's 30 joints, and the civilian robot's pose clip has float tracks
only, so neither gets lines. In the browser the lines form their own section of the clip list, Face performances.
`tests/test_rcra_animation.py` checks Ratchet's `RATC_BOARDS_SLICE1_017`: 497 frames, the jaw closed at silence and
open up to 9.4°.

**Lips.** The pose clips also carry blend-shape weights. A line blends them by the same performance weights as the
joints, so the lips move with the jaw ([Blend shapes](#blend-shapes)).

### Blend shapes

`nwo/games/rcra/morph.py` reads the face shapes of 19 models (Ratchet has 294) and the shape weights of 2,821
clips: the face pose clips, cutscene shots and some animset clips. Section names follow ALERT `dat1lib`; the layout
is in the module docstring.

**Model** (`model_shapes`). The morph info section `0x380A5744` lists the shapes, each with a name, a left/right
mirror partner and one target per mesh it moves. A target's deltas are packed k-bit fields (position xyz, and for
most shapes normal xyz) times a scale plus a bias, in `0x5E709570`. The vertices they move are (skip, run) words in
`0xA600C108`, in windows of 2,560 mesh vertices. On Ratchet the deltas of `LF_x` and `RT_x` at mirrored vertices
differ by a median 0.003 of the shape's largest delta, which is the rounding of their 8-bit fields.

**Clip** (`clip_weights`). The clip morph info `0x277563F5` names its channels by shape name hash, then by the offset
of each shape's name in the clip's string pool (the game's clip loader relocates these offsets; the names match the
hashes in all 2,821 clips, and `clip_weights` returns them as `names`), and holds a frame table: per frame, the count of active channels and a word offset into the frame data `0x69F64588`. There each active
channel is a bit field (the first absolute, then steps from the previous channel) and a 10-bit value. A channel a
frame leaves out has weight 0. The weight is the value / 1023; reading 1023 as full weight is inferred, not traced in
the runtime. Channels are matched to the model's shapes by hash; ones the model lacks are dropped.

**Playback.** `RcraAnimationLibrary.pose` adds the clip's weights per frame (`morph_weights`). A face performance
blends its pose clip's weight rows by the performance weights, like its joints. The viewport adds the weighted deltas
to the rest vertices and normals before skinning (`ModelViewport.set_morph`) and refills the vertex buffers.

**Export.** glTF and GLB carry each shape as a sparse morph target, named after the shape. A clip with weights also
animates the mesh's `weights`, so Blender imports shape keys driven by an action.

`tests/test_rcra_morph.py` checks synthetic shapes and clips, the viewport blend and the sparse accessor. With
`RCRA_GAME` it checks Ratchet's brow shapes against their mirror partners, the silent frame of `ratchet_phonemes`,
and that `RATC_BOARDS_SLICE1_017` drives 53 of Ratchet's shapes.

### Additive layers

1,033 clips carry the additive flag (built-header bit `0x40`): aims, head looks, leans, twitches, face pose clips. The
game adds them to what its animation graph plays. An animset's graph is its driver classes, which
`rcra.animation.driver_graph` decodes (layout in the module docstring):

`RiftApart.exe` carries reflection data for the graph's source types (`AnimDriverDef`, `AnimDriverBiBlendNodeDef`,
`AnimDriverSelectNodeDef` …): their field and enum names. They name the built kinds below, matched by fields and by
the binding targets; the built layout itself is read from the data.

- **Class** (`AnimDriverDef`): an 80-byte record in `0x66CA6C6F`, a 16-byte row in `0xA40B51D2` (offset of its data
  in `0x73CEE17F`), and that data: nodes, then 12-byte variable bindings. Record words: name hash, slot count,
  flags, data size, root node, node count, binding count and offset, and at +56 four transition durations in
  seconds (`TransitionDurationDefault`, `FromNone`, `ToNone`, `Self`: 0.25 in 3,394 classes, 0.1 for deaths, 0 for
  346). Flags bits 0-1 are the layer: 0 base (3,916), 1 partial (49, such as `Wings_Out`, `Spawn_Partial_1`), 3
  partial additive (234, such as `React_Twitch`, `Emergency_Raft_Wobble_Additive`). Bit 2 marks looping classes: by
  class name, 809 loops and idles have it and 220 do not, while 1,228 intros, outros, reacts and deaths lack it and 93
  have it. Names are game CRC-32 hashes of strings in the animset (`Celebrate`, `Node 4`, `Aim_Horizontal`). Empty
  classes have size 0.
- **Node:** a 40-byte header (name hash, kind and flags, id, time at +8, rate at +16, a fixed signature at +24 to
  +36), then its kind's fields. Children are u16 word offsets into the class data, inline or in u16 lists after the
  nodes.

| Kind | Nodes | Fields |
| --- | --- | --- |
| 1 clip | 8,896 | a reference the animset's clip lookup resolves (`clip_references`): a clip asset, or a clip group. Flag 1 (`kClipNodeFlagsIsAdditive`) on 2,383, of which 1,698 name an additive clip and 4 an absolute one (the rest name groups or clips of other sets); a u32 0 or 2 at +40 is not read |
| 2 bi-blend | 1,260 | child 0, child 1, a type byte (`kBiBlendNodeType`: 2 combine additive in 1,183, 1 additive in 56, 3 difference in 21) and a factor (1 in 1,193; 0, 0.25, 0.4, 0.5, 0.7, 0.8 or 0.95 in the others) |
| 5 blend | 1,006 | a variable slot, per child a (low, high) span: 0 to 1, or angles such as −180 to −90, −180 to 0, −90 to 90 … for five children 90° apart, so each child's span reaches its neighbours' positions; the stored current value at +48 (0 in 994) |
| 7 random | 291 | its children (variations) and a u16 weight per child out of 65535 (0.8/0.1/0.1, 0.7/0.3, 0.6/0.4 …), or `0xFFFF` for equal odds (232); no binding drives it |
| 9 select | 158 | its children and the current index at +44 (`CurrentNodeIndex`), within the children in all 158; 88 are bound to a variable (`Direction`, `Ledge Jump Direction`, `Range`, `Initial_Phase_Ascending`), the others always play their stored child |
| 10, 12 | 26 | one child; kind 10 also names a variable (`Direction` on Ratchet's ledge and climb moves). Their types are not named |

- **Bindings:** u32 variable name hash, u16 slot, u16 word offset of the bound field or of a list of them, u32 flags
  (low byte set for a list, `0x100` for a float field). The bound fields are the reflection's connection targets.
  Over the install, 1,654 bindings drive a blend's current value (`kBlendCurrentValue`, 1,004 fields), a clip's time
  (`kClipTime`, 929: the head and aim layers pick their pose by time from `Head_Aim_Horizontal` and the like), a
  select's index (`kSelectCurrentIndex`, 90) and a bi-blend's factor (`kBiBlendFactor`, 15).
- **Variables** `0xDF74DA06` (84 animsets): (u32 variable hash, u32 class hash, u32 word) rows, the variables each
  class uses; bit 16 of the word is set for float variables (the select inputs are integers). The word's low 16 bits
  are not read.
- **Clip lookup** `0xB79CF1D7`: 24-byte rows (u32 hash, u32 kind, i32 group offset, u64 asset id, u32 0). Kinds 0 and
  `0x8000` name a clip by its path CRC-64. Kind 1 names a group: an inner DAT1 in `0x9FD19C20` whose entries list tag
  hashes and their clips (`React_Knockback_Group`: front, back, left and right, high and low). All 3,325 group
  entries' clips resolve through the lookup; 32 groups are empty.

Every class of the install's 321 animsets with classes parses (4,199 classes, 11,637 nodes): the header signature
finds exactly the record's node count, the root is a node and every child link lands on one. 10,439 nodes are
reached from their class root; the others are not linked to it, and 390 of them are named by nothing in their class.

**Partners.** A bi-blend whose child 1 holds additive clips adds them over child 0.
`RcraAnimationLibrary.layer_partners` lists the absolute clips under child 0 (a group's clips included) of every
bi-blend that holds the layer in an offered animset, the one nearest the layer first, then graph order;
`layer_weight` gives its factor. The thug shooter's aim idle class is a blend on `Aim_Horizontal` over three
bi-blends, (`shooter_aim_idle_fwd`,
`Shooter_Aim_Idle_Fwd_Pitch`) and its `_l090` and `_r090` siblings, so each pitch layer plays over its own idle.
Grunthor's eye looks are added at 0.5.

**Default first.** The graph order follows the stored values, as the game plays a node no variable has set
(`animation.default_children`, `graph_reach(..., default_first=True)`): a select's current index first, a random
node's heaviest variation first, and a blend's child nearest its current value first (the spans holding it, then
the nearest span centre: a blend child sits at its span's centre). So the first partner, which the viewer's
**Over**, Play all and the export's layer option start from, is the graph's default state. On the thug, Ratchet and
Rivet this changes the first partner of 16 to 18 layers each, all toward the default: strafe aims over the forward
strafe run (`_n`, was `_s`), the smash head aims over the plain smash (was its `_r090` variant), hover-fire aims over
the idle turn (was turn left), and Rivet's flying-mount layers over level flight (was `fly_down`). The partner sets
are unchanged.

| Count (skeletal additive clips, by name) | Before (clip hashes shared with a class) | Graph |
| --- | --- | --- |
| In a class | 541 of 951 | 806 |
| With an absolute base | 290 | 431 |
| `enm_thug_brawler` (offered clips) | 55 of 160 | 72 |

That a combine-additive bi-blend adds child 1 scaled by its factor is a reading of the type's name and the data
(those bi-blends hold the additive clips in child 1); the evaluator is not traced in the runtime. A factor a variable
drives, or one stored as 0 that no variable drives (game code then sets it), plays at 1.

`tests/test_layers.py` checks the parser, class fields, bindings, random and select nodes, groups, partners and
weight on a synthetic animset, the weighted
`layer_over`, the default-first order, and with `RCRA_GAME` that each thug aim pitch layer plays over its own idle
and moves it, that the thug's strafe aim starts over the forward strafe run, and Grunthor's eye-look weight of 0.5.

### Not decoded

- Game state: the values game code gives the graph's variables at a moment (a blend's input, a select's index, a
  clip's time, a factor), which random variation plays, and which entry of a clip group the game picks by its tags
  (hit direction, height). These are runtime values, not in the files; `driver_graph` returns the stored values a
  node plays without them, and playback uses them only to order the layer partners (default first). A layer whose
  clip time a variable sets (the head and aim pose strips) still plays through all its frames.
- The base of a layer whose class holds no absolute clip under the bi-blend (it plays over whatever other classes
  play), the class flag bits above bit 2, the clip node's +40 word, the types of kinds 10 and 12, and the low half
  of the variable rows' word.
- The gesture clips some performance sets also play.

## Textures

`nwo/games/rcra/texture.py` reads `.texture` assets. It returns the same dicts as the 2016 viewer's texture functions:

- `load_texture`, `load_texture_full` and `read_top_mip` return the loaded top mip. For a one-slice texture `load_texture` (the viewer's read) also returns `mip_chain`, the stored mips below it, largest first: the rest of the HD stream, then the SD chain when it starts where the HD mips end. The viewer uploads them instead of generating mips ([texture filtering](viewer.md#3d-view)).
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
- **Float and 16-bit data formats** (RG32F, RGBA16F, RGBA32F, RG16F, RG16, R16F) used by vertex-animation, sky and lookup textures: sized, and previewed where their values fit an image ([Float and 16-bit textures](#float-and-16-bit-textures)).

### Float and 16-bit textures

`nwo/games/rcra/unit_texture.py` turns these formats into image values. D3D samples a UNORM16 texel as its integer / 65535 and a float texel as the float itself, with 0 for a missing green or blue and 1 for a missing alpha. An 8-bit UNORM image stores the same values rounded to 1/255, so a texture whose values all lie in 0..1 converts by rounding alone.

- **Converted:** 6 of the 156 listed float and 16-bit textures (three `R16G16` ones: the escape-ship and eye-remap normals and a flowmap; the foam and wetness lookups; the coretest copy of the pink normal), and the unlisted `default_pink_shader_n` placeholder.
- **Kept out:** the other 150. 140 vertex-animation tables hold non-finite values, because their floats are packed bits. Ten others, including two HDR skies, hold values outside 0..1. No 0..1 image holds them without a remap the files do not define. The error gives the range or the non-finite count.
- **Textures tab:** a converted texture previews through its RGBA8 conversion; the info line keeps the stored format. Its PNG export is a 16-bit RGBA PNG, exact for UNORM16 and rounded to 1/65535 for floats. DDS export writes every data format as stored, so it keeps the values a PNG cannot hold.
- **Model export:** glTF and OBJ maps use the 8-bit conversion. The viewer's translated materials and the Blender export read these formats as float instead ([Blender materials](#blender-materials)).

## Materials and the viewer preview

`nwo/games/rcra/material.py` reads the texture slots of a `.material`. It also provides the `Game` hooks `material_textures`, `model_texture_list`, `material_rules` and `effect_preview`.

### Layout

- **Asset header:** a material asset is the 36-byte toc header (type `0x88730155`) and a DAT1 of the same type.
- **Section `0xF5260180`** begins with ten u32 (`SERIALIZED_HEADER`; every relation holds in all 5,701 installed materials that have the section):

  | Word | Field |
  | --- | --- |
  | 0 | Section size |
  | 1 | Parameter count |
  | 2 | Value block start, `40 + 8 × count` |
  | 3 | Switch count |
  | 4 | Value block end: the start plus every parameter's byte size. The switch batch starts here |
  | 5 | Texture count |
  | 6 | Texture batch start, `word 4 + 8 × switches` |
  | 7 | Texture batch end, `word 6 + 8 × textures` |
  | 8, 9 | 0 in every material |

- **Parameter records** follow the header: (u16 offset in the value block, u16 byte size, u32 key). See [Architecture glass](#architecture-glass-gbl_arch_glass) for how they fill `cb2`.
- **Switches** (`material_switches`): (u32 value, u32 key) records, values 0 or 1, in 1,557 materials. They override the graph's switch defaults in section `0xBC93FB5E`. `graph_switches` reads the first [technique](#techniques) record: 32 slots of (u32 value, u32 key), then six u32 whose first is the count in use. Graphs with 36 pass records have two records. 706 records name a key their graph does not declare. No shipped program reads a switch: 15 switch keys are also `cb2` parameters (the `gbl_rgb_blend` family, water foam, a shockwave), and no pass of their graph reads those `cb2` dwords. So the material binding ignores the switches, although 350 materials set one to a value other than the graph's `cb2` default. Whether the engine reads any of them elsewhere is not known.
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
| `0x1` | DoubleSided: both faces are drawn, without back-face culling (634 materials) | The main instance draw (`0x1411229ff`, and `0x1411250b9` for a second list) marks the packet two-sided (sort word `+0x2E` bit `0x800`) unless the mesh forces single-sided ([below](#double-sided)) |

`material_blend` returns `opaque`, `additive`, `blended` or `modulate`; `draw_flags` returns the lit bit. Over 5,721 materials: 3,974 opaque, 1,309 blended, 427 additive and 11 modulate. The blend factors are pipeline state outside the files.

#### Double-sided

`material.double_sided` reads bit `0x1`, and `mesh_double_sided` adds the mesh override:

- **The property:** the engine's reflection names the material property DoubleSided: "Both sides of geometry with this material will be drawn (no back-face culling)". Per geometry, `DoubleSidedType` can force single-sided (`kForceSingleSided`, `kForceSingleSidedInLod1`).
- **The draw:** it tests bit `0x1`. When it is set, the draw marks the packet two-sided unless the mesh record's flags (`+0x24` at run time, the file's `+36`) force single-sided. The test is `0x4000` at LOD 0 and `0x4000 | 0x2000` at a higher LOD, the LOD the draw chose for the instance. So `0x4000` is single-sided at every LOD and `0x2000` from LOD 1. A global option sets the packet bit for every mesh.
- **Corpus:** 634 materials set the bit: 265 opaque, 231 blended, 131 additive and 7 modulate. 42 % of the foliage materials set it, against 11 % of all materials. Only 3 meshes set `0x4000`, all with single-sided materials, and none sets `0x2000`.
- **Not traced:** where the pipeline turns the packet's bit into its rasterizer cull mode.

The portable export makes these materials glTF `doubleSided`, and the Blender materials turn back-face culling off for them. The viewer draws their blended passes without culling; it draws every opaque material two-sided anyway.

### Material alpha

f32 5 of section `0xE1275683` is the material's alpha (`material.material_alpha`). The loader copies it to the material object's `+0x70` (`0x1410d16af`), whose default is 1.0 (`0x1410d162a`). Of the 1,309 blended materials, 1,127 have 1, 167 a fraction and 15 have 0; the 0 group includes `invisible_projectile_model`, `null_model`, `gbl_gp_transparent_01` and the `_hidden` rift shards.

The main model-instance draw (`0x1411212d0`, [Render flags](#render-flags)) tests it for each mesh whose material is on the blended path ([Flags word](#flags-word)):
- **Draw alpha:** a = min(1 − LOD fade, instance alpha) × material alpha + a distance fade. The distance fade is non-zero only for material flag `0x20000`.
- **Test:** when a ≤ 0 (`0x141122c58`), the draw moves on to the next mesh (`0x141124adb`) without allocating a draw packet. The game issues no draw for that mesh.
- **Exceptions:** fur materials (flag `0x4000`) skip the test. So does an instance with effect-overlay records (the count in the u16 at `+0xE` of its per-instance record), which may draw the mesh with an effect.
- **The packet:** otherwise its first 16 bytes are the draw's `MatDrawCB` record: render flags, scene object, subset material, and `m_AlphaOverrides` with a × 4096 in its low half.

**Draw alpha.** The draw alpha reaches the pixel programs as the interpolant `TEXCOORD.w`:
- **Packing:** the draw writes trunc(a × 4096) to the low 16 bits of `m_AlphaOverrides` (`0x141122ac9`, `cvttss2si`).
- **Reading:** the engine model vertex program reads it as a signed 4.12 value s (instructions 30 to 38). So does every one of the 208 graphs' own vertex programs, which share that code. They set `TEXCOORD.w` = |s| − (s < 0 ? sat(|s| − 1) : 0); a negative s is a dissolve. When the field is 0, they set it to the object's alpha × `g_MaterialView.m_Alpha` instead.
- **Use:** every translated forward program (306 surveyed) multiplies its alpha by `TEXCOORD.w`. Eleven blended-path G-buffer writers use it in their discard or gloss, and no opaque writer reads it.

So a fractional material alpha is opacity. The translations bind `TEXCOORD.w` to the plain instance's value, `material.draw_alpha`: trunc(material alpha × 4096) / 4096, so 0.3 gives 0.2998 as in the game. A field of 0 keeps the neutral 1 (the profile's records). The limitations give the value.

`material.draw_skip_note` returns a note for a material the plain instance draw never draws:
- the blended path, a material alpha that is not above 0 (NaN fails the engine's test too), and neither flag `0x4000` nor `0x20000`;
- under the [neutral object](#per-object-values): full instance alpha, no LOD fade, no overlay records.

19 materials qualify: 15 blended and 4 additive. They include Ratchet's `hero_ratchet_compositeshell`, a blended layer over the whole head and body in 29 models (`_holo`, in 2 more), which binds only `default_pink_shader` placeholder textures. They also include Clank's `antennaglass` and `eyeglowinner`, and Kit's `emblemglowinner` and `eyeglowinner`. Opaque materials with alpha 0 (12) are drawn: at full instance alpha the draw takes the opaque path before the test.

`model.decode_for_viewer` lists them as `decoded['undrawn_materials']`, with the note in `material_notes`. The viewer leaves their batches out ([In the viewer](#in-the-viewer)), and the model export leaves their meshes out with the note, so the glTF, OBJ and Blender formats do not hold them ([Blended materials](#blended-materials)).

### Slot keys

Slot keys are the low 32 bits of the 2016 game's `0x49A7866B........` slot hashes. Across 5,721 materials they bind the same texture suffixes:

| Slot key | Texture | Count |
| --- | --- | --- |
| `ABDAD780` | Base colour (`_c`) | 2,600 of 2,862 |
| `7AE34E7A` | Normal (`_n`) | 2,591 of 2,647 |
| `1576D848` | `_g`: F0 and gloss | — |
| `E68930AC` | `_m`: occlusion and masks | — |

### `_g` and `_m` channels

What each channel means is read from the translated G-buffer writers ([Surface terms](#surface-terms)): for every material whose writer translates, the survey follows each surface term back to the channels it samples, then evaluates the recipe at several channel values to find the form.

**`_g`** (2,629 materials whose graph has the slot) holds F0 and gloss in one of two layouts. The material's `cb2` parameter `7107A2D8` picks the layout, the way `g_mode` does for [architecture glass](#architecture-glass-gbl_arch_glass). This holds in all 2,390 materials whose graph declares the key. Graphs without the key use the red and green layout.

| Layout | Materials | Red | Green | Blue | Alpha |
| --- | --- | --- | --- | --- | --- |
| `7107A2D8` < 0.5, or absent | 1,368 | gloss² (`gloss = sqrt(r)`) | F0, grey | — | — |
| `7107A2D8` ≥ 0.5 | 1,184 | F0 red | F0 green | F0 blue | gloss |

F0 is the sampled value, and the G-buffer stores its square root. In 796 materials of the first layout and 871 of the second, these are the whole terms. The others scale or remap them with material parameters, and their programs give the exact form. 20 materials (holograms) read no `_g` channel; the remaining 57 use graph-specific layouts, such as emission in blue.

**`_m`** (2,360 materials whose graph has the slot):

| Channel | Use |
| --- | --- |
| Red | Read by 1,443 materials, as a mask whose meaning depends on the graph: it gates emission (757 materials, scaled by the material's emission parameters), an albedo tint or detail layer (895), the detail normal (431), or gloss and F0 (248 and 172, such as the palette armour). One graph often uses it for several of these |
| Green | Read by 78 materials (damage overlays, emissive decals, floating hexes) |
| Blue | Occlusion, in 2,261 materials; the value itself in 1,833. 62 gold weapon materials also take F0 from it |
| Alpha | Unused |

52 materials, mostly holograms, read no `_m` channel.

### The labelled preview

The viewer draws a labelled **Rift Apart preview** (`rule_kind 'rift apart preview'`) in the Textured mode, and in the Shaded mode for materials whose program does not translate or while the native translation is off ([In the viewer](#in-the-viewer)).

**What the preview uses:**
- **Base colour:** from the material's `ABDAD780` slot.
- **Normal:** from the material's `7AE34E7A` slot.
- **Everything else:** neutral parameters.

**What it leaves out:**
- **`_g` and `_m` maps:** the preview does not read them. Their channels ([above](#_g-and-_m-channels)) reach the viewer through the translated programs, which apply each graph's own form.
- **Material graphs:** the preview translates no graph shader. Architecture glass keeps its hand decode (next section) as the fallback of the translated program. 1,474 graphs exist; 1,386 are bound by materials and 29 materials bind none.
- **Gloss, metal, emission and blending:** these are review values, not game data.
- **Fur shells:** the preview does not draw them; it takes a fur material's base colour and normal from its fur record. The translated fur program draws the shells ([Fur materials](#fur-materials)).

### Architecture glass (`gbl_arch_glass`)

The one decoded graph. 44 materials use it, among them the Nefarious trooper and swarmer domes. Every graph embeds its shaders (about 15 DXBC containers each, SM5 and DXIL, with named reflection), so the rule is traced to the graph's forward pixel shader (SM5 `ps_5_0`, disassembled with `D3DDisassemble`), not to a capture. The viewer draws these materials alpha-over through the forward glass program (`uKind 26`; GLSL comment in `nwo/viewer/shading.py`), with `rule_kind 'rift apart glass'`.

**Graph layout** (`graph_layout`). Three sections of the graph, then the material's own values (`material_parameters`):

| Section | Content |
| --- | --- |
| `0x1CAFE804` (graph) | Slot table, rows of (u32 default-texture string offset, u16 exposed flag, u16 unit, u32 slot key, u32 slot type) (`material.graph_slot_rows`). Unit `u` is texture register `t(5 + u)`. |
| `0x45C4F4C0` (graph) | Parameter records (u16 byte offset, u16 byte size, u32 key); the offset is the dword position in `cb2`. |
| `0xA59F667B` (graph) | The float defaults of `cb2`. |
| `0xF5260180` (material) | The same record layout from byte 40, its values in a block that starts at header word 2. |

`cb2` is the graph's defaults overwritten by the material's values at the graph's offsets.

**Texture binding** (`material.slot_texture`, traced in the engine's material loader). For each slot row:
- **Exposed row** (flag 1): the material's texture for the slot key when the material binds that key, else the graph's default path.
- **Hidden row** (flag 0): always the graph's default path, even when the material binds the key. In the install, 4,382 of 32,658 rows are hidden, and 26 material texture records name a hidden row (stale records, such as the orb pickup's colour).
- **Missing path:** when the path is empty or not installed, the engine's default texture for the slot type ([Missing textures](#missing-textures)).

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
- **Exposure:** `m_AdaptedLuminance` is the engine's adapted (eye-adaptation) luminance. It is a per-frame value: the name occurs only in the programs' reflection, and no file or engine table holds it. The preview uses the review value 0.18 (`RCRA_ADAPTED_LUMINANCE`, the same as the translated programs', see [Exposure](#exposure)), so a material with a large `emission` (the trooper and swarmer domes: 15) still draws brighter than a lit surface.
- **Blender:** the preview has no Blender rule; Blender exports use the translated program ([Blender materials](#blender-materials)).

**Translated program.** The automatic path (see [Material graph shaders](#material-graph-shaders)) reproduces these equations: on the trooper dome, with the real material values and textures, its alpha and emissive colour match them to float precision (`tests/test_rcra_shaders.py`). The translation also shows the terms the preview leaves out:
- the material and vertex alpha (`TEXCOORD.w`) and the per-draw alpha override (`MatDrawCB.m_AlphaOverrides`)
- the custom clip plane and the dissolve test on `TEXCOORD.z`, which set the alpha to 0
- a fog factor on the alpha, gated by render flag `0x4000000`, which the engine sets for additive materials ([Render flags](#render-flags))
- the unlit path: with render flag `0x400` clear, the lighting term is exactly 1
- the normal frame: the vertex tangent and its bitangent sign, not screen derivatives

## Material graph shaders

Every `.materialgraph` carries its compiled programs, and the engine's own passes are in `RiftApart.exe`. Both are read in place; nothing is copied out of the game folder.

| Module | Role |
| --- | --- |
| `nwo/shaders/dxbc.py` | DXBC container: chunks, RDEF reflection (constant buffers with struct members, bindings), signatures, and the SM4/SM5 token stream. DXIL containers are identified only. |
| `nwo/shaders/sm5.py` | SM5 program to a typed expression DAG, with a numeric evaluator. |
| `nwo/games/rcra/shaders.py` | Pass table, pass programs, the generic material binding and the neutral engine profile. |
| `nwo/games/rcra/lighting.py` | The engine's per-class deferred apply programs, found in the exe and translated into one direct-light recipe and one ambient-light recipe per class ([Lighting classes](#lighting-classes), [Ambient light](#ambient-light)). |
| `nwo/games/rcra/engine_tables.py` | The BRDF lookup the engine builds from tables in the exe ([Engine lookup tables](#engine-lookup-tables)). |
| `nwo/games/rcra/probe.py` | The review probe: the review environment prefiltered by the engine's probe convolution program ([Ambient light](#ambient-light)). |

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

### Techniques

Section `0xBC93FB5E` holds one 280-byte record per technique; the graph loader counts them as the section size / 280 (`0x1410d912b`). Each record holds 32 switch slots (`graph_switches`, [material layout](#layout-1)), then six u32. The third of those, at record `+0x108`, is the technique flags word. The loader copies it to the runtime technique (`+0x24`) and ORs it into the graph's word (`0x1410d951f`, `0x1410d9604`).

Bit `0x200` marks a forward technique:
- **Cost estimate:** the loader estimates the technique's cost from the size and resource flags of the forward pass (record 1) when the bit is set, and of the G-buffer pass (record 5) when it is clear (`0x1410d952a`).
- **Material bit:** `Material::PatchUpDefaults` sets material `+0x60` bit `0x20 << technique` for it (`0x1410d2317`).
- **Stub writer:** every graph with the bit (862) has a G-buffer writer that stores constant black albedo, and only 4 graphs without it do. For example, `basic_eye`'s slot-4 program writes `o1 = (0, 0, 0, 1)` and gloss 0.8.

Blended materials sit on most of them. An opaque material on one is drawn opaque in the forward layer: the engine's blend enum describes opaque as "GBuffer or Forward layer". Where the draw reads the material bit is not traced. `material.forward_technique` reads the bit.

For such an opaque material, `surface.surface` takes the albedo and an unlit colour from the forward program (`surface.forward_albedo`). The program is translated with every light source symbolic (`_forward_split`: the key light as `L` and `LIGHT`, unshadowed, the others `ENGINE.*`), and each colour channel is split exactly into F0 + Y_d × L_d + Y_s × L_s (`_factor_split`). The split goes through add, negate, a multiply or divide by a light-free value and a select on a light-free condition. It also has two exact rewrites:

- **A select on a light-dependent condition** (shadow cascades, the side of N·L) whose arms have one F0 becomes F0 + Σ Y × select(c, L_a, L_b), the arms' factors grouped by coefficient.
- **A product of two light-dependent values** ((key light × shadows) × (albedo × AO)) is multiplied out into its cross terms, (A0 + Σ Y_a L_a)(B0 + Σ Y_b L_b). This is tried only when a channel does not split without it. It keeps the result exact up to float rounding, but it can move part of a diffuse light into a factor that reads the BRDF lookup.

The light factors are then sorted by what they read:

- **Specular** (L_s): it reads a specular input (`SPECULAR_INPUTS`: the BRDF lookup, reflections, probes).
- **Diffuse** (L_d): it reads a light colour, the key light's `LIGHT` or another engine light (ambient fill, GI, the light grid). Factors with one coefficient are one diffuse light, their sum. The coefficients count as one when they are the same node, or equal at random inputs and texels. More than one coefficient takes the fallback below.
- **Key-light direction only:** a factor that reads `L` but no light colour (the planet surfaces' terminator) is not light. It joins F0, with `L` at a review direction (`REVIEW_KEY_DIRECTION`, game (0.303, 0.808, 0.505): from above and in front; `FORWARD_DIRECTION_NOTE`).

A material without the lit bit has no diffuse or specular light. Its program may still read a light source for an effect, such as the key light's colour or the default probe; those factors are 0, as the forward profile has every light source (`FORWARD_UNLIT_LIGHT_NOTE`).

From the split:

- **Albedo:** Y_d, the colour the program lights diffusely, replaces the stub's black albedo (`FORWARD_ALBEDO_NOTE`). The renderer lights it with its own diffuse light, as the scene behind a lit blended material is lit ([Forward profile](#forward-profile)). Without a diffuse factor the albedo stays black (`FORWARD_UNLIT_ONLY_NOTE`).
- **Unlit colour:** F0 reads no light source: an emissive part, or the whole colour of an unlit program. When it is not 0 at random inputs and texels (`_unlit_zero`), it becomes the terms `unlit.r/g/b` (`FORWARD_UNLIT_NOTE`), which a renderer adds after the lit terms in scene-linear units, negative values clamped to 0. F0 must not read the scene behind (`g_FrameBuffer`): the photo-mode filters do. The zero test draws UVs and the vertex colour (squared, never negative) from [0, 1] and the other inputs from [−1, 1], over 24 trials, so a colour masked by a texel or a UV range is not taken for 0.
- **Black:** with no diffuse, specular or unlit part, the forward colour is 0 and the material is drawn black (`FORWARD_BLACK_NOTE`). For example, the discoball test material's colour multiplies by its own zero constants.
- **Kept from the stub:** the normal, F0, gloss and occlusion.

**Different diffuse coefficients** (`_ManyDiffuse`, `FORWARD_OWN_LIGHT_NOTE`): one albedo cannot carry them, so the albedo stays black. The unlit colour is the program's colour with every light source 0 (the forward profile), and `forward_specular` returns the whole key-light term (`_own_light`: the colour with the `ENGINE.*` inputs 0, minus that with `LIGHT` also 0). The viewer adds that term once per review light; the ambient light, local lights and reflections are not drawn. Blender has no per-light program, so it draws the unlit colour alone.

When the split fails otherwise, the material stays black and its limitation gives the reason (`FORWARD_TECHNIQUE_NOTE`).

For a lit material the viewer also draws the program's own specular term (`surface.forward_specular`, `FORWARD_SPECULAR_NOTE`; the whole key-light term in the fallback, `FORWARD_OWN_SPECULAR_NOTE`), as it draws a blended material's [own key light](#own-key-light). The term is the sum of the specular factors' Y_s × L_s, with every `ENGINE.*` input set to 0 as the forward profile has them (`_zero_inputs`), so only the key light remains: its highlight, without reflections. Setting them to 0 rebuilds the term with constant folding, and a product with 0 folds to 0 only when the other factor has known bounds, so the result stays exact (a reflection weight times a zero reflection). The term must be linear in `LIGHT` and 0 at `LIGHT` = 0, and the viewer adds it once per review light to the lit G-buffer surface. Blender has no per-light program, so there the stub's F0 gives the specular.

On Ratchet's eye (`basic_eye`), Y_d is the eye colour texture at a view-dependent coordinate (it reads the view offset, the tangent frame and the eye displacement texture), F0 evaluates to 0, and Y_s is 1. Its specular term is the cornea's highlight, a sharp lobe (about 24 × `LIGHT` with the light, the view and the normal aligned).

### Engine vertex program

The engine vertex program that feeds slots 1, 4 and 5 is in the exe (vertex pulling from `g_MainVBPosView`, `g_MainVBNrmTanView`, `g_MainVBUV0View` and others). Its translation gives the interpolants:

| Interpolant | Content |
| --- | --- |
| `TEXCOORD.xy` | UV0 (int16 × the subset's UV scale / 16384) |
| `TEXCOORD.zw` | `m_AlphaOverrides`: its high half or the material's `m_AlphaTest`, and the draw alpha ([Material alpha](#material-alpha)) |
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
- **Loops:** a loop is unrolled while its exit condition folds. A data-dependent loop (a light list, for example) poisons only the registers it writes, with an `unknown` node, so outputs that do not depend on it still translate. Branch, break, return and discard conditions act only on active lanes, so they are folded under what a lane's active predicate implies. A counter is then exactly the iteration count, and a loop with an early break still ends at its count. Facts carry over iterations: a lane active now passed the previous iteration's exit test, so its decided parts were false. An exit test that one select condition decides is split on it. A parallax counter that jumps past its step count on a hit (n = trunc(32 − 24 sat(x)), so 8 to 32 steps) then reads "hit, or step ≥ n", and the loop ends at n's bound. Value ranges (`Graph.bounds`) decide comparisons, and an integer test of `ftoi(x)` becomes the exact float test of x. Extra predicated iterations are allowed only up to the bound, at most 64 steps plus the exit test. When the folded exit test no longer shows the counter (16 to 64 portal steps: the test is decided false while the step stays below 16), the budget comes from the unfolded test, from the counter's low bound. Ray marches and parallax decals end this way. When no test shows it (the eyes' 32-step reflection march: its test `k < 32` folds true at every step, and the hit test is ANDed with it as an integer mask), the budget comes from the loop body (`_counted_steps`): a register compared with a literal whose only writes in the body add a literal to it, from a constant at entry. A budget never changes the result: past it the loop is poisoned as before. The outputs are bit-identical to the unfolded translation wherever both translate. Past the budget, one more predicated iteration is still run and kept when no lane can continue after it (`_final_iteration`): the remaining continuation predicate is false for every truth value of the comparisons it combines (`sm5.never`, at most 12 of them). A light or probe list loops over the set bits of a constant word and can break early on an accumulated weight; its large body allows no predicated iterations, so its second iteration used to poison it, although the word's last bit was already cleared.
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

These counts take the neutral profile alone: no material constants or textures and no engine resources. What stays unknown in them is engine data that profile leaves unbound: per-object and per-material records, the scene G-buffer and depth, light-grid and probe buffers. The bindings the browser uses resolve them:
- **G-buffer writers:** 5,712 of 5,721 materials translate ([Surface terms](#surface-terms)). The other nine have no graph in the install.
- **Forward programs:** all 1,747 blended materials ([Forward profile](#forward-profile)). The forward programs' own key-light term translates for 678 of the 792 lit ones ([Own key light](#own-key-light)).

With the vertex bindings of [Vertex programs](#vertex-programs), which bind the vertex streams and leave the force-field bitset empty, all five vertex slots translate for the 207 graphs that materials use (583 of 583 uses); the 208th, `gbl_displacement_singularity`, has no material.

**Other program features.** The translator also takes:
- **Tessellation:** a domain program's control points (`vicp`, inputs `CP<n>.<semantic>.<c>` from the input signature), patch constants (`vpc`, `PATCH.<semantic>.<c>` from the `PCSG` signature) and `vDomain` are inputs. Components are named by register component, as interpolants are. The six domain programs in the graphs (`amb_uwb_jellyfish` and the two `hero_thin_thrust_exhaust` graphs, slots 2 and 6) translate fully.
- **UAV writes:** `store_raw`, `store_structured` and the UAV atomics are kept in `Translation.stores` with their operation, address and predicate. The value an `imm_atomic_*` returns is memory that other threads write, so it is `unknown`.
- **Groupshared memory** (per thread): a `ld_raw` or `ld_structured` of `g#` at the same address node and byte offset as this thread's last store returns that value. Any other read is a value another thread wrote: `unknown`. An atomic on an address, or a store at a run-time offset, drops this thread's values there. No graph program uses groupshared memory or atomics; engine compute programs do.

Not translated: the DXIL-only raytracing libraries (`IGRS` entries, 395 graphs). They need an LLVM bitcode reader, which `dxbc` does not have.

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
| GGX | `α = r⁴`, r = `0.9725 − 0.7514 gloss` (the program squares r three times: r², r⁴, r⁸ = α²) |
| Visibility | Smith joint approximation, `0.5 / (NL (NV (1 − α) + α) + NV (NL (1 − α) + α))` |
| Fresnel | Schlick, F90 = 1 |
| Diffuse | Lambert `sat(N·L)`, no 1/π, times the light colour |

At 300 random inputs the translated program matches this form within 0.9 % (the 12-bit normal). The form with α = r² differs by up to 370 %.

### Lighting classes

The writer stores a lighting class in bits 13 to 15 of NormalSpec x; bit 12 set means class 0 whatever those bits hold. Each class has its own apply program in the exe. `lighting.find_programs` finds them by the class test at their start, `ubfe(3, 13, NormalSpec.x)` against the class:
- Of the two programs that test classes 1 and 2 together, the one that binds `g_SkinProfiles` applies class 1 and the other class 2.
- Per class it takes the smallest program without the rain inputs (`g_WeatherPorosityMap`).

The scan parses every DXBC container of the exe (about 2.5 s). `lighting.programs` keeps the programs' offsets and sizes in the source's translation disk cache, keyed by the exe's path, size and modification time and the scanning code (`lighting.py`, `dxbc.py`), and reads their bytes back from the exe in later sessions.

Classes 1 to 5 read a fifth target, `g_GBufExtra`. It is the writer's `SV_Target3`, one u32 whose fields the apply program reads at fixed bit offsets (`lighting.EXTRA_FIELDS`):

| Class | Writers (graphs) | Apply program | `g_GBufExtra` fields (bit offset, width) |
| --- | --- | --- | --- |
| 0 | the rest (862 with bit 12 clear, 556 with it set) | `0x386225C` | none |
| 1 subsurface | 15: skin and heads (`basic_normal_gloss_subsurface`, `basic_subsurface_head`), nymphs, dinosaurs | `0x383BE0C`, reads `g_SkinProfiles`; diffuse and specular to separate targets | (0, 3) scatter, (9, 6) transmission, (18, 7) and (25, 7) a second octahedral normal, (5, 4) the skin profile |
| 2 cloth | 11 (`basic_normal_gloss_cloth`, `cloth_distressed`, hex platforms) | `0x37C962C` | (0, 8) and (8, 8) a second normal, (16, 5) sheen mix, (21, 6) scatter lobe, (27, 5) wrap |
| 3 hair | the engine fur writer ([Fur materials](#fur-materials)) | `0x381EE4C` | (0, 9), (9, 9) and (18, 1) the strand tangent, (19, 7) transmittance, (26, 6) lobe shift |
| 4 anisotropic | 7 (`basic_normal_gloss_aniso*`, `hero_armor_palette_aniso`) | `0x37E5A7C` | (0, 9), (9, 9) and (18, 1) the tangent, (19, 6) its rotation, (25, 7) a second gloss |
| 5 foliage | 20 (`basic_normal_gloss_foliage`, canopies, shrubs, topiary, transmittance) | `0x380292C` | (0, 6), (6, 6) and (12, 6) the transmission colour, (18, 7) transmittance, (25, 7) wrap |

The class programs, as their translations read:
- **Cloth:** sheen from a Charlie distribution (`(2 + 1/α) sin^(1/α)θ`) with the Neubelt visibility `1 / (4 (NL + NV − NL·NV))`, coloured by F0. It is mixed with GGX by the sheen field, both on the second normal, with r = `0.9725 − 0.6011 gloss`. The diffuse wraps by the wrap field.
- **Anisotropic:** Lambert, plus the average of two anisotropic GGX lobes, `D = 1 / (αt αb (h·n² + h·t² / αt² + h·b² / αb²)²)`. Lobe 1 sits on the G-buffer tangent t (decoded like the writers' octahedral normal) with b = normalize(n × t); lobe 2 on t turned about n by the rotation field × 0.04987 rad. Both have αt = r(gloss)⁴ along the tangent and αb = r(second gloss / 127)⁴ across it. The visibility is separable, `0.25 / ((v·h (1 − k²) + k²)(n·l (1 − k²) + k²))` with k = (0.75 min(αt, αb) + 0.25 max(αt, αb) + 1) / √8, and the Fresnel is Schlick on v·h (`lighting.anisotropic_lobes`; `test_class_4_is_two_anisotropic_ggx_lobes` matches the translated program within 1 %).
- **Subsurface:** two GGX lobes, mixed by view depth. Transmission through the shadow map's depth, plus a Henyey–Greenstein scatter lobe (g = 0.65) scaled by the scatter field. The transmission is Burley's normalised diffusion, `0.125 / d (e^(−t/d) + e^(−t/3d))`, with t the thickness from the shadow map and d the skin profile colour per channel (see [Skin profiles](#skin-profiles)). It is gated by `0 < sat(0.2 − 0.8 N·L) × m_KeyLightShellLodConsts.y`, a shadow cascade constant. The skin pass (`0x33CC4FC`) blurs the diffuse light in screen space with the skin profiles, then adds the specular.
- **Foliage:** wrapped diffuse plus back transmission, `(1 − sat(N·L + 0.9)²) sat((w − N·L) / (1 + w)²)`, scaled by the transmittance field.
- **Hair:** lobes shifted along the strand tangent, and transmission.

**One light's term** (`lighting.direct`). Each class program is translated with these bindings:
- **G-buffer:** packed from term inputs exactly as the writers pack it. NormalSpec gets the 12-bit normal, F0 square roots and gloss as bytes, the class bits and occlusion mode. DiffuseEmis gets albedo and occlusion, and `g_GBufExtra` the `extra.<k>` fields.
- **The light:** one key light, direction `L` and colour `LIGHT`, unshadowed. The shadow atlas, cloud shadows, ray-traced shadows and gobo read 1, the point lies past every shadow shell, and the sunlight occlusion factor is 0.
- **The view:** the camera-to-point offset `VIEW` and the view depth `DEPTH`.
- **Everything else:** 0. That covers ambient fill, the light grid, probes, GI, SSR and the local light lists (empty).
- **Skin profiles:** the class-1 load of texel 64 + the profile field reads the inputs `PROFILE.rgba`. `lighting.translate` checks the address at random field values. Under these bindings the profile reaches the light only through the shadow-map transmission, which is off, so the recipe folds it to 0 (`profile_reach`, `SKIN_NOTE`).

The packed fields read back exactly through `surface._integers_in_float` (`PACKED_NOTE`), as for the writers. Each output is split as `albedo × diffuse + specular`: the apply program multiplies its diffuse light by DiffuseEmis rgb, and for class 1 the skin pass does. The recipe holds the light's own term, the output minus its value with `LIGHT` = 0. That removes the hair program's light-independent ambient transmission, which comes from inputs bound to 1. Against the program translated without the integer-to-float rewrite, the recipes agree within 1.1 × 10⁻⁷ relative at 300 random inputs per class. The viewer's GLSL agrees with the recipes within 4 × 10⁻⁴ relative on the GPU.

`surface.surface` returns the class as the constant term `class` and the fields as the terms `extra.<k>`: `ubfe` of the writer's `SV_Target3`, read back by the packed-field rewrite. At random inputs they equal the fields of the writer's word, on cloth, foliage, subsurface and anisotropic materials. A word that is a sum of fields, as the foliage writers pack it, is read as their or, under the same packing assumption. A writer that picks its class at run time (a select of packed words) is lit as class 0 and says so.

#### Skin profiles

`g_SkinProfiles` is `required/textures/char-skin-profiles-srgb_1d.texture`: `R8G8B8A8_UNORM_SRGB`, 128 × 1. The engine's getter (`0x141353a90`) returns that texture once it has loaded, else its "Default Skin Profile Lookup", which `TextureDefaultsInit` builds from the same 128 texels embedded in the exe. Texels 0 to 7 are the eight skin tones of `char-skin-tone-srgb_1d.texture`. Texels 64 to 67 and 79 are the profile colours (64 is (80, 36, 29), a red scatter distance), and the rest are black. The class-1 apply program and the skin pass both load texel 64 + bits 5 to 8 of `g_GBufExtra`.

#### Engine lookup tables

`TextureDefaultsInit` (`0x141353e10`) also builds the BRDF lookup, "Default Brdf Lookup". No texture asset replaces it: the only other reference to its texture is at shutdown. It is a 64 × 64 `R16G16B16A16_FLOAT` array of 4 slices. `engine_tables.brdf_lookup` reads it through the instructions that load it: it decodes their `lea` targets at the traced addresses, so another build of the exe raises instead of returning other bytes.

| Slice | Content |
| --- | --- |
| 0 | Red and green from 4,096 half-float pairs at `0x1444c6cc0` (blue and alpha 0): a split-sum style table, red falling and green rising along the roughness axis |
| 1, 2 | 64 × 64 × 4 half floats copied from `0x14548e340` and `0x145496340` |
| 3 | 0 |

The probes the programs read are prefiltered on the GPU by `EnvProbeMipGen` (render task `0x14108d140`). Per probe face it dispatches the convolution program (exe offset 63366492): GGX importance sampling with filtered source mips. The CPU fills one 64-byte `g_MiscArray` record per output mip (`0x14108cd60`). The mips are 128, 64, 32, 16 and 8 texels. Each record holds the probe and face, 1 / size and half of it, and four per-mip tables from the exe:

| Mip | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| `0x1432d1f50` | 0.000362 | 0.00548 | 0.0416 | 0.209 | 0.8 |
| `0x1432d1f68` | 0.12 | 0.12 | 0.08 | 0.04 | 0 |
| `0x1452f0b60` (c; the record also holds 4 / (1 − c)) | 0.982 | 0.914 | 0.487 | 0.00316 | 0 |
| `0x1432d1ff0` (n; the record holds 2ⁿ and n) | 6 | 6 | 7 | 8 | 8 |

These addresses are for the Steam build the browser was checked against (`probe.RECORD_TABLES`). Every apply program reads slice 0 of the BRDF lookup, through `g_LinearClampSampler`.

#### Ambient light

`lighting.ambient` translates each class's apply program with the key light off, so its output is the ambient light alone (albedo included; class 1 as albedo × diffuse + specular, as the skin pass adds them). The bindings stand for a scene lit by its environment only:

- **Default probe** (`g_EnvProbeDefault`, a cube read with `sample_l`): kept as a sample node, with its direction and lod. The renderer looks up the review probe.
- **BRDF lookup** (`g_BrdfLookup`): kept as a sample node, slice 0.
- **Light grid:** the grid is centred on the camera and reaches 512 units each way. It blends each point between its radiance and the default probe by a probe weight: the corner cells' max(cell data.w & 0x3C0, lookup word & 0x3FC), averaged by the trilinear and facing weights, over 960 (beyond its extent the weight is 960). The same weight scales the probe's reflection by the grid's irradiance over the probe's. Every lookup word reads `0x3C0` (weight 960) and every cell is empty (`GRID_PROBE_WORD`). So the program lights each point from the default probe alone, the grid's radiance cancels exactly, and the reflection scale is the probe's own: its lod-5 radiance at the normal over that at the reflection. An empty grid with lookup words of 0 would give every point the grid's zero radiance and dim every reflection.
- **Everything else:** local probes (empty lists), screen-space reflections, distant GI, ambient fill and light volumes read 0. `m_LightGridIntensity` is 1, a review value.
- **Key light:** off. The result must not change with its direction (`_light_free`, at random inputs); the direction is then folded to 0.

All six classes translate. Each recipe reads the G-buffer terms, the view, three default-probe reads (two at lod 5, one at a lod from the gloss) and, except class 4, one BRDF lookup read. Under a constant probe P, a white surface (F0 0.04) returns 1.0 to 1.06 P for classes 0, 1, 2, 4 and 5, and a white metal returns P at high gloss (`test_ambient_furnace`; hair has its own transmission model). The apply programs also apply the G-buffer occlusion to the probe's reflection (a specular occlusion term), where the rig occludes only diffuse ambient light.

**Review probe** (`probe.probe_levels`, `probe.cached_probe_levels`): the review HDRI without its sun (`environment.review_residual`, the image the rig's SH and bands come from) becomes a 256 × 256 cube in the environment's world frame. Its source mips are a 2 × 2 box chain (`SOURCE_MIPS_NOTE`). The convolution program is translated once per output mip, with that mip's record bound, so its 2ⁿ-sample loop has a constant count (the translator's unroll cap is raised to it for that translation only). It is then evaluated over the cube's texels with trilinear D3D cube lookups of the source chain. A constant source comes back unchanged at every mip (`test_probe_convolution_keeps_a_constant_environment`), and each level keeps the environment's mean radiance. The probe takes about 12 seconds once per HDRI and install, and is cached under `.cache/environment`.


### Forward profile

Blended materials are drawn by their forward program (slot 1), which lights itself. For export, `nwo/games/rcra/shaders.py` defines one labelled set of engine inputs. Its rule is that every light source is zero, and the renderer lights the material's G-buffer surface instead:
- **Lights:** key light, light grid and lists, shadow atlas, ray-traced shadows, gobos, probes, GI, SSR, AO and the BRDF lookup read 0 (`FORWARD_LIGHTING`).
- **Draw flags:** only the material's lit bit (`material.draw_flags`).
- **Draw alpha:** `TEXCOORD.w` is the material alpha as the plain instance draw packs it ([Material alpha](#material-alpha)); `TEXCOORD.z` is 0.
- **Object:** the neutral object of [Per-object values](#per-object-values).
- **Depth:** the stored scene depth sits at the far clip, so there is no soft depth fade.
- **Scene G-buffer:** the scene the far-clip depth stands for is empty, so reads of `g_GBufNormalSpec` (screen effects, eye normal smoothing, decals reading around the pixel) return a cleared target, every word 0, at the review viewport size (`SCENE_GBUFFER`, `SCENE_SIZE`). G-buffer writers that read these engine textures, or the default probe in their emission, get the same values.
- **Own screen-space reflections:** programs that trace their own reflection (the eyes march through the half-resolution depth, then reproject the previous frame's reflection by the velocity buffer) see the same empty scene: `g_HalfResLinearDepth` at the far clip, `g_SSRHistory` 0 (a light source) and `g_SSRVelocityTex` 0, no motion (`SCREEN_REFLECTION`).
- **Clip plane:** the custom clip plane clips nothing.
- **View:** the programs rebuild the view position from `SV_POSITION` and `m_ViewToWorldMat`. With `FORWARD_VIEW_FLOATS`, `SV_POSITION.xyw` is the camera-to-point offset in world axes, so a renderer supplies it as `VIEW.xyz` and the matrix translation as the camera position `CAMERA.xyz`.

`surface.forward` returns the target under this profile: colour (HDR), alpha and the discard. On the trooper dome it matches the hand decode of `gbl_arch_glass` with lighting zero: alpha within 4 × 10⁻⁷, colour within 7 × 10⁻⁷ relative. Two translator folds make the profile's dead code disappear: a comparison `x < 0` of a value that is never negative is false, and a shifted zero is zero.

**The scene behind.** A program that samples the scene colour behind it (`g_FrameBuffer`: heat haze, shockwaves, water splats) binds the sample as symbolic `BACKGROUND.rgb`. `surface._affine_split` keeps it only when each colour channel is affine in its own background channel, colour = A + B × background (through add, negate, a multiply or divide by a free value, or a select on a free condition), and the alpha and discard do not read it. It then returns A as the colour and B as the `background` terms. Over the 45 materials it splits, A + B × background matches the unsplit program within 1.4 × 10⁻⁷ relative at random inputs. The sample's screen-space offset (the distortion) is dropped and labelled.

A lit material's background coefficient could carry the forward lighting the profile zeroes. `surface._lit_background` translates such a program again with every value the profile sets to 0 left symbolic (`_EngineFreeBindings`: the unbound engine constants, such as the key light, ambient fill, GI and cloud shadows, and samples of the lighting textures; the light-list counts stay 0 and the view matrix keeps the forward view binding). Then:

- **B reads none of them:** the scene passes through unlit whatever the lights, and the profile's B is the game's: the shockwaves, ground waves and portal energy (10 materials, whose split matches the unsplit program within 8 × 10⁻⁸ relative).
- **B = B0 + Y × L:** the program lights the scene behind as albedo. L is one engine factor per channel that also lights the program's own colour and reads no specular input (`SPECULAR_INPUTS`): the forward diffuse light, key light × shadows (cascades, cloud shadows, gobo) × N·L plus (1 − the BRDF lookup's diffuse term) × the ambient fill or light grid, distant GI and AO. In `gbl_heat_distortion_01`, B is 0 with every light zero and 1 with only the ambient fill at 1. `surface._factor_split` finds L; B0 and Y are free of every symbolic value and become the terms `background` and `lit_background`, and a renderer supplies its own diffuse light for L (26 materials: heat distortion, goo and water splat decals, ripples, bubbles, liquids). Over them, A + (B0 + Y × L) × background matches the program within float rounding (3.8 × 10⁻⁵ relative at random engine values).
- **B also has specular factors:** the three goo decals (`fx_goo_decal_simple_alt`) also pass the background through a specular factor (the BRDF lookup's scale term and the gloss reflection constants). The forward profile's specular light is 0, so those factors are left out (`LIT_BACKGROUND_SPECULAR_NOTE`), and the one diffuse factor gives Y as above. Their diffuse factor does not light the decal's own colour A, so it qualifies by reading a diffuse light colour (`DIFFUSE_LIGHT_INPUTS`: the key light's colour or the ambient fill).
- **Otherwise** it stays unsupported: more than one diffuse factor in B.

**Scene mode.** A program that reads the scene behind other than as an affine tint keeps it: the screen blend curves of the holograms and camera panes (`bg < 0.5 ? 2ab : …`), a vignette's contrast power, the absolute value or saturate of a distorted scene, and particles that read one background channel into all three. `surface.forward` then returns the whole target with `BACKGROUND.rgba` as inputs and a background tint of 0 (`reads_scene`), labelled (`SCENE_NOTE`). A lit one qualifies only when its target reads the scene and the light sources in separate summands (`surface._separable` over the symbolic translation), so the scene passes through unlit. The symbolic values are the light sources only: the unbound `GlobalWorldCBuffer` constants and the lighting textures; view and draw settings (clip plane, depth of field, temporal jitter, alpha overrides) keep their profile values. The viewer samples the inputs from a copy of its frame taken before the batch; Blender has no node for the scene behind, so these materials keep their G-buffer surface there.

**Texture kinds.** Material units are 2D arrays, sampled at a slice. 3D textures (flipbook volumes) are sampled trilinearly with a normalised depth ([Textures](#textures)). Cube maps (`TextureCube`, and a `TextureCubeArray` read at a constant cube index) are sampled along the direction the program computes (the first three address lanes, in game space). The explicit mip level, bias or gradients of a cube lookup are not applied: the top mip is read, and the result says so. An array read at a run-time index, or a cube read by `gather` or `ld`, is unsupported. A cube texture's slices are its faces, six per cube in the order +X, -X, +Y, -Y, +Z, -Z, top row first. A constant array index that is NaN (a per-object seed of 0 divided by 0) reads slice 0, as D3D converts NaN to the integer 0 (`blender_export._slice_index`), and one past the array reads its last slice, as D3D clamps an array index (`slice_index`). A texture the install does not hold gets the engine's default ([Missing textures](#missing-textures)).

#### Own key light

`surface.forward_lighting` translates a lit forward program again with the key light symbolic: direction `L.xyz` (to the light, game space) and colour `LIGHT.rgb`, unshadowed as `lighting.direct` binds it (past every shadow shell; the shadow atlas, ray-traced and cloud shadows read 1). Every other light source keeps the profile's 0. With `LIGHT` = 0 the colour is the profile's. It is kept only when it is linear in `LIGHT`, checked at random inputs and texels as colour(2C) − colour(0) = 2 (colour(C) − colour(0)). A renderer then adds colour(L, C) − colour per light.

Over the 1,736 blended and additive materials with a graph:

| Result | Materials |
| --- | --- |
| Lit, own key light translates | 678 |
| Unlit (no lit bit: the program reads no light) | 944 |
| Lit, but the colour reads no key light (local lights or ambient only) | 72 |
| Lit, and the lit colour reads the scene behind | 33 |
| Not linear in the light colour | 9 |

The viewer draws the 678 with their own term per review light ([In the viewer](#in-the-viewer)). Blender has no per-light hook in a material, so it keeps lighting their G-buffer surface.

#### Own ambient light

`surface.forward_ambient` translates a lit forward program with the environment as its only light, for the viewer's review probe ([Ambient light](#ambient-light)):

- **Probe lists:** the lists hold one review probe (`REVIEW_PROBE`), selected by a lookup word of 1 and z-bin words with every bit set. The probe has an identity transform, and its extents and falloffs are 0, so its weight is 1 everywhere. Its proxy planes sit 10⁶ units away each way (the negative ones stored negated), so the parallax correction leaves the reflection direction as it is. Its cube index is 0. The program's list loop translates through its final iteration ([Translation](#translation)).
- **Default probe and BRDF lookup:** kept as sample nodes, as for the apply programs. With a probe in the lists, the program also reads the default probe along the reflection at the gloss lod, for the weight the list leaves.
- **Everything else:** as the apply programs' ambient (grid cells hand their points to the probe, `m_LightGridIntensity` 1), and the key light off.

The term is the colour minus the colour with every probe read 0, and it must double when the probe's radiance doubles (`_check_linear_in_probe`). On the trooper dome the probe array's sharp read looks along exactly the default probe's reflection direction (`test_forward_ambient_reads_one_review_probe`), and every BRDF read that reaches the colour is slice 0. Of the 678 materials with their own key-light term, 671 translate (every BRDF read slice 0); 4 are not linear in the probe (the planet atmosphere and edge, a screen, a lava splash), and 3 have no probe list (bubbles, debris stuffing). Those keep the surface's ambient light.

#### Exposure

The game's tone map exposes the scene by `m_AcesMiddleGray / L`, L the adapted luminance. The eye-adaptation program (`g_LumAccumBuffer` to `g_RWAdaptedLumBuffer`) stores L and 1 / L. The tone-map program multiplies 1 / L by `ToneMapCBuffer.m_AcesMiddleGray`, and its basic variant (`g_TmBasicToneMapLookup`) by the literal 0.18. Then it reads its lookup at log₂(x) / 16 + 0.65625.

Materials that scale their emission by `m_AdaptedLuminance` therefore keep their on-screen brightness whatever the exposure. The planet surfaces (graph `gbl_uniq_planet_01`, eight materials such as `sav_uniq_planet_surface_01`) go further: they encode their colour to sRGB, map it through a polynomial fit of the inverse tone curve and `exp2`, then multiply by L / 0.18. The game shows that colour at its authored display value. Their sunlit limb is multiplied by up to 0.5 + 19.5 × max(0, 2 N·L − 1) × (1 − N·V)⁴, so it reaches the top of the tone curve and blooms.

The review render exposes by 1 and has no automatic exposure, which is the game's exposure at L = 0.18. That is the review value of `m_AdaptedLuminance` (`surface.REVIEW_ADAPTED_LUMINANCE`, and the viewer's `RCRA_ADAPTED_LUMINANCE`). An emission scaled by L then draws at the brightness the game gives it relative to the display. The model viewer shows it through the game's own lookup by default ([Display transform](#display-transform)); under the viewer's ACES fit, an inverse-tone-mapped colour draws lighter in the mid-tones than in the game (display 0.3 draws about 0.49). Two more luminances in the same units are bound at 0.18 too. The frame's world-buffer fill (`0x141206700`) stores the measured luminance as `m_ActualLuminance`, clamped to [10⁻⁴, 100] as the adapted one is; within the adaptation range the adapted luminance follows it. It stores `m_ToneMapAnchor` as 2^(EV − 13) from the tone-map settings (`0x1411dcf00`). With the tone map off, that EV is 10.526, so the anchor is 0.18. About 60 graphs scale their emission by one of these two instead of L: the Nefarious sniper's bulbs and laser, Clank's dimensional and holo eye glow, the nymph enemies, the galactic map lines and HUD effects.

#### Display transform

The game's lookup is built by `CS_ToneMapCreateLookup` in `RiftApart.exe`: a 32³ table over linear Rec.709 at 2^(t × 16/31 − 10.5), t = 0 to 31 per axis. Each entry goes through ACES 1.0: Rec.709 to XYZ to AP0 without a D65 to D60 adaptation, the glow and red-modifier terms, AP1, RRT saturation 0.96, the C5 segmented spline, the 48-nit C9 output spline, dim surround 0.9811, output saturation 0.93, D60 to D65 only when `m_A.x` is set, Rec.709 and the sRGB encode. For SDR output the function that dispatches it (`0x1411db740`) writes `m_A.x` = 0 into its `MiscCBuffer`, so the game skips the D60 to D65 step: mid grey stays neutral and white is the ACES D60 white, slightly warm. The tone map multiplies the scene by `m_AcesMiddleGray / L` ([Exposure](#exposure)) and reads that table trilinearly at log₂(x) / 16 + 0.65625. Its colour-correction variants (`CS_ToneMapCreateLookupCC1` to `CC4`) mix in one to four LUTs on the encoded value: each LUT sampled trilinearly at encoded × 31/32 + 1/64, times its weight from `m_B`, plus the uncorrected value times sat(1 − the weights' sum) ([Atmosphere lighting](#atmosphere-lighting)).

`nwo/games/rcra/display.py` is the port. Translated by `nwo.shaders.sm5` and evaluated at random thread ids, the exe's program matches it to about 1e-5. The model viewer's **Game (Rift Apart)** tonemap (`uTonemap` 3, `riftApartDisplay` in `shading.LIGHTING`) samples the same table from the shared display texture, and on a Ratchet frame under the Shared HDRI it agrees with the CPU port to within 8-bit rounding. It is the default display for Rift Apart in the model browser, the clip viewer and the gallery thumbnails (`Game.display_tonemap`). The exposure is the viewer's (the lighting's, 1 for the rig), except under the Game atmosphere lighting, which meters the frame as the game does ([Atmosphere lighting](#atmosphere-lighting)).

Not part of it: lens flares, vignette and film grain; the colour-correction LUT only under the Game atmosphere lighting. Bloom is drawn ([Bloom](#bloom)).

#### Bloom

The bloom programs are SM5 pixel programs in `RiftApart.exe`. The exe's program names are sorted alphabetically and the programs sit in the same order, which `PS_BloomDownsampleLatLongThresholded` and `PS_ToneMapApplyBasic` confirm by their bindings. The screen chain uses `PS_BloomDownsampleThresholded`, `PS_BloomDownsample` and `PS_BloomUpsample`. The pass function (`0x1411da720`) fills their constants from the atmosphere's `BloomSettings`: Intensity, Threshold and GaussianStrength0 to 4, at offsets 656 to 680 of the built block, in schema order after the tone-map fields. It runs over `kBloom0` to `kBloom6`, at 1/2 to 1/128 of the screen (the render-target table's size shifts −1 to −7):

1. B1 = down13(B0) × sat(luma × (1/L) / max(Threshold, 1/16) − 1) / (G0 + … + G4), with L the adapted luminance. down13 is a 13-tap filter at ±1 and ±2 source texels; the gate reads the luma of the filtered colour.
2. B2 to B6: down13 of the level above, without the gate.
3. Back up, each pass draws a 3 × 3 tent at ±1.5 source texels with blend state 6 (colour = src + srcAlpha × dst): B5 = G4 tent(B6) + G3 B5, B4 = tent(B5) + G2 B4, B3 = tent(B4) + G1 B3, B2 = tent(B3) + G0 B2, B1 = tent(B2), and B0 = Intensity × tent(B1), which overwrites. The five strengths weight the 1/8 to 1/128 levels.
4. `CS_ToneMap` adds B0 to the scene radiance when `m_DoBloom` is set, before the exposure and the lookup.

`nwo/games/rcra/bloom.py` holds the pass list, the GLSL and a numpy reference. The model viewer runs it on every frame under **Game (Rift Apart)**, over its float scene target. On a Ratchet frame with a raised intensity and lowered threshold (bloom changing pixels by up to 0.84), and on Clank with the default settings (up to 0.23), the viewer agrees with the reference to within 1/255. Viewer choices, not game data:

- **Settings:** `required/atmospheres/default_atmosphere.atmosphere` (intensity 0.1, threshold 3, strengths 0.4, 0.5, 0.6, 0.8 and 1), since a model has no atmosphere. Most of the 95 atmospheres use intensity 0.1 to 0.3 with threshold 3; `atm_default_day` uses 0.
- **B0:** a 2 × 2 box copy of the scene at half resolution. How the game fills `kBloom0` is not traced.
- **Gate:** 1/L is the viewer exposure / 0.18, since the display exposure is `m_AcesMiddleGray / L`.
- **Sizes:** levels halve with a floor, and the programs' uv scale for odd sizes (2 × destination / source) is taken as 1.
- **Not drawn:** lens dirt (`BloomDirtiness`, 0 in the default atmosphere), lens flares and the LatLong variants.

#### Atmosphere lighting

A level's lighting lives in its `.atmosphere` assets (95 in the install; Corson V is `atm_malinon_*`, Megalopolis, and `atm_neo_nefarious_*`, Nefarious City). The asset is 36 bytes of toc header and a DAT1 whose section `0x02F06D4E` is the built parameter block (loaded at `0x140fae710`, `kBuiltAtmosphereBaseVersion` 229). Its fields follow the order of the exe's reflected type records (`AtmosphereDef`, `EnvLightingDef`, `ToneMapSettingsDef`); references are 24 bytes (a u64 asset id, 12 zero bytes, the tag `0x95A3A227`). `nwo/games/rcra/atmosphere.py` reads:

| Offset | Field | Megalopolis day |
| --- | --- | --- |
| 64 (reference) | SkyBoxCubeMap | `malinon_skybox_day_006` |
| 88 (reference) | FillLightCubeMap, the default env probe | `malinon_intro/envprobes_day/0xe54a425db9195aba` |
| 112 to 120 | KeyLightColor | (1, 1, 0.831) |
| 124 | KeyLightIntensity | 6 |
| 128, 132 | KeyLightAzimuth, KeyLightElevation (degrees) | 220, 45 |
| 636, 640, 644 | AdaptationAnchor, AdaptationRange, AdaptationBias (EV) | 12, 10, 0 |
| 768 (reference) | ColorCorrection SourcePath | `lut_megalopolis_warm9` |

The model browser's **Game atmosphere** lighting, lights a model with Corson V's Megalopolis by day (`atm_malinon_day`, `atmosphere.DEFAULT`); the other atmospheres are not offered:

- **Key light:** KeyLightColor × KeyLightIntensity is the apply programs' `LIGHT`. Diffuse is albedo × `LIGHT` × N·L with no 1/π, so the viewer's sun (and a Blender sun) gets irradiance π × `LIGHT`. Its direction is (cos el sin az, sin el, cos el cos az) in game space, azimuth from +Z towards +X. That convention is fitted from where the probes are brightest (Megalopolis day: key 220°, probe 209°; Savali sunset: 308° and 327°); the code that converts the angles is not traced.
- **Environment:** the default env probe, a BC6H cube in scene radiance (Megalopolis day: mean radiance (0.82, 1.25, 1.80)), decoded at mip 1 by `nwo.formats.bc6h` and sampled into an equirectangular map. It gives the diffuse and specular lighting maps, and the Rift Apart ambient light convolves it with the engine's probe program ([Ambient light](#ambient-light)) as it does the review HDRI.
- **Colour correction:** the ColorCorrection SourcePath, an R8G8B8A8_UNORM LUT of 32³ or 33³ texels (39 distinct LUTs in 54 of the 95 atmospheres), applied at weight 1 ([below](#how-the-game-weights-a-colour-correction-lut)). `lut_megalopolis_warm9` lifts mid grey by about 0.05 with a slight magenta lean and saturates colours (red 0.75 becomes 0.84); despite its name it does not warm greys.
- **Light grid:** where the probe's level has a light grid, its bounce light replaces the probe's diffuse light ([Light grid](#light-grid)).
- **Placement:** the model has no place in the level, so the environment is turned about the vertical until the key light comes from the review HDRI sun's azimuth, keeping its elevation. That turn is a presentation choice.
- **Exposure:** the game's auto-exposure. `CS_AverageLumAccum` takes per pixel log₂ of the Rec.709 luminance, weighted by (1 − |x|)(1 − |y|) over the screen; `CS_AverageLumFinalize` soft-clamps the mean to (AdaptationAnchor − 13) ± AdaptationRange with a smootherstep scaled so the slope is 1 at the anchor; the display exposes by 0.18 / adapted luminance. The viewer meters its float scene target, with the probe composited behind the model along the camera rays, at most four times a second, and redraws when the exposure moves (`ModelViewport._meter_atmosphere`). Until the first metering it uses the anchor.

The **Blender Scene** export uses the atmosphere while it is the 3D view's lighting (`ExportOptions.atmosphere`, `blender_convert.py --atmosphere`, `nwo/export/rcra_atmosphere_blender.py`). The world becomes the probe at strength 1, the review sun becomes the key light (visible in glossy rays: the key light has a specular term; it casts no shadows, a presentation choice matching the game's gallery viewer, which shows models without self-shadowing, while the probe still occludes), and a 512-pixel render from the export camera is metered as above. The scene shows through the game's display as compositor nodes: the metered exposure, then the 32³ lookup of [Display transform](#display-transform) with the atmosphere's LUT mixed in, as a packed image sampled trilinearly at its log₂ coordinate, with colour un-premultiplied for the lookup so transparent film keeps its edges. The view transform is Raw. On the Metropolis blimp the compositor output matches `display.py` on the raw render to 0.001/255. The exposure is fixed at export: moving the camera does not meter again (the group's Exposure input can be changed by hand). The reopen check validates the probe image, the key light and the display (`rcra_atmosphere_blender.check`).

**Presentation fill (not game lighting):** after the metering, the `.blend` gets one more lamp, "Presentation fill from below (not game lighting)" (`rcra_atmosphere_blender.apply_fill`): a neutral sun shining straight up, 90° wide for soft shading, at a third of the key light's strength (a 3:1 key-to-fill ratio, a photographic convention) and invisible in glossy rays. It opens up undersides the game leaves in the key light's shadow; since it comes after the metering, it brightens the shot instead of being exposed away. Its strength can be changed or the lamp deleted. The viewer does not draw it.

Not reproduced: the level's geometry and shadows, local lights, fog, the sky model, the atmosphere's own bloom settings (the viewer keeps the default atmosphere's; the `.blend` has no bloom), AdaptationBias, ExposureBias and LuminanceKeyValue (their GPU path is not traced; all neutral in Megalopolis day), and the about 60 graphs that scale emission by the adapted luminance (they keep the review value 0.18, [Exposure](#exposure)).

##### Light grid

A level's bounce light is baked into light grids, `<level>/<name>_light_grid_<time of day>.zonelightbin` (648 in the install), read by `nwo/games/rcra/light_grid.py`:

- **Asset:** two toc rows. The headered one is a DAT1 (type `0xFA8D90B3`): section `0x27204B67` lists the brick centres (f32 × 3, game space) and `0xC72A514C` one u32 offset per brick plus the end, into the headerless row. A brick is 16 × 16 × 16 cells of one unit; cell x + 16 y + 256 z.
- **Brick** (`unpack_brick`, the CPU unpack at `0x1410a06a0`): 1024 bytes of 2-bit codes (high bits of each u64 first), six u16 planes of the N stored cells, N bytes (bits 6 to 13 of the cell's last word: probe weight and exponent), and three bleed-term bytes per code-3 cell. Code 0 is a blocked cell, 1 a cell the GPU fills from its neighbours (three passes, programs `0x3ce17bc`, `0x3ce689c`, `0x3cfb1ac`), 2 and 3 stored cells. Every Megalopolis brick's size is 1024 + 13 N + 3 N₃ bytes.
- **Decode** (`irradiance`, the apply programs' `g_LightGridData` reads): an ambient cube. Each axis word holds the + direction in its low 16 bits and the − direction in the high 16, each 6 bits Y and two 5-bit chroma fields. The programs sum the fields of the directions the normal faces, weighted by n|n|, then decode u = (b / 992 − 1), v = (c / 31744 − 1) as signed squares to (Y (1 − u + v), Y (1 + u), Y (1 − u − v)) × 2^(e − 10) / 63: diffuse light with no 1/π. They blend it with the default probe by the probe weight max(cell bits 6 to 9 << 6, lookup word & `0x3FC`) / 960.
- **The spot** (`street_cell`): the model has no place in the level, so one stored cell standing on a surface (a blocked cell directly below it) lights it: across the grids of the probe's level folder and time of day, the one nearest the median log luminance of all such cells in each of the six directions. Megalopolis day: 65,717 such cells; the chosen one, at (861.5, −117.5, 1891.5), gives about 0.50 from above, 0.30 from the sides and 0.22 from below (luminance), neutral grey from below, and probe weight 0. The probe alone gives about 1.2 in every direction. The choice is a presentation choice; the cell's values are the game's.
- **World** (`world`, `fit_world`): a world cannot give diffuse and glossy rays different light, so the environment becomes k × the probe plus 96 smooth non-negative lobes, fitted so its diffuse light (cosine convolution / π) matches (1 − weight) × the cell's light + weight × the probe's at every normal. k is the largest probe share that does not worsen that fit, up to the game's reflection scale (the cell's light over the probe's); it keeps part of the sky's detail in reflections. Megalopolis day: k 0.15 (the game's scale is about 0.28), diffuse within 6.5 % RMS, 19 % at worst. The viewer and the `.blend` both use this environment, so the metered exposure follows the dimmer street-level frame.

Not reproduced: the trilinear blend and bleed term between cells (one cell lights the model), `LightGridIntensity` (taken as 1: its place in the built atmosphere block is not traced), the lookup word's probe weight (0), and the local probes and screen-space reflections the game reflects at street level.

##### How the game weights a colour-correction LUT

Traced in `RiftApart.exe`:

1. **Blend:** every frame the atmosphere update (`0x1411d3500`) blends the active atmospheres. The callback for texture fields (`0x1411d62b0`) adds each atmosphere's texture to that field's list (at most four, sorted by weight, `0x1411d4de0`) with the atmosphere's blend weight, and keeps the sum. A sum below 1 is topped up from the base list (`0x1411d5960`).
2. **Copy:** the ColorCorrection list (field kind 11, a blended texture list) is copied to the post-process state at `+0x310` (`0x1411d3b3a`), whose `+0x70` and `+0x88` hold the tone-map Enabled word and the bloom settings (built-block offsets 632 and 656).
3. **Dispatch:** `0x1411db740` picks `CS_ToneMapCreateLookup` for an empty list or `…CC1` to `…CC4` by its count, binds the LUTs to `g_ColorCorrectionTexA` to `D`, and copies the four weights into `m_B`.

So in one atmosphere, fully blended in, its LUT has weight 1 and replaces the encoded value entirely; during a transition the two atmospheres' LUTs cross-fade. Two other sources change nothing on PC: `ColorCorrectionSystem` (`0x1411cb1a0`) would layer LUTs from `system_ColorCorrection.config`, which the install does not ship, and a debug toggle (`EnableColorCorrection`, in the engine's debug-settings table) empties the list when set. `display.colour_corrected` is the port; translated by `nwo.shaders.sm5` with the game's SDR constants and a real LUT bound, the CC1 program matches it to about 1e-5 for a 33³ and a 32³ LUT at weights 1 and 0.35.

#### Missing textures

`RiftApart.exe` builds its default textures at start-up ("TextureDefaultsInit"). When a material texture's asset does not exist, the asset manager reports it missing ("did not load because the asset does not exist"). The material loader's default for the slot then stays bound in its place. The texture's own graph default is not tried. `blender_export.replace_missing` binds the same default (`texture.ENGINE_DEFAULTS`), and the material's limitations name each texture and its stand-in. A unit that no slot row names reads 0, as an unbound D3D resource does (`texture.UNBOUND`).

The default follows the slot type, the high u32 of the slot hash. Each default is a 4 × 4 `R8G8B8A8_UNORM` texture of one texel value:

| Slot type | Rows in the install | Default | Texel |
| --- | --- | --- | --- |
| `0x49A7866B` (2D), and any other type | 29,765 | Default 2D | 128/255 in every channel |
| `0x628AD5A8` (1D) | 2,086 | Default 1D | 128/255 in every channel |
| `0x50BCB72A` (3D) | 685 | Default 3D | 1 in every channel |
| `0x884C4BF8`, `0x186494E9` (cube) | 122 | Default Cube | 128/255 in every channel |

107 slot bindings in the install name a texture that is not installed. The engine also builds "Asset Missing" textures (opaque blue) and "Asset Error" textures (opaque red), but a material texture never shows them. The missing hook runs while the reference is created, and the loader's default replaces its proxy afterwards.

**Coverage:** all 1,747 blended materials (the 77 cube-map materials translate; the cube arrays among them are all read at index 0), 27 of them in scene mode, which Blender does not draw.

### Render flags

`MatDrawCB.m_RenderFlags` is set per draw, traced in `RiftApart.exe`. The decal draw (`0x141038b60`, called for the scene's `DecalSubmitQuery` list, `0x141038800`) reads the material flags word ([Flags word](#flags-word)) and the renderer's pass id:

| Bits | Set when |
| --- | --- |
| `0x2` | The base for this packet kind. Another packet kind writes `0x381`, and the main instance draw below uses `0x1`. |
| `0x382` | Pass ids `0x14` and `0x28`, which also set `0x100` and never the lit bit |
| `0x400` | Lit: material flag `0x800`, only in pass ids from `0x2C` up |
| `0x100` | Material flag `0x1000`, or flag `0x1000000` under a global option |
| `0x4000` | Material flag `0x2000` |
| `0x4000000` | Material flag `0x80`: additive blend. The forward programs then apply a fog factor to the alpha. |

The main model-instance draw (`0x1411212d0`, the packet word stored at `0x141122aa5`) builds the word itself:

| Bits | Set when |
| --- | --- |
| `0x1` | The base, from the view's per-pass word (`0x101` under the same global option that sets `0x100` above) |
| `0x400`, `0x4000000`, `0x100` | The material bits above; here the lit bit has no pass-id test, and `0x4000` is not added |
| `0x8000` | The instance's geometry override (instance `+0x90`, a 128-byte record in a global table, freed by `ModelInst::FreeGeometryOverride`, `0x1410fb160`) is of kind 10 |
| `0x10000` | The instance's world matrix is mirrored: its 3 × 3 determinant is negative. The instance set-up (`0x1410fc6b0`) stores the transform's class in the low bits of `+0x5C` (`0x141629e60`: `4` for a negative determinant, `0x10` for non-uniform scale, bits 0 and 1 for the axis with the largest scale), and the draw takes bit 2. |
| `0x80`, `0x40`, `0x8000000` | Instance flags (`+0x80` bits 25, 8 and 26), and `0x80` again when the packet has a second record |

No draw sets `0x800`: none of the 117 functions that submit draw packets (callers of `0x141215870`) puts it in the render-flag word. Their other `0x800` constants go to the packet's sort word at `+0x2E`. The programs read the three per-instance bits this way:
- **`0x8000`:** set, the colour comes from `g_MainColorVBView` at half the subset's colour index; clear, from `g_MainPaintVBView` at the index.
- **`0x800`:** with no colour stream, the colour is 0 when set and 1 when clear.
- **`0x10000`:** negates `TEXCOORD3.w`, so the bitangent stays right-handed under a mirrored transform.

**Where the colour lives.** Every model stream sits in one shared geometry buffer, allocated through one allocator (`0x1465a4260`) and read through the typed views of [Vertex colour](#in-the-viewer). The model loader (`0x1410e7ec0`) uploads, per model:
- the vertices, 16 bytes each
- the two 4-byte streams, UV1 and the colour section
- the R16 indices

The colour section is RGBA8 there, and the subset's `m_VertexColorIndex` counts 4-byte elements. So a plain instance, with `0x8000` clear, reads the model's own colour section through `g_MainPaintVBView`, the RGBA8 view. Per-instance vertex paint (`VertexPaint`, `0x141103d30`, for models of kind 4) is allocated as RGBA8 in the same buffer. The `0x8000` path halves the index, so `g_MainColorVBView` is an 8-byte view, where kind-10 geometry overrides keep their colours.

The pass ids matter only to decals. The model instance draw sets the lit bit with no pass test, so a model's G-buffer and forward passes get the same flags.

So the translations bind the plain instance draw: base `0x1`, `0x8000` clear, and the lit bit in both passes (`material.draw_flags`), as the game sets it for a model. `0x800` stays clear, as no draw sets it, and `0x10000` stays clear for an unmirrored draw ([Vertex programs](#vertex-programs)).

### Per-object values

Programs read the engine's per-object record, `g_SceneObjectView` (element `$Element`, 128 bytes; names and offsets from the reflection). The engine's model vertex program shows its layout:
- **Transform:** row-major 3 × 4, rows at bytes 0, 16 and 32, translation in dword 3. The world normal is the object normal dotted with each row.
- **Values:** alpha, an fp16 object colour (`m_ColorRGFp16`, `m_ColorBMiscFp16`), custom XY, Z and random halves, creation time, LOD override.
- **Interpolants:** the vertex program passes the custom values on as `TEXCOORD6 = (1, Misc, Rand, LOD factor)` and `TEXCOORD7 = (custom X, custom Y, custom Z, FX lighting index)`.

The game sets these per object at run time. `shaders.NEUTRAL_OBJECT` is a plain object: identity transform, alpha 1, white colour, and every custom value, time and bias 0. Both the G-buffer and the forward translations bind it, and a material that reads it says so in its limitations. A visual-effect mesh replaces its colour, alpha and custom members with its node's, and a particle quad binds its interpolants the way the engine's particle vertex program writes them ([Draw values](#draw-values)).

### Vertex programs

208 graphs (583 material uses) replace the engine model vertex program with their own, in pass slots 3, 7, 10, 13 and 17: vertex-animation textures (crowds, prison-cell idles, the Zurkon pub band, swarmer eyes and dials), wind sway, flags, beams and shockwaves, rotating debris, and offsets along the normal. `nwo/games/rcra/vertex.py` translates them with the SM5 translator and places each mesh's vertices where its material's program puts them at rest.

**Vertex pulling.** The programs read the vertex themselves from `SV_VertexID`. The streams are typed views, and each read lands on the vertex's own element of the 16-byte vertex ([Vertex decoding](#vertex-decoding)), which the translation checks for every stream:

| View | Element | Read as |
| --- | --- | --- |
| `g_MainVBPosView` | 2v | int16 x, y, z, w, scaled by the subset's `m_MetersPerUnit` (the model's position scale) |
| `g_MainVBNrmTanView` | 4v + 2 | The normal word as four UNORM channels, 10, 10, 10 and 2 bits. The programs rebuild the normal from them as `model.unpack_normals` does. |
| `g_MainVBUV0View` | 4v + 3 | int16 u, v, times the UV scale of `m_GeomLodAndUVScales` bits 0 to 3 (UV1: bits 4 to 7) |
| `g_MainVBUV1View`, `g_MainPaintVBView` | v | UV1 and the colour section (a plain instance, [Render flags](#render-flags)) |

The subset record (`g_ModelSubsetView`) gives the stream offsets, the scale and the UV-scale word; the paint and skinned streams are not read for a static subset.

**Bindings** (`vertex.translate`; review values, labelled `REST_NOTE`):
- **Draw:** `MatDrawCB.m_RenderFlags` holds the plain instance draw's 0x1 and the material's lit bit ([Render flags](#render-flags)). The programs test the two low bits only together and write no geometry when both are clear: all 583 material uses translate the same with 0x2. With 0x8000 clear they read the model's colour section through `g_MainPaintVBView`. All 583 read the colour, and their translations are the same as with 0x8000 set and the section bound as `g_MainColorVBView`.
- **Object:** the neutral object of [Per-object values](#per-object-values), and the same transform as its previous one.
- **Engine:** the shader timer `GlobalWorldCBuffer.m_ShaderTimer` is an input of the translation, run at 0 for the at-rest pose (Motion over time, below). Every other engine constant is 0: no wind, no ambient animation, no force fields (the per-tile force-field bitset is empty).
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

**Motion over time.** A batch whose vertices differ at shader time 0.75 s (`CHECK_TIME`) from time 0 moves over time. `deform_model` keeps such batches in `decoded['vertex_motion']`, a `vertex.VertexMotion`: its recipe, the vertices' inputs, texture reader and stored normals. `at(seconds)` runs them at any shader time (`vertex.at_time`) and returns each moving mesh's positions and normals, at time 0 the at-rest ones. The mesh's note says it moves (`TIME_NOTE`).
- **Viewer:** **Vertex animation** (Materials group, shown for such a model, off by default and remembered) runs the shader clock from 0 in real time. A worker evaluates the newest time (`loader._motion_payload`), and the viewport puts those vertices in place of the rest ones before blend shapes are added (`ModelViewport.set_vertex_motion`). Evaluation per frame: the flag 2 ms, the swarmer 16 ms (27 meshes), the stadium crowd 337 ms with all its LODs (550 meshes, 553,153 vertices). The viewer runs only the shown meshes (`VertexMotion.at(seconds, meshes)`), which plays the crowd's Highest LOD (14,897 moving vertices) at about 10 frames a second in the app; a heavier model steps at the rate its frames finish.
- **Export:** **Vertex animation** under Options bakes 4 s at 15 fps (`models.vertex_frames`): one morph target per frame, named `time <seconds>`, and the glTF animation `Vertex animation`, whose weights put frame k at full weight at time k / 15 and blend linearly between frames. Blender imports them as shape keys and an action. It is large: each frame is a full copy of the moving vertices' deltas.
- Wind, force fields, the object's custom values and the camera stay at their rest values over time. The baked span is not the motion's period: the crowd's frame index loops every 1 / 0.1277 s (7.8 s).

`tests/test_vertex_motion.py` checks the timer input, a motion batch, the payload, the viewport replacement and the baked export without game files. With `RCRA_GAME` it checks that the swarmer and the flag move between 0 and 0.5 s with time 0 equal to the at-rest vertices, and that a still plant has no motion.

**Limitations:**
- On a skinned mesh, the at-rest offset (and the motion over time) is added to the bind pose, so its joints carry it; the game adds it after skinning, in object space.
- A moved mesh takes the program's tangent output (`TEXCOORD3`) with its normal. Where that output does not translate, the stored tangent stays, and the note says so.

### Fur materials

A fur material (19: Ratchet's and Rivet's head, ears and limbs, Daxter, the sheep, a carpet, a towel, a cotton canopy and a cloud) has no graph. In place of the serialized parameters it carries a fur record, DAT1 section 0xD9B12454 (`fur.fur_record`; the viewer's preview takes the first three textures as its slots). The executable names its fields: the material loader copies them into its fur object, and a per-instance override pass writes each field by its name hash.

| Record u32 | Field | Ratchet's head |
| --- | --- | --- |
| 0 | `Fur_LayerCount` (integer) | 32 |
| 1 | `Fur_LoDReduction` | 0 |
| 2 | `Fur_Length` | 0.03 |
| 3 | `Fur_Density` | 16 |
| 4 | `Fur_OffsetScale` | 1 |
| 5 | `Fur_GlossScale` | 1 |
| 6 | `Fur_SpecularScale` | 1 |
| 7 | `Fur_TransmittanceScale` | 0.1 |
| 8 | `Fur_Wetness` | 0 |
| 9 to 12 | Offsets into the material's DAT1 of `Fur_BaseMap`, `Fur_NormalMap`, `Fur_GlossMap` and `Fur_ControlMap` | `_c`, `_n`, `_g`, fur control |

`Fur_WindStrength` and `Fur_WindTurbulence` follow in the fur object and are 0 from the file. Every installed record holds wetness 0.

The executable holds the fur programs, found by their reflection names (`fur.fur_writer`): the fur G-buffer writer, a ps_5_0 with constant buffer `MatrialFurCBuffer` (`m_Density`, `m_Length`, `m_GlossScale`, `m_SpecularScale`, `m_TransmittanceScale`, `m_Wetness`) and textures `g_BaseMap` (t5), `g_FurLayerMap` (t7, 3D), `g_FurGlossMap` (t8) and `g_FurControlMap` (t9), writing the four engine targets; a five-target variant; and the shell vertex programs (`ModelFurCBuffer`). `surface.surface` takes the writer as its `program` (`fur.fur_program`), so the material gets the same terms and packing as a graph's writer: albedo from the base map, F0 and gloss from the gloss map, occlusion from the control map, and the vertex normal (the writer binds no normal map).

The fur draw (render command `0x21`, "ModelFurShellGBufferDeferred") copies six fields into `g_Fur` unchanged. `fur.fur_constants` binds them the same way:
- `m_Density` from `Fur_Density`
- `m_Length` from `Fur_Length`
- `m_GlossScale` from `Fur_GlossScale`
- `m_SpecularScale` from `Fur_SpecularScale`
- `m_TransmittanceScale` from `Fur_TransmittanceScale`
- `m_Wetness` from `Fur_Wetness`

It binds the base, normal, gloss and control maps at t5, t6, t8 and t9. The shell constant buffer (`ModelFurCBuffer`) takes `Fur_OffsetScale`, one over the drawn layer count (`Fur_LayerCount` cut by `Fur_LoDReduction` with distance), `Fur_Length` and the wind terms clamped to [0, 1].

**Layer map.** `g_FurLayerMap` is the engine's "Default Fur Shells" texture, built at start-up, so no file of it ships. It is a 128 × 128 × 32 R8 volume made from the engine's xorshift128 generator, seeded with the canonical state (123456789, 362436069, 521288629, 88675123). For each texel column, x outer and y inner:
- **Strand length:** n = min(⌊32 (r / 2²⁴)²⌋ + 1, 32), where r is the next random number shifted right by 8.
- **Layers:** layer d < n holds ⌊255 sat(1 − 0.8 max(2 (d + 1) / n − 1, 0)²)⌋.

The generator saves and restores the global random state, so the map is the same on every run.

**Shells.** The fur draw renders the fur meshes `Fur_LayerCount` (L) times, instanced. The shell vertex program (`VS_ModelFurShellGBufferDeferred`) gives each instance a layer k at height t = k / L and spreads the layers by the view:
- **Spacing:** t' = sat(t / s), with s = min((1 − sat(N·(−V) − 0.0065 q − 0.1))² + 0.1, 1), q the stored `TEXCOORD3.w` magnitude. A layer whose t / s exceeds 1 is culled. Facing the camera a few layers span the strand; at the silhouette (s = 1) every layer is drawn at t' = t.
- **Position:** the vertex moves along its normal by control.b × t' × `Fur_Length` (the control map point-sampled at UV0, mip 0).
- **Interpolants:** `TEXCOORD.z` = 32 t', `TEXCOORD1.w` = `Fur_OffsetScale` (signed by the bitangent), `TEXCOORD2.w` = (t'² + 0.4 t') `Fur_Length` `Fur_OffsetScale`, and `TEXCOORD3.w` = 1 on the base layer (k = 0), else 0 (`fur.shell_layer`).

The writer cuts each layer. It reads the layer map at slice round(`TEXCOORD.z`) and at uv0 × `Fur_Density`, shifted along the control map's comb direction ((control.rg × 2 − 1) × |`TEXCOORD1.w`|) by `TEXCOORD2.w`. It keeps the pixel when sat(layer × base.a + `TEXCOORD3.w`) × sat(1 − t' / 2) exceeds a screen-space dither threshold. The game's temporal filter averages the dither, so both renderers draw that product as the surface's `coverage` term (`fur.shell_coverage`). A layer's occlusion is lerp(control.a, 1, t' / 2); its other terms are the base surface's. Layer 0 is the base surface (coverage 1).

- **Viewer:** `viewer_materials.shell_receipt` translates the writer once with the interpolants as renderer inputs (`fur.shell_symbolic_program`). After the opaque batches, `rcra_glsl.shell_vertex_source` draws each fur batch again for layers 1 to L − 1, inner layer first, with the view spacing, cull and displacement per vertex. Each layer is blended over by its coverage without depth writes.
- **Blender:** `rcra_blender.build_fur_shells` gives every fur object one linked copy of its fur faces per layer 1 to L − 1, each with that layer's material (`fur.shell_layer` values as constants) and a Displace modifier ahead of the rig (the control map's blue at UV0 times t × `Fur_Length`). Coverage is the material's alpha. Every layer is drawn at the silhouette spacing, because a Blender material cannot move its own vertices by the view.

What is not decoded, and labelled (`fur.FUR_NOTE`, `fur.SHELL_NOTE`, `fur.SHELL_VIEW_NOTE`):
- **Wetness at run time:** gameplay can raise `Fur_Wetness` through the override pass and the FurWetness settings, for example in rain. The material keeps the record's dry value.
- **Layer map mip:** both renderers read the layer map at its top mip. The writer's view-dependent mip bias is not applied.
- **Control map filter:** the viewer reads the control map bilinear in the vertex stage, where the game point-samples it.
- **Distance and wind:** `Fur_LoDReduction` and the wind terms are not applied; every layer is drawn and the strands are at rest.
- **Other formats:** glTF, OBJ, FBX and USD keep the base layer only; the shells are in the `.blend` alone.

## Blender materials

A Blender export (`.blend`, FBX or USD, through `nwo/export/blender_convert.py`) replaces each imported glTF material with the game's own material. The work splits into three modules:

| Module | Role |
| --- | --- |
| `nwo/games/rcra/surface.py` | Translates the G-buffer writer (slot 4) and unpacks the engine's packing into surface terms: a recipe. For blended materials it also translates the forward program (slot 1) into colour and alpha. |
| `nwo/games/rcra/blender_export.py` | The `Game.blender_materials` hook: recipes and unit textures, written as `rcra_materials.json` beside the export. Each material's entry (`cached_material_entry`) is kept across sessions: the entry in the source's translation disk cache, its unit textures in a content-addressed store beside it (`<game cache>/blender_images`, at most 2 GB). The key (`entry_cache_key`) covers the material and graph bytes (or the fur program), the archive's identity, the translation code (`viewer_materials.CODE_SHA256`) and the image-writing code (`IMAGE_CODE_FILES`); a missing stored image means translating again. Each fur shell layer is cached the same way (`cached_shell_layer`). An export first translates what is not cached in worker processes (`translation_units`, run on a `viewer.loader.TranslationPool`): the materials without fur and every shell layer, then the fur materials, which read their layers back; each worker keeps one image folder and texture cache per export, as `blender_materials` does |
| `nwo/export/rcra_blender.py` | Runs in Blender: the "RC RA Deferred BRDF" group, and one Math node per recipe operation. A division by a constant 0 (a neutral profile value) is x × Exponent(10000), which is x × inf as in D3D; Blender's Divide alone returns 0. A `lookup` (below) is a one-row float32 image read with Closest. Saturate is Maximum then Minimum, which drop NaN as D3D does |

### Surface terms

The packing is the same engine code in every graph (see [G-buffer and deferred lighting](#g-buffer-and-deferred-lighting)). `surface` translates the writer with the material bound and unpacks:

- albedo, F0 rgb, gloss and the world normal
- occlusion, or emission `E` (self-lit `albedo × E`)
- for an opaque material on a forward technique, the forward program's unlit colour `unlit.r/g/b` ([Techniques](#techniques))
- the writer's discard, which becomes alpha clip

F0 and gloss keep the game's 8-bit rounding. A material passes only when every term is a float expression of the mesh interpolants (UV0, UV1, normal, tangent and sign, vertex colour, front face) and its own textures and constants. Integer arithmetic on constants and at most four boolean conditions, such as a triplanar writer's dominant-axis index, is evaluated in every case and becomes a select of constants. This is exact. Other integer arithmetic of float values (cell indices, hashes, a remainder of case constants) is carried in float on the same integers: casts round toward zero, shifts and masks become floored division and modulo, and a remainder of a select of constants folds per case. It is exact while every value stays within ±2²⁴, and the result says so. A word the program packs from fields (an or of shifted values, a `bfi`) and reads back with a mask, `ubfe` or a shift gives each field its own value, which holds while every field fits below the next one, as the packing's scales are set for; that is labelled too. Over the materials these rewrites gain, the terms are bit-identical to the unrewritten translation at random inputs. A normal word chosen by a select between two packed words is unpacked per case. The snow writer does this: its snow arm writes another lighting class, but it is gated by the deformable-snow inputs, which hold review values of 0.

Values a material does not own are bound and listed in its limitations:
- **Neutral draw:** no dissolve, opaque, no render flags (lighting class 0), shader time 0.
- **Volume flags:** the per-tile bitset of engine flag volumes (`g_VolumeFlagBitsetView`, whose bits index the boxes in `GlobalVolumeFlagBuffer`) is empty in every pass, so no flag volume covers the material. Rain on glass and the rain, snow and puddle writers read it to find shelters.
- **Material LOD distance:** the per-material and per-subset records (`g_MaterialView.m_LodDist`, `g_ModelSubsetView.m_MaterialLodDist`, `shaders.NEUTRAL_RECORDS`) hold no distance, like the neutral object's override. Writers then take no LOD fade, so they draw at full detail. Other members of these records are not bound. The material record's alpha (`g_MaterialView.m_Alpha`) is 1, opaque, as the neutral object's.
- **Review values:** screen position for dither, LOD fade, and the exposure: the adapted and measured luminance and the tone-map anchor at 0.18 (`m_AdaptedLuminance`, `m_ActualLuminance`, `m_ToneMapAnchor`; `REVIEW_ADAPTED_LUMINANCE`, see [Exposure](#exposure)). Emissive effects such as the electricity arcs and the planet surfaces scale by them. The review render has no automatic exposure.
- **View direction:** under the review screen position the camera sits at the shaded point, so a writer that normalises the direction to the camera (tangent-space parallax in the holo and dimensional glass, fresnel, the emissive sequence, the two-layer snow) gets a normalised zero vector, NaN. When a term depends on NaN, `surface` translates the writer again with the forward profile's view (`_ViewBindings`: `VIEW`, `CAMERA`, see [Forward profile](#forward-profile)) and uses that when no NaN is left, with a limitation saying so (449 materials). The snow writers then read the deformable-snow map, `g_DeformablesView`, bound empty (0) as its constants are.
- **Undefined terms:** 77 writers still divide by a per-object value the neutral object holds at 0: the stadium crowd's vertex-animation frame (0/0) and a butterfly wing's random seed. Their limitations say which terms are undefined (`NAN_NOTE`); renderers resolve the NaN by D3D rules.
- **Loop flags in float registers:** an unrolled loop can keep its exit flag as a boolean mask in a register that later iterations read as a float (`bits2f(b2i(c))`, NaN when set: the parallax loop of `jn_glass_01_opaque`). It becomes `select(c, NaN, 0)`, and `_prune_nan` then rebuilds every select whose arms reach a NaN with its condition known in each arm (through not, and, or, and the selects the condition reads, such as the loop counter), so selects on a condition that follows take their arm. The loop's own conditions rule the flag out of every term, so no NaN is left; the prune is exact, and a NaN a term does read stays.

**Coverage:** 5,712 of 5,721 materials, the 19 fur materials and `jn_glass_01_opaque` (a loop flag in a float register, above) among them ([Fur materials](#fur-materials)). Not translated: eight materials name a graph by file name that the install does not contain (`material.no_graph_reason`), and one has no graph at all (`flowmap_water_custom`).

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
| Specular | Metallic BSDF, GGX, roughness r² with r = `0.9725 − 0.7514 gloss`, so Blender's α (roughness²) is the game's r⁴; F82 Fresnel with a white edge tint (Schlick with F90 = 1) |
| Diffuse | Diffuse BSDF, roughness 0 |
| Self-lit | Emission: albedo at strength `E` |
| Unlit colour | Emission of the surface group's `Unlit` output at strength 1, added to the group's shader for an opaque material that has the `unlit` terms ([Techniques](#techniques)) |

Light strength is π × the game's light colour, as for R&C 2016. Cycles uses the exact Smith term rather than the game's joint approximation. The game applies occlusion to ambient light only, which Cycles cannot separate; the group's "Occlusion on diffuse" input (default 0) applies it to the diffuse colour instead.

The group is the class-0 BRDF. Cycles has no closure that runs another apply program per light. Classes 1, 2, 4 and 5 ([Lighting classes](#lighting-classes)) have their own groups, built from the closed forms in `nwo/export/rcra_closures.py` (checked against the translated programs by `tests/test_rcra_lighting.py`):

| Class | Group | Specular | Diffuse |
| --- | --- | --- | --- |
| 1 subsurface | "RC RA Subsurface BRDF" | exact: (1 − m) GGX(α) + m GGX(α^1.66), m = sat(0.15 − 0.03 × View Z Depth) | Cycles' Christensen-Burley subsurface at the skin pass's distance d = profile texel × 0.005 × (0.6 occlusion + 0.4) × (2^f − 1) / 7 (f: extra bits 15 to 17, `EXTRA_FIELDS[1]` field 5). Cycles 5.0's Burley has d = radius × scale / 4π at any albedo (measured on a shadow edge, within 1 % at albedos 0.2, 0.5 and 0.9) |
| 2 cloth | "RC RA Cloth BRDF" | on the second normal (8-bit octahedral about game Y): mix(GGX, sheen, field / 31), r = 0.9725 − 0.6011 gloss; GGX exact, the Charlie sheen a fitted Ashikhmin velvet (σ and an F0 scale keeping its mean reflectance; about 55 % L2 difference in shape) | Diffuse and Translucent BSDFs weighted by the shape's irradiance from each hemisphere (2 ∫ f), per channel from a 129 × 32 table |
| 4 anisotropic | "RC RA Anisotropic BRDF" | below | Lambert |
| 5 foliage | "RC RA Foliage BRDF" | class 0's | wrapped diffuse and back transmission as Diffuse and Translucent BSDFs of the same irradiance |

The projected diffuse keeps ambient light; a single light falls off as Lambert on each side. Not drawn: class 1's and class 2's Henyey–Greenstein scatter terms (g = 0.65, small except at grazing views) and class 1's transmission through the shadow-map depth. Class 3 (hair) takes the class-0 group (`blender_export.CLASS_LIMITATION`). The glTF export's roughness is r² (`models.rcra_roughness`).

**Anisotropic group** (class 4, "RC RA Anisotropic BRDF", `rcra_blender.build_anisotropic_group`). It has the class-0 diffuse, self-lit and occlusion terms. Its specular is two Metallic BSDFs (GGX, F82 with a white edge tint), mixed at 0.5. The surface group passes the five G-buffer fields as `Extra 0` to `Extra 4`. The group decodes the tangent from them, turns it into Blender space and maps the lobes with `lighting.cycles_anisotropic`:

| Cycles input | Value |
| --- | --- |
| Roughness | r1 r2, with r1 = r(gloss) and r2 = r(second gloss), so Cycles' αx αy = αt αb |
| Anisotropy | (1 − (min r / max r)⁴) / 0.9, the game's αt / αb |
| Tangent | the decoded G-buffer tangent |
| Rotation | lobe 1: 0; lobe 2: the rotation field's angle / 2π. Both turn a further quarter when r2 > r1: Cycles has no negative anisotropy, so its x axis goes onto b |

Cycles' mapping was measured in Blender 5.0 with single-sun renders of a plane against the analytic lobe, within 4 × 10⁻⁶: αx = R² / √(1 − 0.9 A) along the tangent, αy = R² √(1 − 0.9 A), A clamped to [0, 1], Rotation turning the tangent right-handed about the normal. So the ratio of the two alphas caps at 10, keeping their product. Two limitations are labelled (`blender_export.ANISOTROPIC_LIMITATION`): Cycles uses the exact anisotropic Smith term where the game uses its separable approximation, and Cycles makes the tangent orthogonal to the normal. `test_blender_anisotropic_group_renders_the_class_4_lobes` renders the group in Cycles and compares it with the analytic lobes of the mapping, within 10⁻⁴. `blender_check` also bakes the fields (`extra_a`, `extra_b`): on Clank's body they agree with the recipe except at under 1 % of texels, by one integer step. Clank's body, head and limbs materials take the group.

### Blended materials

A blended material's entry carries `blend` and a `forward` recipe ([Forward profile](#forward-profile)), and keeps its G-buffer `recipe` when that translates. The Blender material composes them over a Transparent BSDF:

| Mode | Shader |
| --- | --- |
| Blended | Transparent(mix(white, B, alpha)) + alpha × (BRDF + Emission(colour)); with B = 0 this is mix(Transparent, BRDF + Emission(colour), alpha). The BRDF's own emission input is left unlinked, because the forward colour already holds it. |
| Additive | Transparent(1 + alpha B) + Emission(colour, strength alpha) |
| Modulate | Transparent tinted by mix(white, colour, alpha) |

A material that lights the scene behind it (`lit_background`, [The scene behind](#forward-profile)) needs the diffuse light reaching the point, which Cycles has no node for. `surface.unit_irradiance` folds it at a unit white light, B = background + lit_background, and the material's limitations say so (`UNIT_IRRADIANCE_NOTE`).

B is the background tint of a program that passes the scene behind it through ([Forward profile](#forward-profile)); a modulate material that does so is not linear in it and keeps its opaque surface.

Back-face culling follows the flags word's DoubleSided bit (`material.double_sided`; [Double-sided](#double-sided)): off for a double-sided material, on for the others. Cycles ignores it.

`VIEW` and `CAMERA` come from Geometry Incoming and Camera Data View Distance. EEVEE materials use the blended render method. The engine names the modes, but the blend factors are the standard ones for those names and are labelled as such. A blended material whose forward program does not translate keeps its opaque surface, with the reason in its limitations.

A blended material whose alpha makes the plain instance draw skip it ([Material alpha](#material-alpha)) is not exported. The model export leaves its meshes out, unless they are all the model has, and `export.json` gives the note.

### Checks

Export check: each export builds the `.blend` as a candidate, reopens it in a second Blender (`blender_convert.py --validate`), then moves it into place. For a Blender Scene the check also compares the world, HDRI and colour management with the recorded review lighting.

`python -m nwo.games.rcra.blender_check --game <install> --blender <blender.exe> --output-root <folder> <material> ...` bakes every surface output of the emitted node tree in Cycles. It compares the bake, texel by texel, with the recipe evaluated by the translator on the same textures:
- **Window:** a UV window small enough that the bake's sub-pixel jitter stays below a 32nd of a texel, so a material can tile its UV up to 8 times. With the old quarter-texel window, a triplanar rock that tiles ×2 differed by 0.11 at sharp texels.
- **Undefined pixels:** a pixel whose recipe samples a texture at a non-finite coordinate is left out and counted as `undefined`. This happens when a neutral profile value ends up as a divisor, such as a custom object value of 0 dividing a distortion offset in the waterspray material. Neither renderer defines that lookup.
- **Results:** on one material from each of the ten most-used graphs, F0 and alpha are exact. Albedo agrees within 1.1 × 10⁻⁶ and the normal within 3 × 10⁻⁵, and gloss, occlusion and emission within 8-bit rounding ties. The snow writer, a triplanar rock and a layered robot material agree within 3 × 10⁻⁵ too. Before the sRGB units were stored linear, they differed by up to 0.11, all from filtering between contrasting sRGB texels.
- **Forward:** a blended material's forward colour, alpha and background tint are baked too, with the view held at fixed values (`FORWARD_VIEW`). On the trooper dome and the wine glass, colour agrees within 2 × 10⁻⁶ and alpha within 6 × 10⁻⁸; a 128-slice noise volume within 5 × 10⁻⁶. A cube map is compared with the exact D3D lookup of its stored faces, so the difference includes the equirectangular resampling: on a laser-tube material the colour (values near 31) differs by at most 0.1, and on a synthetic cube with six different faces 93 % of the baked pixels agree within 0.02 and the rest sit on face seams. The 64-step convergence-tear portal march and the rain torso glass bake exactly.

### In the viewer

The model viewer's Shaded mode draws the translated programs (`nwo/games/rcra/viewer_materials.py` builds each material's receipt; `nwo/viewer/rcra_glsl.py` emits its GLSL, one statement per recipe node). The receipt follows the export's choices: the G-buffer surface for opaque materials; the forward program, plus the lit surface for blended ones. The viewer's native translation toggle switches back to the preview.

| Part | Viewer |
| --- | --- |
| Vertices | Where the material's vertex program places them at rest, evaluated when the model is decoded ([Vertex programs](#vertex-programs)) |
| Undrawn materials | Batches of a material the plain instance draw skips ([Material alpha](#material-alpha)) are left out. The texture entry carries `not_drawn`, and the material list shows the note. |
| Inputs | UV0 and UV1, the world normal, the decoded tangent and the whole `TEXCOORD3.w` (attribute `aTangent`, skinned with the vertex, interpolated as the game's interpolant; render flag 0x10000 clear), the vertex colour, front face, and the view from `vTintViewOffset`. A mesh without a tangent stream gets the UV-derivative frame (`native_glsl.FRAME_HELPERS`). |
| Textures | A sampled array slice as stored. Arrays with a run-time slice and 3D textures as the padded atlas of the Blender export, without mipmaps. A cube map as a GL cube map of its six stored faces (face k to `GL_TEXTURE_CUBE_MAP_POSITIVE_X + k`, rows top first), top mip, linear, seamless where the driver has it: D3D and GL share the face table, so `textureCube` along the game direction returns the D3D texel. Float data formats as `RGBA32F`. |
| Lighting | Each review light (key, fill, preview) goes through the material's lighting class's own apply program: `lighting.direct`, emitted as the function `rcraLight` and called once per light ([Lighting classes](#lighting-classes)). Inputs are in game space: the world normal, `vTintViewOffset`, the view depth, and the light turned back from view space through `vNativeToView`. `LIGHT` is the rig colour / π, so the game's convention (diffuse = albedo × N·L × `LIGHT`) gives the rig's old balance. Key and fill keep the rig's diffuse scale. Ambient light under the Shared HDRI lighting is the class's own ([Ambient light](#ambient-light), `rcraGameAmbient`): its probe reads look up the review probe, a GL cube with all six levels, at the program's lod (`textureCubeLod`, `GL_ARB_shader_texture_lod`) along its direction turned into the environment's frame, and its BRDF lookup read takes the engine's table, linear and clamped. The probe is built on a worker thread; until it is ready, under the viewer rig, or without the extension, ambient light is the rig's, with the class-0 BRDF (GGX α = r⁴) and occlusion on ambient light only. Forward materials with their [own key light](#own-key-light) add colour(L, `LIGHT`) − colour per light, plus their [own ambient light](#own-ambient-light) (`rcraFwdAmbient`) once the review probe is ready, else the G-buffer surface's ambient light. An opaque material on a forward technique adds its unlit colour (`rcraUnlit`) after the lit terms, and its unlit colour gates its bloom as emission does. Statements that read no light are emitted once and shared. |
| Blending | Forward materials draw in the float scene target, tested but not writing depth. A multiply pass scales what is behind by the background factor (blended: mix(1, B, α); additive: 1 + αB; modulate: the tint). B adds lit_background × the rig's diffuse light at the surface normal (`rcraDiffuse` with albedo 1 and no occlusion) for a material that lights the scene behind it. Then a colour pass adds the premultiplied colour, coverage blended over. A program in scene mode reads `BACKGROUND` from a copy of the frame taken before its batch (`_scene_copies`): scene-linear in the float scene target, display-referred otherwise. |
| Texture sizes | Programs that query a texture's size () get it from the full-resolution texture, as in the Blender export. Without it 143 materials failed in the viewer; with it 31 do (29 have no graph, 2 fail the G-buffer writer). |
| Caching | The translation (`viewer_materials.translate`: recipes, blend, limitations, bindings, or the reason it fails) is kept across sessions in the source's translation disk cache, keyed by the material and graph bytes and `CODE_SHA256`: a hash of `CODE_FILES`, the modules a translation runs or reads constants from (the translator side of `games/rcra`, `sm5`, `dxbc`, `formats/texture`, `common` and `rcra_glsl`), so editing the model decoder, blend shapes or the browser keeps the cache. The asset readers are left out, since the key holds the bytes they return. `tests/test_translation_cache.py` traces a few translations and checks that every function that runs is keyed, and that every package module the keyed files import is keyed or listed in `NOT_TRANSLATION`. Within one translation the forward albedo and the own specular term share one forward split (`surface.shared_forward_split`), which dominates both: the stadium crowd's eyeball program takes about 10 s to split. Infinite and NaN constants are stored tagged. Atlases are kept per source (least recently used out past 256 MB), since materials share their texture arrays, and 8-bit atlases are built in bytes. |
| Non-finite constants | Receipts resolve constant NaN by D3D rules (`surface.fold_nan`, as the Blender export does), since GLSL leaves min, max and clamp of NaN undefined. The inf and NaN constants that remain (a division by a zero profile value, a NaN in a select arm) are made at run time from a uniform that is never set, `rcraInf = 1 / uRcraZero` and `rcraNan = uRcraZero / uRcraZero`: GLSL 1.20 has no literal for them, and the compiler cannot fold a uniform. |

**Viewer check.** `python -m nwo.viewer.rcra_check --game <install> --output-root <folder> <material> ...` is the viewer's counterpart of `blender_check`. It emits a material's receipt with `rcra_glsl`'s own term emitter into a check program that draws the same UV window and inputs on a float target, four terms per pass, and compares each term with the recipe evaluated on the viewer's uploads (the driver's decoded top mip, read back with `glGetTexImage`; the unpadded tiles of an atlas; the D3D lookup of a cube's faces). `over` counts pixels whose difference exceeds 0.01, relative above 1. On 57 materials (one from each top-level folder, twelve forward materials, fur, a cube map, the lit background, a volume, the dome, NaN writers and the god-ray lookup tables), 54 have none and one has three, a threshold the GPU's rounding crosses; the Blender check of the same set has 53 within 0.01, the rest being the laser tube's resampled cube (0.058), the shield below, and the crowd. Otherwise terms agree within 8-bit steps, and a fresnel shield whose colour reaches 8 × 10⁵ under the neutral values agrees within 4 × 10⁻⁴ relative (GPU `exp2` and `log2`). The stadium crowd's two writers are undefined everywhere (their 0/0 frame). Screen derivatives are the quad's in GL and 0 in the reference.

**Vertex colour.** The engine model vertex program loads a colour, squares it channel by channel and passes the square on as `TEXCOORD4`, so the square is interpolated. Which colour it loads depends on the draw:
- **Model colour:** when the subset's colour index (`g_ModelSubsetView.m_VertexColorIndex`) is not negative, `g_MainPaintVBView` at that index plus the vertex index: the model's colour section, for a plain instance. With render flag 0x8000 (a kind-10 geometry override), `g_MainColorVBView`, an 8-byte view, at half the index.
- **Object paint:** when the subset's paint offset (dword 3) plus the object's (`g_SceneObjectView` byte 0x4C) is positive, `g_MainPaintVBView` there, plus the vertex index, replaces it.
- **No colour stream:** 1, or 0 with render flag 0x800.

The viewer reads the model's colour section (`0x5CBA9DE9`) as normalized bytes in R, G, B, A order. The translated materials' vertex program (`rcra_glsl.vertex_source`, attribute `aRcraColour`) squares it. The engine's buffer manager (`D3DBufferManager::InitBuffer`) creates one typed view per format over the shared geometry buffer. Its only four-channel 8-bit view is `R8G8B8A8_UNORM`, the view a plain instance reads the colour through. It creates no BGRA view, so the colour bytes are R, G, B, A in order, as the reader takes them. The other views are `R32G32B32A32_UINT`, `R16G16B16A16_SINT` and `_FLOAT`, `R10G10B10A2_UNORM`, `R32G32_UINT`, `R16G16_SINT`, `R16_UINT` and a raw view. A model without the section reads 1. Object paint and geometry-override colours are not drawn: the viewer draws a plain instance, with render flags 0x8000 and 0x800 clear.

`tests/test_rcra_viewer.py` draws the swarmer through the model viewport (Windows, `RCRA_GAME`) and checks that the translated programs link and draw. It also draws a pressure button whose material reads `TEXCOORD4.x`, and checks that its colour bytes reach the program: the frame changes when they are replaced by 1. The swarmer's materials read `TEXCOORD3`: its decoded tangent is uploaded, and the frame changes when the tangent is turned a quarter turn about the normal.

**Tangent checks.** On the swarmer, the decoded frame and the UV-derivative frame draw the same picture, apart from silhouette edges and the grille, where the per-pixel derivative frame is noisy. On the helper bot's `.blend`:
- the stored coordinates rebuild the decoded tangent at every corner within 2.4 × 10⁻⁷;
- a Cycles bake of the material's own `TEXCOORD3` agrees with it at a median of 4 × 10⁻⁴, and 99.9 % of corners are within 0.01;
- 13 corners (0.02 %) differ by more than 0.1:
  - 12 lie on four zero-area triangles, which the bake leaves black and which nothing draws.
  - One lies on a triangle of 7.5 × 10⁻⁶ square units whose corner tangents differ. The bake samples inside that triangle, not at its corner.
- Before the back-face fix, four more corners were flipped. They lie on two folded triangles, whose corner normals oppose their winding, so Cycles shades them as back faces.

### Not translated

- **Blended materials that read the scene behind in scene mode** (27 of 1,747): Blender gets their G-buffer surface, opaque.
- **Forward lighting in Blender:** Blender's lights shade the G-buffer surface, which assumes that the forward program lights the same terms as the deferred apply. The viewer uses the program's own key-light term ([Own key light](#own-key-light)). Only that term is used: the programs' local light lists, probes and reflections are not (the review rig has no light grid). The viewer draws the forward programs' own ambient light and probe reflections under the Shared HDRI lighting ([Own ambient light](#own-ambient-light)), with one review probe and no parallax; screen-space reflections and the scene's own probes are not drawn.
- **Lighting classes in Blender:** class 3 (hair) is lit with the class-0 BRDF there; classes 1, 2, 4 and 5 have their own groups, with the projections and fits above ([In Blender](#in-blender)); the viewer runs every class's own program ([Lighting classes](#lighting-classes)).
- **Skin:** the viewer does not draw the skin pass's screen-space blur, a seven-tap pass over a four-level pyramid of the diffuse light that other skin passes write. It does not draw the transmission through the shadow-map depth either (the model viewer has no shadow maps), which is the only path by which the skin profiles ([Skin profiles](#skin-profiles)) reach the light.
- **Opaque materials on forward techniques** (245 materials, 44 of them eyes; [Techniques](#techniques)): 232 draw from their forward program.
  - 73 draw with its diffuse coefficient as albedo, 36 of them eyes (`basic_eye`, the crowd eyeballs, `eye_colortint_random`).
  - 44 draw with an albedo and an unlit colour, 6 of them eyes (`eye_emissive`, `basic_eye_holo`, `basic_eye_dimensional`), the rest flares, arcs, glows, and the jellies and gels whose key and ambient light share one albedo.
  - 110 draw with an unlit colour alone: unlit holograms, tracers, glows and screen effects, the electricity arcs at the review exposure, the planet surfaces with the review key direction, and lit materials without a diffuse factor (`cy_solid_color`, `hero_solid_color_fresnel`).
  - 2 draw with the fallback for different diffuse coefficients: the amoeboid (its rim as the unlit colour) and a dirt debris chunk. The viewer adds the program's whole key-light term; Blender draws the unlit colour alone.
  - 3 draw black, as the game does: the discoball test material (its own constants are 0) and two unlit glass chunks whose colour is the key light's colour times the engine's default probe, both 0 under the profile.

  The viewer adds the key-light term of 125 of the lit ones; the other 35 have no term that reads the key light. Their reflections (screen-space, probes) are not drawn, as for every forward program, and Blender draws no specular for them (the stub's F0 is 0). The other 13 keep the black stub, with the reason in their limitations: twelve photo-mode filters and a distortion bubble, whose unlit colour reads the scene behind.
- **Instance effects:** a material the plain instance draw skips ([Material alpha](#material-alpha)) may still be drawn by a per-instance effect overlay or a raised instance alpha. Neither is decoded, so such a layer is not drawn.

## Visual effects

Holograms, thrusters, glows and other effects are not part of a model. A conduit attaches `.visualeffect` assets to a model's locators at run time. `nwo/games/rcra/vfx.py` decodes both files; the model view draws part of them ([Effects in the viewer](#effects-in-the-viewer)).

### Attachment

A conduit's `0xCEB30E68` section is a serialized object, which the guarded ALERT reader reads whole. `vfx.conduit_effects` returns its `kVFX` responses: each has the event (`ActivatedEvent`, or a script or animation event with a label such as `start_fx`), the effect (`FXPackage`), the locator (`AttachLocJoint`) and whether it starts or stops the effect.

- **Corpus:** all 1,070 conduits read. 530 attach effects, through 5,213 responses.
- **Example:** the weapons vendor's actor names `weapons_vendor_vfx.conduit`. On activation it starts `weapons_vendor_hover_01` at `igLoc_hover` and the holo logo and its projector at `igLoc_projection_vfx`. The model's decoded [locators](#locators) hold both.

### File layout

The asset is the 36-byte toc header and a DAT1 whose string pool holds "VisualEffect Built File" and the referenced paths. The loader (`0x14136F9A0`) divides each section's size by its record size (`vfx.STRIDES`):

| Section | Record | Contents |
| --- | --- | --- |
| `0xB86CF442` | 448 | Nodes (below) |
| `0xC3CC2AB5` | 448 | Emitters (below) |
| `0xD78E007C` | 16 | Curves: u32 hash, u32 first key, u32 key count, f32 (1 in the corpus). The keys of consecutive curves follow on, in every file |
| `0xB1F4C248` | 16 | Curve keys (below) |
| `0xAF650E9E` | 8 | Materials: u32 path offset, i32 −1 or the offset of an inline material's name (`materialnode: material_15, <graph path>`) |
| `0xB40E134B` | 12 | Assets: u32 path offset, u32, u32 type hash (`vfx.ASSET_TYPES`: model, visualeffect, conduit, actor) |
| `0x73CE6A3F` | 32 | One record, (1, 1, 1, 1) in the vendor's effects; meaning not traced |
| `0x3F03BE86` | 120 | Spawn shapes ([Spawn](#spawn)); an emitter's u32 +0x118 indexes them |
| `0xEFD8600D`, `0x01670690`, `0xECC0D7AF` (+ `0x4A868898` strings) | 16, 20, 24 | Not read |

**Corpus:** 2,985 effects, 20,490 nodes and 9,698 emitters. `vfx.decode_effect` reads all of them.

**Nodes:**

- **Render type:** byte +0x17A. The values follow the editor enum the exe reflects (`kSingleQuad` … `kSingleDecalTrail`), with its one asset pair split three ways: 0 SingleQuad, 1 MultiQuads, 2 SingleLight, 3 MultiLights, 4 SingleSwipe, 5 SingleTrail, 6 MultiTrails, 7 RadialBlur, 8 MultiSplines, 9/10 single and multi child effects, 11/12 models, 13/14 conduits or actors, 15 SunLensFlare, 16 ForceField, 17 SingleDecal, 18 MultiDecals, 19 SingleTube, 20 SingleDecalTrail.
- **Names 9 to 14:** from the asset each node references. The asset entry's type hash agrees on every node.
- **Asset ids:** +0x00 is the u64 id of the node's model or child asset, +0x08 its material and, for lights, +0x104 an IES profile texture (loaded with the tag "VFX IES Profile"; the game builds an IES material from it over +0x08). Each is the path CRC64 of an entry in the effect's own lists. For an inline material it is the CRC64 of the material's name. 6 node material ids match no entry. The game's unload routine (`0x1413703E0`) releases these ids by render type.
- **Emitter link:** a Multi node names its emitter by the u32 at +0x10, which equals the emitter's +0x100. Every Multi node has one, and no Single node does.
- **Transform:** a 3×3 matrix at +0x20, a position at +0x44 and a scale at +0xC4. The vendor's holo logo plane sits 0.725 above its locator at scale (2, 2, 1).
- **Draw order:** byte +0x17F. The draw routine shifts it into the sort key.
- **Curve slots:** +0x180, eight (u32, u32) pairs. The low 16 bits of the first word index the curves; the high 16 bits name a run-time override of the effect instance (0 for none), which multiplies, replaces or adds its vector (`0x14138D350`). The half at byte +6 is the slot's time rate ([Curve keys](#curve-keys)). Multi nodes leave them 0: their particles take the emitter's slots 12 to 19.
- **Facing:** byte +0x178, the reflected `kCamera`, `kLocal`, `kWorld`, `kBillboard` order. The facing switch (`0x14139E020`) gives a `kCamera` node the camera's own axes (it copies the camera matrix rows) and turns a `kBillboard` node toward the camera position, with the native up (0, 1, 0) as the up hint. `kLocal` keeps the parent's rotation without its scale.

**Emitters:** a 3×3 matrix at +0x00 (rows: the local axes), a position at +0x24, the name hash at +0x100, twenty curve slots at +0x120, and the spawn and motion fields of [Spawn](#spawn) (`vfx.EMITTER_FIELDS`).

**Curve slot names** (`vfx.EMITTER_CURVES`) come from the code that reads each slot: the spawn loop (`0x14137C7B0`) and the ten curve refs the particle update reads (built at `0x14137DC67`):

| Slot | Name | Read by |
| --- | --- | --- |
| 0 | size at spawn | update: a size factor, read at the particle's spawn time on the emitter's timeline |
| 1 | speed | spawn: its x scales the direction |
| 2 | burst | spawn: fires once each time the emitter time passes a key (`0x14138D570`) |
| 3 | per metre | spawn: particles per metre the emitter moved |
| 4 | rate | spawn: particles per second |
| 5 | gap | spawn: forces a particle when the last one is too far from the emitter (name inferred) |
| 6 | life | spawn: seconds, stored as a half at particle +0x3E |
| 7, 8 | emitter rotation, emitter position | the emitter's transform (degrees, offset) |
| 9 | shape scales | spawn: the vector handed to the shape sampler |
| 10 | follow | update: how far a particle follows its emitter |
| 11 | acceleration | update: times the acceleration direction (+0x30) |
| 12 | turbulence | update: noise amplitude |
| 13 | velocity | update: velocity multiplier |
| 14 to 19 | luminance, rotation, custom, scales, colour, alpha | update: the render values ([Draw values](#draw-values)) |

A Single node's slots 2 to 7 bind the same refs as emitter slots 14 to 19. A slot whose u16 at +4 is set reads the emitter's times instead of the particle's.

### Curve keys

- **Layout:** eight halves: base (x, y, z), a u16 key time, range (x, y, z) and a mode.
- **Key times:** 0x0000 to 0xFFFF map to 0 to 1, with thirds and eighths common.
- **Key value:** the key evaluator (`0x14138D0E0`) returns base + range × a random vector that the mode picks against two fixed thresholds, 1.9 and 2.9 (static initialisers at `0x14015A740`). Each vector holds one value for x, y and z, from a table of 255 uniform numbers in [0, 1) (`0x141393A20`):
  - mode 1: the particle's value u (table entry seed mod 255);
  - mode 2: a value seeded by the particle and the effect instance;
  - mode 3: a coin, 1 when u > 0.5, else 0.
  The corpus has 286,962 mode 1 keys, 30,566 mode 2 and 1,168 mode 3.
- **Between keys** (`0x14138D140`): the two keys around the time, interpolated linearly, times the header's |scale|. Scales run from 1 to 2,000 (rotations in degrees, rates). A negative scale makes the curve read the effect instance's times instead of the particle's (`0x14138D4C0`).
- **Time rate:** the half at slot byte +6 picks the time. 0: the normalised age, clamped to 1. Above 0: `frac(rate × seconds)`, a loop at `rate` cycles per second. Below 0: `min(|rate| × seconds, 1)`, played once over 1 / |rate| seconds. About 16,000 of the corpus's 280,000 slots have a rate.
- **In the reader:** `vfx.curve_value(curve, t, random, instance_random)` and `vfx.curve_time(rate, seconds, normalised)` follow these routines. The viewer evaluates with u = 0.5 (so a coin reads 0) and with its own still time ([Effects in the viewer](#effects-in-the-viewer)).

### Spawn

The spawn loop (`0x14137C7B0`) and particle init (`0x141395800`), traced:

- **Count per frame:** `carry + rate × dt`, the fraction carried over, plus slot 2's bursts, capped at 4,096 per frame, by +0x108 (particles spawned in total) and +0x11C (alive). Nothing spawns while the emitter's age is below 0: it starts at minus the spawn delay (+0x6C, plus a random part up to +0x70). Rate spawning stops when the emitter's lifetime ends (+0x64, plus up to +0x68; 0 is infinite), except for the on-death burst (+0x10C).
- **Time:** spawn-time curves read the emitter's age in seconds and its normalised age (0 for an infinite emitter).
- **New particle:** the shape sampler returns a local position and direction. Both go through the emitter's rows and position (slot 8 adds its offset). Velocity = direction × slot 1's x, plus +0x48 × the emitter's velocity. Life = slot 6. Initial rotation = random × +0x80 + +0x8C per axis (radians). Size base 1 (or the effect scale with flag +0x104 bit 0x400).
- **Particle space** (+0x115): 0 keeps particles in the emitter's frame; 1 and 2 are world.
- **Shapes** (section `0x3F03BE86`, 120 bytes, type hash at +0x0C). Every analytic shape emits along local +Z. The names (`vfx.SHAPE_KINDS`) are read from the sampling maths; the hashes match no plain hash of the reflected names (`kCone`, `kDisc`, `kBoxEx`, `Plane`):
  - `BD6A18A2` cone or sphere (the default): direction at an angle from +Z in [+0x38, +0x38 + +0x3C] degrees and a sweep in [+0x40, +0x40 + +0x44] + 180, position = direction × radius (+0x50 + random × +0x54).
  - `323F9580` disc, ring or cylinder; `D8E0E141` plane (half sizes +0x58, +0x5C); `D6F48302` box by faces; `22FC61F3` box along the cone direction; `CEC6BE18` a model's surface.
  - +0x48 + coin × +0x4C signs the direction (outward, inward or random), +0x68 spreads it (degrees per Euler angle), +0x6C to +0x74 offsets the origin.

### Update

The default particle update (`0x141391F00`), traced. Particles are 64 bytes:

| Bytes | Content |
| --- | --- |
| +0x00, +0x0C | position (3 f32), age (f32 seconds) |
| +0x10 | rotation, 3 s16 (32,768 = 2π) |
| +0x16 | size base, u16 (512 = 1) |
| +0x18, +0x1E, +0x1F | colour RGB (3 halves), alpha-test threshold and alpha (bytes /255) |
| +0x20, +0x28 | tail vector, velocity (3 halves each) |
| +0x2E, +0x37, +0x3E | spawn time (half), random byte, life (half) |
| +0x30, +0x38 | size xyz, custom xyz (3 halves each) |

- **Time:** t = min(age / life, 0.999999); a particle dies when its age passes its life.
- **Motion:** velocity += dt × the acceleration direction (+0x30) × slot 11, then × the drag factor exp(−dt × +0x54), plus a gravity step (0, −dt × +0x4C, 0) in world axes. Position += dt × velocity × slot 13. Wind (+0x50 × a run-time vector), turbulence (slot 12, when the record's flag 2 is set) and following (slot 10) are added on top.
- **Values:** colour = slot 18 × slot 14's x; alpha = slot 19's x, the threshold its y; custom = slot 16; size = base × slot 17 × slot 0, with random mirror flips; rotation integrates slot 15 as degrees per second, or reads it absolutely with node flag 0x800.

Other updates share the layout: frozen emitters copy the particle, the first step of a new particle runs at t 0, and swipes and trails (render types 4 to 6) run their own per-point update (`0x14139F240`).

### Draw values

- **Model nodes** (`0x141377EC0`): the particle's colour halves, custom halves and alpha byte go to the draw object, and the object writer (`0x1410FDC00`) puts them in `g_SceneObjectView`: colour RGB in `m_ColorRGFp16` and the low half of `m_ColorBMiscFp16`, alpha / 255 in `m_Alpha`, custom X, Y, Z in `m_CustomXYFp16` and `m_CustomZRandFp16`. Misc and Rand are not particle values: Misc comes from an aux table the draw does not index (0), Rand from a fixed table by draw byte +0x9C.
- **Quads** (render types 0 and 1) are expanded on the GPU: engine vertex programs in the exe read the 64-byte particles from `g_VisualEffectParticlesU4`, one particle per six vertices. They write TEXCOORD4 = (particle RGB × the object's custom X, Y, Z, 1), TEXCOORD.z = the alpha-test threshold, TEXCOORD.w = alpha × the object's Misc × a near fade, TEXCOORD6 = (max of the size × width and height, age / life, a random term, distance LOD) and TEXCOORD7 = (custom, FX lighting slot). The quad is |size x| × `m_Width` by |size y| × `m_Height` (`VfxMatInstanceCBuffer`), along `m_OrientA` (right) and `m_OrientB` (up), which the CPU fills per facing.
- **Lights** (render types 2 and 3, `0x14137A1E0`): type byte +0x17B (`kPyramidLight`, `kPointLight`, `kSpotLight`), range +0x130, spot cone +0x120 / +0x124 degrees (defaults 15 and 45). Colour = the particle's hue at luminance 1, intensity = max(|size x|, |size y|) × |RGB| × alpha, 0 unless node flag bit 0 is set. Node +0x10C to +0x11C and +0x128, +0x12C go into the light record with unknown meanings.

### Effects in the viewer

The view decode (`rcra_game.decode_model` with arrays) runs `nwo/games/rcra/effects.py`. The export decode does not.

- **Which effects:** the model's actors (reverse dag), their conduits, and each conduit's `ActivatedEvent` responses that start an effect. A conduit counts only when every locator its activation responses name is a locator or joint of the model. This skips shared conduits such as the gallery's `hero_gallery_vfx`, which names the locators of many models. Responses behind a filter that tests game state are left out: the shared grunthor conduit starts the skeleton look's eye glow and soul fire only when `kMissionName` is one of three missions. Identical (effect, locator) pairs from several actors attach once.
- **What draws:** SingleQuad, MultiQuads, SingleModel and MultiModels nodes, each as a mesh with its own material entry, weighted 1 to the locator's joint so it follows the animation. The effect is placed by the locator's rest transform (its joint's inverse bind, inverted, times the locator matrix), then the node's position and 3×3 (applied with the nine floats as rows; that order is not traced, and the vendor's nodes have the identity).
- **Particles:** a Multi node's particles come from `nwo/games/rcra/particles.py`, a replay of [Spawn](#spawn) and [Update](#update) to a still emitter age: one particle life after the first spawn (a steady state), half a life for an emitter that only bursts, within the emitter's lifetime. The replay steps at 1/60 s. In place of the game's random table it uses a fixed low-discrepancy sequence per particle, so spreads are even rather than those of a given game frame. It follows the cone, disc and plane shapes (box and model-surface shapes spawn at the origin, noted), acceleration, gravity and drag; not wind, turbulence, following, inherited velocity, per-metre spawning, random flips or slot 7's emitter rotation.
- **Single nodes:** their curves at mid-life (t 0.5), and slots with a time rate at 1 s.
- **Facing:** `kLocal` and `kWorld` quads lie in the effect's XY plane. `kCamera` and `kBillboard` nodes and particles store every vertex at their centre, with the offset in their own frame as a vertex stream (`billboard`, `loader._billboards`). The Rift Apart vertex program turns them toward the view after skinning (`rcra_glsl.BILLBOARD_VERTEX`, attribute 11): `kCamera` along the view's axes, `kBillboard` toward the eye with the native up as its up hint, as the facing switch does.
- **Values:** a model node binds its colour, alpha and custom values as per-object members, a quad node its interpolants as the particle vertex program writes them ([Draw values](#draw-values)), through `shaders.draw_override` on every translation of the material. One draw state stands for a node, so a Multi node takes the values of its particle nearest mid-life. The object custom and Misc factors of a quad, the near fade and the object Rand half are not traced: they read 1, 1, 1 and 0. Quad size = node scale × particle size, the reading of `m_Width` and `m_Height`, whose CPU writer is not traced.
- **Lights:** listed in `decoded['effects']` rows (`lights`: type, colour, intensity, range, cone), not drawn: the translated programs read the game's light lists, which the review profile binds empty.
- **Example:** the weapons vendor's holo logo plane draws 0.725 above `igLoc_projection_vfx` (head joint 13), 2 × 2 units, facing the view (`kBillboard`), with the logo program at object colour (0, 6.52, 10). Its projector draws a glow, two ray cones and 17 beam quads; her hover effect draws the thrust rings, glows and lit smoke. Its anamorphic lens flare quad (node scale 0.5 × 12) draws as a tall strip; whether the game swaps width and height for it is not traced.
- **Left out:** lights, trails, swipes, decals, lens flares, radial blur, splines, tubes, force fields and child effects, and nodes whose material is an inline graph. `decoded['effects']` lists, per attached effect, the nodes drawn and those left out, and the material note repeats them.

### Not decoded

- **Quad draw state:** which object record a quad draw binds, who sets its custom and Misc halves (the effect instance's tint at +0xEC is the likely source), and the CPU writer of `VfxMatInstanceCBuffer` (`m_OrientA` to `m_OrientC`, `m_Width`, `m_Height`).
- **Spawn:** the shape hash names, the difference between particle spaces 1 and 2, and the run-time globals (wind, the first-frame step).
- **Lights:** the light record fields with unknown meanings, and the IES profiles.
- **Trails and swipes:** their point layout and the 15 curves of their update.
- **Render targets:** effects that show a render target, such as the vendor's weapon hologram (`gbl_gp_vendorrtt_01`), cannot be drawn from the files.

## Evidence tool

```bash
python -m nwo.games.rcra.inspect --game <install> --output-root <folder> --summary --list model --verify
```

`--verify` decompresses every block of every installed archive once, and checks that every toc row lies inside its archive. `--extract <asset path>` writes an asset's raw bytes, plus a `.stream` file for a streamed texture.

## Tests

`tests/test_rcra_archive.py`, `tests/test_rcra_model.py`, `tests/test_rcra_texture.py`, `tests/test_rcra_glass.py` and `tests/test_rcra_animation.py` each have two parts:

- **Synthetic tests** build a small toc, dag and set of DSAR archives, a two-mesh model, texture and material assets, clips (plain and with a frame table) or skin batches (with a palette and a joint base).
- **Real-install tests** read `RCRA_GAME` (the install folder) and skip when it is not set. They also compare bytes against ALERT's own TOC2 reader.

`tests/test_rcra_model.py` also checks the tangent decode. Without game files it covers a round trip of the packed fields, the viewer payload (display basis, whole `TEXCOORD3.w`) and the glTF attribute. With `RCRA_GAME` it finds the engine model vertex programs in the exe and translates them, and checks their `TEXCOORD3` against the decoder on Ratchet, with and without render flag 0x10000. It also checks that the tangent follows Ratchet's UV frame.

`tests/test_rcra_vertex.py` checks the vertex-program path. Without game files, it checks the vectorised evaluator against the translator's operations bit for bit, the view check, the stored vertices as a glTF morph target, and the notes in the loader. With `RCRA_GAME` it checks that the swarmer's vertex-animation texture moves every mesh, with each vertex where the translator's own evaluator puts it. It also checks that a dimensional glitch is view-dependent and that a balloon keeps its stored vertices.

`tests/test_rcra_bake.py` checks the portable export's UV0 bake. Without game files, it checks the structural test (UV1, the normal, a cube map or volume block a group; the front-face flag does not), terms at the texel centres, sRGB decoding before filtering, bilinear wrap at a scaled UV, lookup tables, and that chunks and threads give the same values. With `RCRA_GAME` it checks that Ratchet's head material bakes to the scalar evaluator's values at random texels.

`tests/test_rcra_vfx.py` checks the [visual effect](#visual-effects) reader on a synthetic effect: asset, material (also inline) and emitter links, emitter fields and slot names, curve keys, times and values, curve scale, key modes and time rates, and failures on a broken curve chain or slot. It also checks the particle replay (count, travel, gravity), quads lying flat or facing the view, a facing model's offsets, the quad interpolants, light values, the per-object override and an effect mesh's placement and joint weights. With `RCRA_GAME` it checks the vendor's holo logo, its projector's emitter links, the conduit's attachments, and that the vendor's view decode carries the effect meshes (the logo facing the view on the head joint in every look and LOD, the thrust rings flat, camera-facing smoke, one ray cone per particle, the logo's spot light, the override in the logo's translation) while the export decode does not.

`tests/test_rcra_draw_skip.py` checks two things on synthetic flags blocks: the draw-skip rule (blended, additive, opaque, fur and distance-fade cases) and the packed draw alpha. It also checks the viewport and export wiring. With `RCRA_GAME` it checks that Ratchet's head lists only `compositeshell` as undrawn and that the texture entries mark its slot. It also checks that a glass of material alpha 0.3 gets forward alpha × 1228/4096. `tests/test_rcra_cube.py` checks the cube-map path without game files: the D3D cube lookup at texel centres, the equirectangular image against that lookup and Blender's mapping, the viewer's `samplerCube` emission, and (Windows OpenGL) that the viewport's GL cube map returns the same texels for the game direction. `tests/test_rcra_surface.py` checks the surface terms: packed again with the engine's formulas they give the writer's own G-buffer bits, on four real graphs. It also checks the texture writer and the export wiring, and, with Blender installed (`RC2016_BLENDER` or the default path), bakes an emitted material against its recipe. `tests/test_sm5.py` compiles small HLSL programs with Windows' `d3dcompiler_47` at test time, checks the decoder against `D3DDisassemble` and the translator against known results (branches, loops, switch, bit packing, discard, reflection names, UAV stores, raw stores and atomics, groupshared memory, domain-program inputs). It skips without the DLL. `tests/test_rcra_lighting.py` checks the lighting classes. Without game files it covers the class bits, the extra-field layout, the emitted `rcraLight` and `rcraClassLit`, and the forward programs' own light emitted per light. With `RCRA_GAME` it finds the six apply programs, checks class 0 against GGX with α = r⁴ and that every class recipe is the light's own term. It also checks the cloth and foliage writers' extra fields against their packed word, that the fur writer is class 3, the dome's own key light, the jellyfish domain program, and engine compute programs with groupshared memory. `tests/test_rcra_shaders.py` (`RCRA_GAME`) checks the pass table, the glass forward program against the hand decode on the trooper dome, and the `basic_normal_gloss` G-buffer writer.
