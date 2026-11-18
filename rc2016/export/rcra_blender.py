"""Rift Apart materials in Blender: the game's deferred BRDF as a node group, and node trees emitted from surface recipes.

Runs inside Blender only (``bpy``). A recipe (``rc2016.games.rcra.surface.export_recipe``) is the material's translated
G-buffer writer reduced to float expressions of its surface terms; ``apply_recipe`` turns every expression node into a
Math node, so the material is the game's own arithmetic, not a fitted approximation.

Inputs the recipe reads, in Blender:
- ``TEXCOORD.xy`` (UV0) and ``TEXCOORD1.xy`` (UV1): the mesh's first and second UV maps, with game v = 1 - Blender v.
  Each texture lookup flips back (images are stored top row first and Blender reads them bottom row first).
- ``TEXCOORD2.xyz`` (world normal): Geometry Normal, un-flipped on back faces as the game's interpolant is.
- ``TEXCOORD3.xyz`` and ``.w`` (tangent and TEXCOORD3.w): the decoded vertex tangent and its w
  (rc2016.games.rcra.model.unpack_tangents), exported as the glTF point attribute TANGENT_ATTRIBUTE (native xyz, w).
  ``store_tangent_frame`` rewrites it after import as the tangent's coordinates (a, b, c) in Blender's UV0 tangent frame
  at each corner, T = a t + b (n x t) + c n with t the MikkTSpace tangent and n the corner normal, and w; the
  out-of-surface part c is kept, since the game's tangent is not orthogonalised against its normal. The material rebuilds
  T from the Normal Map node's tangent and the geometry normal, so it turns with the armature as both do. A mesh without
  the attribute gets Blender's UV0 frame itself, bitangent along Blender's +v (the game's -v, the orientation of the stored
  tangents), labelled.
- ``TEXCOORD4`` (vertex colour): the mesh's first colour attribute, else 1.
- ``SV_IsFrontFace``: not Geometry Backfacing.
- ``VIEW.xyz`` and ``CAMERA.xyz`` (forward recipes): the offset from the camera to the shaded point, -Incoming x View
  Distance, and the camera position, Position + Incoming x View Distance, both converted to game space.

A cube map sample (``_Emitter.cube``) feeds the game direction, unconverted, to an Environment Texture on the equirectangular
image that rc2016.games.rcra.blender_export writes; the image's pixels are the D3D cube lookup of the direction Blender maps
to them, so the node returns the cube texel for the game direction.

Vectors convert between game space (Y-up) and Blender (Z-up, the glTF importer's turn): game (x, y, z) = Blender (x, z, -y).

BRDF group "RC RA Deferred BRDF", from the engine's class-0 deferred apply program (docs/rift-apart.md):
- **Specular:** GGX with Blender roughness r = 0.9725 - 0.7514 gloss (Blender alpha = r^2, the game's alpha). Metallic
  BSDF with F82 Fresnel and a white edge tint is Schlick with F90 = 1, as in the game. Cycles' Smith term is exact; the game
  uses the joint approximation.
- **Diffuse:** Lambert, plus the self-lit term albedo x E.
- **Light units:** the game's lights carry no 1/pi, so a Blender light's strength is pi x the game's colour, as for R&C 2016.
- **Occlusion:** the game applies it to ambient light only, which Cycles cannot separate. "Occlusion on diffuse" (default 0)
  multiplies the diffuse colour by it instead.

Blended materials (entry ``blend`` additive, blended or modulate) carry a forward recipe: the colour and alpha the game
blends, with every engine light source zero, and a background tint B where the program passes the scene behind it
through (colour = A + B x background). ``apply_recipe`` composes it with a Transparent BSDF by the mode, and for 'blended'
adds the G-buffer surface lit by Blender (its emission left to the forward colour, which already holds it). A lit
background term arrives folded into B at a unit light (rc2016.games.rcra.surface.unit_irradiance):
- blended: Transparent(mix(white, B, alpha)) + alpha x (BRDF + Emission(colour)), i.e. mix(Transparent, body, alpha)
  when B = 0
- additive: Transparent(1 + alpha B) + Emission(colour, strength alpha)
- modulate: Transparent tinted by mix(white, colour, alpha)
"""
import hashlib
import json
import math
from pathlib import Path

import bpy

BRDF_GROUP = 'RC RA Deferred BRDF'
GROUP_VERSION = 1
SURFACE_PREFIX = 'RC RA surface '
RECIPE_PROPERTY = 'rc_rcra_recipe_sha256'
FORWARD_PREFIX = 'RC RA forward '
FORWARD_PROPERTY = 'rc_rcra_forward_sha256'
BLEND_MODES = ('opaque', 'additive', 'blended', 'modulate')
TANGENT_ATTRIBUTE = '_RCRA_TANGENT'    # rc2016.export.models: point domain, native tangent xyz and TEXCOORD3.w
TANGENT_FRAME = 'rcra_tangent_frame'   # corner domain: the tangent's coordinates in the UV0 tangent frame
TANGENT_W_ATTRIBUTE = 'rcra_tangent_w'   # point domain: TEXCOORD3.w
NO_TANGENT_NOTE = ('This mesh has no decoded tangent attribute: the tangent frame is Blender\'s UV0 tangent space and the '
                   'bitangent sign its +v side (the game\'s -v).')
# Blender's node creation slows as a tree grows (3,000 nodes in one tree take about 35 s, in groups of 100 about 0.6 s), so a
# large recipe is emitted into chained part groups of at most PART_SIZE nodes; values cross parts through group sockets.
PART_ABOVE = 400   # recipes with more operations are split
PART_SIZE = 150
SOCKET_KINDS = {'VALUE': 'NodeSocketFloat', 'VECTOR': 'NodeSocketVector', 'RGBA': 'NodeSocketColor'}
LEAF_OPS = ('const', 'input', 'channel', 'ddx', 'ddy')   # recipe nodes _Emitter.value builds without its arguments
MATH = {'add': 'ADD', 'mul': 'MULTIPLY', 'div': 'DIVIDE', 'min': 'MINIMUM', 'max': 'MAXIMUM', 'abs': 'ABSOLUTE',
        'sqrt': 'SQRT', 'rsq': 'INVERSE_SQRT', 'frc': 'FRACT', 'round_ne': 'ROUND', 'round_ni': 'FLOOR',
        'round_pi': 'CEIL', 'round_z': 'TRUNC', 'sin': 'SINE', 'cos': 'COSINE', 'lt': 'LESS_THAN'}


def recipe_sha256(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# --------------------------------------------------------------------------------------------------- BRDF group

def _socket(tree, name, in_out, kind, default=None):
    socket = tree.interface.new_socket(name=name, in_out=in_out, socket_type=kind)
    if default is not None:
        socket.default_value = default
    return socket


def build_brdf_group():
    """The shared "RC RA Deferred BRDF" group (rebuilt when its version changes)."""
    group = bpy.data.node_groups.get(BRDF_GROUP)
    if group is not None and group.get('rc_version') == GROUP_VERSION:
        return group
    if group is None:
        group = bpy.data.node_groups.new(BRDF_GROUP, 'ShaderNodeTree')
    group.nodes.clear()
    group.interface.clear()
    group['rc_version'] = GROUP_VERSION
    _socket(group, 'Albedo', 'INPUT', 'NodeSocketColor', (0.5, 0.5, 0.5, 1.0))
    _socket(group, 'F0', 'INPUT', 'NodeSocketColor', (0.04, 0.04, 0.04, 1.0))
    _socket(group, 'Gloss', 'INPUT', 'NodeSocketFloat', 0.5)
    _socket(group, 'Normal', 'INPUT', 'NodeSocketVector')
    _socket(group, 'Occlusion', 'INPUT', 'NodeSocketFloat', 1.0)
    _socket(group, 'Emission', 'INPUT', 'NodeSocketFloat', 0.0)
    _socket(group, 'Occlusion on diffuse', 'INPUT', 'NodeSocketFloat', 0.0)
    _socket(group, 'Shader', 'OUTPUT', 'NodeSocketShader')
    nodes, links = group.nodes, group.links
    inp, out = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')
    rough = nodes.new('ShaderNodeMath')
    rough.operation = 'MULTIPLY_ADD'
    rough.name = 'Roughness 0.9725 - 0.7514 gloss'
    links.new(inp.outputs['Gloss'], rough.inputs[0])
    rough.inputs[1].default_value = -0.7514
    rough.inputs[2].default_value = 0.9725
    spec = nodes.new('ShaderNodeBsdfMetallic')
    spec.name = 'Specular GGX Schlick F90 1'
    spec.distribution = 'GGX'
    spec.fresnel_type = 'F82'
    links.new(inp.outputs['F0'], spec.inputs['Base Color'])
    spec.inputs['Edge Tint'].default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(rough.outputs[0], spec.inputs['Roughness'])
    links.new(inp.outputs['Normal'], spec.inputs['Normal'])
    # Diffuse colour x (1 + f (occlusion - 1)), f = "Occlusion on diffuse".
    occ = nodes.new('ShaderNodeMath')
    occ.operation = 'SUBTRACT'
    links.new(inp.outputs['Occlusion'], occ.inputs[0])
    occ.inputs[1].default_value = 1.0
    scale = nodes.new('ShaderNodeMath')
    scale.operation = 'MULTIPLY_ADD'
    links.new(occ.outputs[0], scale.inputs[0])
    links.new(inp.outputs['Occlusion on diffuse'], scale.inputs[1])
    scale.inputs[2].default_value = 1.0
    tint = nodes.new('ShaderNodeVectorMath')
    tint.operation = 'SCALE'
    links.new(inp.outputs['Albedo'], tint.inputs[0])
    links.new(scale.outputs[0], tint.inputs['Scale'])
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.name = 'Diffuse Lambert'
    links.new(tint.outputs[0], diffuse.inputs['Color'])
    diffuse.inputs['Roughness'].default_value = 0.0
    links.new(inp.outputs['Normal'], diffuse.inputs['Normal'])
    emission = nodes.new('ShaderNodeEmission')
    emission.name = 'Self-lit albedo x E'
    links.new(inp.outputs['Albedo'], emission.inputs['Color'])
    links.new(inp.outputs['Emission'], emission.inputs['Strength'])
    add = nodes.new('ShaderNodeAddShader')
    links.new(diffuse.outputs[0], add.inputs[0])
    links.new(spec.outputs[0], add.inputs[1])
    total = nodes.new('ShaderNodeAddShader')
    links.new(add.outputs[0], total.inputs[0])
    links.new(emission.outputs[0], total.inputs[1])
    links.new(total.outputs[0], out.inputs['Shader'])
    for x, node in enumerate((inp, rough, spec, occ, scale, tint, diffuse, emission, add, total, out)):
        node.location = (x * 220, 0)
    return group


# ----------------------------------------------------------------------------------------------- surface emitter

class _Emitter:
    def __init__(self, tree, recipe, images, uv_maps, colour_attribute, interpolation='Linear', fixed=None,
                 tangent=False):
        self.fixed = fixed or {}   # input name -> float, replacing the mesh/camera value (checks)
        self.tangent = tangent     # the mesh carries TANGENT_FRAME and TANGENT_W_ATTRIBUTE
        self.root = self.tree = tree
        self.nodes, self.links = tree.nodes, tree.links
        self.parts, self.carried, self.sockets = {}, {}, 0
        self.split = len(recipe['nodes']) > PART_ABOVE
        self.recipe, self.images = recipe, images
        self.uv_maps, self.colour_attribute = uv_maps, colour_attribute
        self.interpolation = interpolation
        self.memo, self.inputs, self.samples, self.tables, self.specials = {}, {}, {}, {}, {}
        self.count = 0
        if self.split:
            self.new_part()

    def node(self, kind, **props):
        n = self.nodes.new(kind)
        for key, value in props.items():
            setattr(n, key, value)
        n.location = ((self.count // 40) * 200, -(self.count % 40) * 60)
        self.count += 1
        return n

    def new_part(self):
        """Continue emitting in a new part group, chained in the root tree."""
        tree = bpy.data.node_groups.new('%s part %d' % (self.root.name, len(self.parts) + 1), 'ShaderNodeTree')
        gin, gout = tree.nodes.new('NodeGroupInput'), tree.nodes.new('NodeGroupOutput')
        node = self.root.nodes.new('ShaderNodeGroup')
        node.node_tree, node.name = tree, 'Part %d' % (len(self.parts) + 1)
        node.location = (len(self.parts) * 220, 400)
        self.parts[tree.name] = dict(tree=tree, node=node, gin=gin, gout=gout)
        self.tree, self.nodes, self.links, self.count = tree, tree.nodes, tree.links, 0

    def link(self, src, dst):
        """Link an output socket to an input socket, carrying it across part groups when they are in different trees."""
        tree = dst.id_data
        if src.id_data != tree:
            src = self.carry(src, tree)
        tree.links.new(src, dst)

    def carry(self, src, tree):
        """``src`` as a socket in ``tree``: out of its part through the part's Group Output, then into ``tree``'s part through
        its Group Input (the root links the two group nodes)."""
        key = (src.as_pointer(), tree.name)
        if key in self.carried:
            return self.carried[key]
        kind = SOCKET_KINDS.get(src.type, 'NodeSocketFloat')
        root_key = (src.as_pointer(), self.root.name)
        if src.id_data == self.root:
            at_root = src
        elif root_key in self.carried:
            at_root = self.carried[root_key]
        else:
            part = self.parts[src.id_data.name]
            self.sockets += 1
            name = 'v%d' % self.sockets
            part['tree'].interface.new_socket(name, in_out='OUTPUT', socket_type=kind)
            part['tree'].links.new(src, part['gout'].inputs[name])
            at_root = self.carried[root_key] = part['node'].outputs[name]
        if tree == self.root:
            return at_root
        part = self.parts[tree.name]
        self.sockets += 1
        name = 'v%d' % self.sockets
        part['tree'].interface.new_socket(name, in_out='INPUT', socket_type=kind)
        self.root.links.new(at_root, part['node'].inputs[name])
        self.carried[key] = part['gin'].outputs[name]
        return self.carried[key]

    def math(self, operation, *args, clamp=False):
        n = self.node('ShaderNodeMath', operation=operation, use_clamp=clamp)
        for k, a in enumerate(args):
            self.feed(a, n.inputs[k])
        return n.outputs[0]

    def feed(self, value, socket):
        if isinstance(value, (int, float)) and math.isfinite(value):
            socket.default_value = float(value)
        elif isinstance(value, (int, float)):   # a socket default does not keep inf or NaN; D3D arithmetic does
            self.link(self.non_finite(value), socket)
        else:
            self.link(value, socket)

    def non_finite(self, value):
        """An output holding inf, -inf or NaN: Exponent(10000) overflows float32 to +inf in Cycles and EEVEE, and NaN is
        inf - inf (Cycles folds x * 0 to 0, so inf * 0 would not stay NaN)."""
        key = 'nan' if value != value else ('+inf' if value > 0 else '-inf')
        if key not in self.specials:
            inf = self.math('EXPONENT', 10000.0)
            self.specials[key] = (inf if key == '+inf' else self.math('SUBTRACT', inf, inf) if key == 'nan' else
                                  self.math('MULTIPLY', inf, -1.0))
        return self.specials[key]

    # A value is a float or an output socket. Constants stay floats so they become socket defaults.
    def value(self, index):
        if index in self.memo:
            return self.memo[index]
        nodes = self.recipe['nodes']
        # Operation arguments first, in post-order without recursion: folded case chains run thousands of selects deep.
        stack, order, seen = [(index, False)], [], set()
        while stack:
            i, done = stack.pop()
            if done:
                order.append(i)
            elif i not in seen and i not in self.memo and nodes[i]['op'] == 'channel':
                seen.add(i)   # a texture's address is built before the sample, so parts only ever feed later parts
                stack.extend((x, False) for x in nodes[nodes[i]['args'][0]]['args'])
            elif i not in seen and i not in self.memo and nodes[i]['op'] not in LEAF_OPS:
                seen.add(i)   # shared subexpressions are walked once
                stack.append((i, True))
                stack.extend((x, False) for x in nodes[i]['args'])
        for i in order:
            if i not in self.memo:
                if self.split and self.count >= PART_SIZE:
                    self.new_part()
                self._value(i)
        return self._value(index)

    def _value(self, index):
        if index in self.memo:
            return self.memo[index]
        n = self.recipe['nodes'][index]
        op = n['op']
        if op == 'const':
            v = float(n['value']) if n['type'] != 'b' else (1.0 if n['value'] else 0.0)
        elif op == 'input':
            v = self.input(n['name'])
        elif op == 'channel':
            v = self.sample(n['args'][0])[n['index']]
        elif op in ('ddx', 'ddy'):
            v = 0.0   # derivatives of a constant-rate surface; the game's screen-space derivative is not available
        else:
            a = [self._value(x) for x in n['args']]
            v = self.lookup(n, *a) if op == 'lookup' else self.operation(op, a)
        self.memo[index] = v
        return v

    def operation(self, op, a):
        if op == 'div' and isinstance(a[1], float) and a[1] == 0.0:
            # Blender's Divide returns 0 for a zero divisor; D3D gives x * inf (sign(x) inf, NaN for 0). A neutral profile
            # value of 0 leaves such divisors. Exponent(10000) overflows float32 to +inf in Cycles and EEVEE.
            return self.math('MULTIPLY', a[0], self.math('EXPONENT', 10000.0))
        if op in MATH:
            return self.math(MATH[op], *a)
        if op == 'neg':
            return self.math('MULTIPLY', a[0], -1.0)
        if op == 'sat':   # D3D saturate takes NaN to 0; a clamped Add passes it on, Maximum (fmaxf) does not
            return self.math('MINIMUM', self.math('MAXIMUM', a[0], 0.0), 1.0)
        if op == 'rcp':
            return self.math('DIVIDE', 1.0, a[0])
        if op == 'log2':
            return self.math('LOGARITHM', a[0], 2.0)
        if op == 'exp2':
            return self.math('POWER', 2.0, a[0])
        if op == 'ge':
            return self.math('SUBTRACT', 1.0, self.math('LESS_THAN', a[0], a[1]))
        if op in ('eq', 'ne'):
            n = self.node('ShaderNodeMath', operation='COMPARE')
            self.feed(a[0], n.inputs[0])
            self.feed(a[1], n.inputs[1])
            n.inputs[2].default_value = 0.0
            return n.outputs[0] if op == 'eq' else self.math('SUBTRACT', 1.0, n.outputs[0])
        if op == 'b_and':
            return self.math('MULTIPLY', a[0], a[1])
        if op == 'b_or':
            return self.math('MAXIMUM', a[0], a[1])
        if op == 'b_not':
            return self.math('SUBTRACT', 1.0, a[0])
        if op == 'select':   # b + c (a - b), c in {0, 1}
            c, x, y = a
            return self.math('MULTIPLY_ADD', c, self.math('SUBTRACT', x, y), y)
        raise ValueError('Recipe operation not emitted: ' + op)

    # -- inputs
    def geometry(self):
        if 'geometry' not in self.inputs:
            self.inputs['geometry'] = self.node('ShaderNodeNewGeometry')
        return self.inputs['geometry']

    def game_vector(self, socket):
        """A Blender world vector as game (x, y, z) = Blender (x, z, -y)."""
        sep = self.node('ShaderNodeSeparateXYZ')
        self.link(socket, sep.inputs[0])
        return [sep.outputs[0], sep.outputs[2], self.math('MULTIPLY', sep.outputs[1], -1.0)]

    def frame(self):
        if 'frame' in self.inputs:
            return self.inputs['frame']
        g = self.geometry()
        # Cycles turns the shading normal towards the viewer on back faces; the game's interpolant is not turned.
        facing = self.math('MULTIPLY_ADD', g.outputs['Backfacing'], -2.0, 1.0)
        normal = self.node('ShaderNodeVectorMath', operation='SCALE')
        self.link(g.outputs['Normal'], normal.inputs[0])
        self.link(facing, normal.inputs['Scale'])
        probes = {}
        for name, colour in (('tangent', (1.0, 0.5, 0.5, 1.0)), ('bitangent', (0.5, 1.0, 0.5, 1.0))):
            if name == 'bitangent' and self.tangent:
                continue
            probe = self.node('ShaderNodeNormalMap', space='TANGENT', uv_map=self.uv_maps[0] if self.uv_maps else '')
            probe.inputs['Color'].default_value = colour
            probe.inputs['Strength'].default_value = 1.0
            # The Normal Map node negates its output on back faces too (Cycles svm_node_normal_map, EEVEE
            # node_normal_map): un-flipped with the normal, so the tangent is the mesh's on both sides.
            unflip = self.node('ShaderNodeVectorMath', operation='SCALE')
            self.link(probe.outputs[0], unflip.inputs[0])
            self.link(facing, unflip.inputs['Scale'])
            probes[name] = unflip.outputs[0]
        cross = self.node('ShaderNodeVectorMath', operation='CROSS_PRODUCT')
        self.link(normal.outputs[0], cross.inputs[0])
        self.link(probes['tangent'], cross.inputs[1])
        if self.tangent:   # T = a t + b (n x t) + c n from the stored coordinates (store_tangent_frame); w as stored
            coords = self.node('ShaderNodeSeparateXYZ')
            self.link(self.node('ShaderNodeAttribute', attribute_name=TANGENT_FRAME).outputs['Vector'], coords.inputs[0])
            tangent = None
            for k, axis in enumerate((probes['tangent'], cross.outputs[0], normal.outputs[0])):
                part = self.node('ShaderNodeVectorMath', operation='SCALE')
                self.link(axis, part.inputs[0])
                self.link(coords.outputs[k], part.inputs['Scale'])
                if tangent is None:
                    tangent = part.outputs[0]
                    continue
                total = self.node('ShaderNodeVectorMath', operation='ADD')
                self.link(tangent, total.inputs[0])
                self.link(part.outputs[0], total.inputs[1])
                tangent = total.outputs[0]
            sign = self.node('ShaderNodeAttribute', attribute_name=TANGENT_W_ATTRIBUTE).outputs['Fac']
        else:
            dot = self.node('ShaderNodeVectorMath', operation='DOT_PRODUCT')
            self.link(cross.outputs[0], dot.inputs[0])
            self.link(probes['bitangent'], dot.inputs[1])
            tangent = probes['tangent']
            sign = self.math('MULTIPLY_ADD', self.math('LESS_THAN', dot.outputs['Value'], 0.0), -2.0, 1.0)
        self.inputs['frame'] = dict(normal=self.game_vector(normal.outputs[0]), tangent=self.game_vector(tangent),
                                    sign=sign)
        return self.inputs['frame']

    def uv(self, layer):
        key = 'uv%d' % layer
        if key not in self.inputs:
            if layer >= len(self.uv_maps):
                self.inputs[key] = [0.0, 0.0]
            else:
                uv = self.node('ShaderNodeUVMap', uv_map=self.uv_maps[layer])
                sep = self.node('ShaderNodeSeparateXYZ')
                self.link(uv.outputs['UV'], sep.inputs[0])
                self.inputs[key] = [sep.outputs[0], self.math('SUBTRACT', 1.0, sep.outputs[1])]
        return self.inputs[key]

    def colour(self):
        if 'colour' not in self.inputs:
            if self.colour_attribute:
                attr = self.node('ShaderNodeVertexColor', layer_name=self.colour_attribute)
                sep = self.node('ShaderNodeSeparateColor')
                self.link(attr.outputs['Color'], sep.inputs[0])
                self.inputs['colour'] = [sep.outputs[0], sep.outputs[1], sep.outputs[2], attr.outputs['Alpha']]
            else:
                self.inputs['colour'] = [1.0, 1.0, 1.0, 1.0]
        return self.inputs['colour']

    def view(self):
        """(camera-to-point offset, camera position) as game vectors."""
        if 'view' not in self.inputs:
            g = self.geometry()
            camera = self.node('ShaderNodeCameraData')
            towards = self.node('ShaderNodeVectorMath', operation='SCALE')
            self.link(g.outputs['Incoming'], towards.inputs[0])
            self.link(camera.outputs['View Distance'], towards.inputs['Scale'])
            offset = self.node('ShaderNodeVectorMath', operation='SCALE')
            self.link(towards.outputs[0], offset.inputs[0])
            offset.inputs['Scale'].default_value = -1.0
            eye = self.node('ShaderNodeVectorMath', operation='ADD')
            self.link(g.outputs['Position'], eye.inputs[0])
            self.link(towards.outputs[0], eye.inputs[1])
            self.inputs['view'] = (self.game_vector(offset.outputs[0]), self.game_vector(eye.outputs[0]))
        return self.inputs['view']

    def input(self, name):
        axis = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
        if name in self.fixed:
            return float(self.fixed[name])
        if name in ('VIEW.x', 'VIEW.y', 'VIEW.z'):
            return self.view()[0][axis[name[-1]]]
        if name in ('CAMERA.x', 'CAMERA.y', 'CAMERA.z'):
            return self.view()[1][axis[name[-1]]]
        if name == 'SV_IsFrontFace':
            return self.math('SUBTRACT', 1.0, self.geometry().outputs['Backfacing'])
        semantic, _, component = name.partition('.')
        c = axis.get(component)
        if semantic == 'TEXCOORD' and c in (0, 1):
            return self.uv(0)[c]
        if semantic == 'TEXCOORD1' and c in (0, 1):
            return self.uv(1)[c]
        if semantic == 'TEXCOORD2' and c in (0, 1, 2):
            return self.frame()['normal'][c]
        if semantic == 'TEXCOORD3':
            return self.frame()['sign'] if c == 3 else self.frame()['tangent'][c]
        if semantic == 'TEXCOORD4':
            return self.colour()[c]
        raise ValueError('Recipe input not available in Blender: ' + name)

    # -- textures
    def lookup(self, n, x, rest):
        """A table lookup (surface.lookup_tables): table[X - low] where X is an integer in the table and the entry is present,
        else rest. The table is a one-row float32 image (R value, G present), read at the texel centre with Closest."""
        width = len(n['table'])
        key = (n['low'], tuple(n['table']), tuple(n['present']))
        if key not in self.tables:
            image = bpy.data.images.new('RC RA lookup %d' % len(self.tables), width=width, height=1, float_buffer=True,
                                        alpha=True)
            image.colorspace_settings.name = 'Non-Color'
            image.use_half_precision = False
            pixels = []
            for value, present in zip(n['table'], n['present']):
                pixels += [value, 1.0 if present else 0.0, 0.0, 1.0]
            image.pixels.foreach_set(pixels)
            image.file_format = 'OPEN_EXR'   # packed as float32 EXR: the values survive saving exactly
            image.pack()
            self.tables[key] = image
        if isinstance(x, float):   # a constant index: the chain's own answer
            k = x - n['low']
            inside = x == int(x) and 0 <= k < width and n['present'][int(k)]
            return float(n['table'][int(k)]) if inside else rest
        index = self.math('SUBTRACT', x, float(n['low']))
        combine = self.node('ShaderNodeCombineXYZ')
        self.feed(self.math('DIVIDE', self.math('ADD', index, 0.5), float(width)), combine.inputs[0])
        combine.inputs[1].default_value = 0.5
        tex = self.node('ShaderNodeTexImage', image=self.tables[key], interpolation='Closest', extension='EXTEND')
        tex.name = 'Lookup %d' % n['low']
        self.link(combine.outputs[0], tex.inputs['Vector'])
        sep = self.node('ShaderNodeSeparateColor')
        self.link(tex.outputs['Color'], sep.inputs[0])
        whole = self.node('ShaderNodeMath', operation='COMPARE')   # X is an integer: eq(X, key) can hold
        self.link(x, whole.inputs[0])
        self.feed(self.math('ROUND', x), whole.inputs[1])
        whole.inputs[2].default_value = 0.0
        inside = self.math('MULTIPLY', self.math('GREATER_THAN', index, -0.5), self.math('LESS_THAN', index, width - 0.5))
        hit = self.math('MULTIPLY', self.math('MULTIPLY', whole.outputs[0], inside), sep.outputs[1])
        return self.operation('select', [hit, sep.outputs[0], rest])

    def sample(self, index):
        if index in self.samples:
            return self.samples[index]
        n = self.recipe['nodes'][index]
        image = self.images.get(n['binding'])
        if image is None:
            raise ValueError('No image for ' + n['binding'])
        if isinstance(image, Cube):
            self.samples[index] = self.cube(image, n)
            return self.samples[index]
        u, v = self.value(n['args'][0]), self.value(n['args'][1])
        if isinstance(image, Volume):
            self.samples[index] = self.volume(image, u, v, self.value(n['args'][2]), n['binding'])
            return self.samples[index]
        combine = self.node('ShaderNodeCombineXYZ')
        self.feed(u, combine.inputs[0])
        self.feed(self.math('SUBTRACT', 1.0, v) if not isinstance(v, float) else 1.0 - v, combine.inputs[1])

        def lookup(img, label):
            tex = self.node('ShaderNodeTexImage', image=img, interpolation=self.interpolation, extension='REPEAT')
            tex.name = label
            self.link(combine.outputs[0], tex.inputs['Vector'])
            sep = self.node('ShaderNodeSeparateColor')
            self.link(tex.outputs['Color'], sep.inputs[0])
            return [sep.outputs[0], sep.outputs[1], sep.outputs[2], tex.outputs['Alpha']]
        if not isinstance(image, list):
            self.samples[index] = lookup(image, 'Unit %s' % n['binding'])
            return self.samples[index]
        # Array slice round(z), clamped to the array, picks one slice's lookup.
        last = float(len(image) - 1)
        z = self.math('MINIMUM', self.math('MAXIMUM', self.math('ROUND', self.value(n['args'][2])), 0.0), last)
        channels = [0.0, 0.0, 0.0, 0.0]
        for k, img in enumerate(image):
            pick = self.node('ShaderNodeMath', operation='COMPARE')
            self.link(z, pick.inputs[0])
            pick.inputs[1].default_value = float(k)
            pick.inputs[2].default_value = 0.25
            looked = lookup(img, 'Unit %s slice %d' % (n['binding'], k))
            channels = [self.math('MULTIPLY_ADD', pick.outputs[0], c, total) if k else
                        self.math('MULTIPLY', pick.outputs[0], c) for c, total in zip(looked, channels)]
        self.samples[index] = channels
        return channels


    def cube(self, cube, n):
        """A cube map lookup along the game direction (the sample's first three address lanes) as the Vector of an Environment
        Texture on the equirectangular image: u = 0.5 - atan2(y, x) / 2 pi, v = 0.5 + atan2(z, hypot(x, y)) / pi, the mapping
        rc2016.games.rcra.blender_export.equirect_directions inverts. Alpha comes from the second image, or is 1."""
        direction = self.node('ShaderNodeCombineXYZ')
        for k in range(3):
            self.feed(self.value(n['args'][k]), direction.inputs[k])

        def look(image, label):
            tex = self.node('ShaderNodeTexEnvironment', image=image, interpolation=self.interpolation)
            tex.name = label
            self.link(direction.outputs[0], tex.inputs['Vector'])
            sep = self.node('ShaderNodeSeparateColor')
            self.link(tex.outputs['Color'], sep.inputs[0])
            return sep.outputs
        rgb = look(cube.image, 'Unit %s cube' % n['binding'])
        alpha = look(cube.alpha, 'Unit %s cube alpha' % n['binding'])[0] if cube.alpha is not None else 1.0
        return [rgb[0], rgb[1], rgb[2], alpha]

    def volume(self, vol, u, v, z, binding):
        """Trilinear wrap lookup in a padded atlas: bilinear in the slices floor(s) and floor(s) + 1 (mod depth), s = z depth
        - 0.5, mixed by frac(s)."""
        d, cols, w, h = float(vol.depth), vol.columns, float(vol.width), float(vol.height)
        rows = -(-vol.depth // cols)
        aw, ah = cols * (w + 2.0), rows * (h + 2.0)
        s = self.math('SUBTRACT', self.math('MULTIPLY', z, d), 0.5)
        k0 = self.math('FLOOR', s)
        f = self.math('SUBTRACT', s, k0)
        fu, fv = self.math('FRACT', u), self.math('FRACT', v)
        looked = []
        for step in (0.0, 1.0):
            k = self.math('FLOORED_MODULO', self.math('ADD', k0, step), d)
            row = self.math('FLOOR', self.math('DIVIDE', k, float(cols)))
            col = self.math('SUBTRACT', k, self.math('MULTIPLY', row, float(cols)))
            x = self.math('DIVIDE', self.math('ADD', self.math('MULTIPLY', col, w + 2.0), self.math('MULTIPLY_ADD', fu, w, 1.0)), aw)
            y = self.math('DIVIDE', self.math('ADD', self.math('MULTIPLY', row, h + 2.0), self.math('MULTIPLY_ADD', fv, h, 1.0)), ah)
            combine = self.node('ShaderNodeCombineXYZ')
            self.feed(x, combine.inputs[0])
            self.feed(self.math('SUBTRACT', 1.0, y), combine.inputs[1])   # atlas rows are stored top first
            tex = self.node('ShaderNodeTexImage', image=vol.image, interpolation=self.interpolation, extension='CLIP')
            tex.name = 'Unit %s volume %d' % (binding, int(step))
            self.link(combine.outputs[0], tex.inputs['Vector'])
            sep = self.node('ShaderNodeSeparateColor')
            self.link(tex.outputs['Color'], sep.inputs[0])
            looked.append([sep.outputs[0], sep.outputs[1], sep.outputs[2], tex.outputs['Alpha']])
        return [self.math('MULTIPLY_ADD', f, self.math('SUBTRACT', b, a), a) for a, b in zip(*looked)]


SURFACE_OUTPUTS = (('Albedo', 'NodeSocketColor'), ('F0', 'NodeSocketColor'), ('Gloss', 'NodeSocketFloat'),
                   ('Normal', 'NodeSocketVector'), ('Occlusion', 'NodeSocketFloat'), ('Emission', 'NodeSocketFloat'),
                   ('Alpha', 'NodeSocketFloat'))


def build_surface_group(recipe, images, uv_maps, colour_attribute=None, interpolation='Linear', fixed=None, tangent=False):
    """A node group computing the recipe's surface terms (Albedo, F0, Gloss, Normal, Occlusion, Emission, Alpha)."""
    digest = recipe_sha256(recipe)
    name = SURFACE_PREFIX + digest[:16]
    group = bpy.data.node_groups.new(name, 'ShaderNodeTree')
    group[RECIPE_PROPERTY] = digest
    for socket, kind in SURFACE_OUTPUTS:
        _socket(group, socket, 'OUTPUT', kind)
    out = group.nodes.new('NodeGroupOutput')
    em = _Emitter(group, recipe, images, uv_maps, colour_attribute, interpolation, fixed, tangent)
    terms = {k: em.value(i) for k, i in recipe['terms'].items()}

    def combine(names, kind):
        node = em.node(kind)
        for k, n in enumerate(names):
            em.feed(terms[n], node.inputs[k])
        return node.outputs[0]
    em.link(combine(('albedo.r', 'albedo.g', 'albedo.b'), 'ShaderNodeCombineColor'), out.inputs['Albedo'])
    em.link(combine(('f0.r', 'f0.g', 'f0.b'), 'ShaderNodeCombineColor'), out.inputs['F0'])
    # Game normal (x, y, z) -> Blender (x, -z, y), normalised.
    normal = em.node('ShaderNodeCombineXYZ')
    em.feed(terms['normal.x'], normal.inputs[0])
    em.feed(em.math('MULTIPLY', terms['normal.z'], -1.0) if not isinstance(terms['normal.z'], float)
            else -terms['normal.z'], normal.inputs[1])
    em.feed(terms['normal.y'], normal.inputs[2])
    unit = em.node('ShaderNodeVectorMath', operation='NORMALIZE')
    em.link(normal.outputs[0], unit.inputs[0])
    em.link(unit.outputs[0], out.inputs['Normal'])
    for socket, term in (('Gloss', 'gloss'), ('Occlusion', 'occlusion'), ('Emission', 'emission')):
        em.feed(terms[term], out.inputs[socket])
    clip = terms['clip']
    keep = em.math('SUBTRACT', 1.0, clip) if not isinstance(clip, float) else 1.0 - clip
    if 'coverage' in terms:   # the fraction of the surface the program keeps (fur shells): alpha, cut by the discard
        cover = terms['coverage']
        keep = cover * keep if isinstance(cover, float) and isinstance(keep, float) else em.math('MULTIPLY', cover, keep)
    em.feed(keep, out.inputs['Alpha'])
    out.location = ((em.count // 40 + 1) * 200, 0)
    return group


FORWARD_OUTPUTS = (('Colour', 'NodeSocketColor'), ('Alpha', 'NodeSocketFloat'), ('Background', 'NodeSocketColor'))


def build_forward_group(recipe, images, uv_maps, colour_attribute=None, interpolation='Linear', fixed=None, tangent=False):
    """A node group computing a forward recipe's Colour (HDR), Alpha (with the program's discard applied) and Background
    tint."""
    digest = recipe_sha256(recipe)
    group = bpy.data.node_groups.new(FORWARD_PREFIX + digest[:16], 'ShaderNodeTree')
    group[FORWARD_PROPERTY] = digest
    for socket, kind in FORWARD_OUTPUTS:
        _socket(group, socket, 'OUTPUT', kind)
    out = group.nodes.new('NodeGroupOutput')
    em = _Emitter(group, recipe, images, uv_maps, colour_attribute, interpolation, fixed, tangent)
    terms = {k: em.value(i) for k, i in recipe['terms'].items()}
    colour = em.node('ShaderNodeCombineColor')
    for k, name in enumerate(('colour.r', 'colour.g', 'colour.b')):
        em.feed(terms[name], colour.inputs[k])
    em.link(colour.outputs[0], out.inputs['Colour'])
    tint = em.node('ShaderNodeCombineColor')
    for k, name in enumerate(('background.r', 'background.g', 'background.b')):
        em.feed(terms.get(name, 0.0), tint.inputs[k])
    em.link(tint.outputs[0], out.inputs['Background'])
    clip = terms['clip']
    keep = em.math('SUBTRACT', 1.0, clip) if not isinstance(clip, float) else 1.0 - clip
    alpha = terms['alpha']
    em.feed(alpha * keep if isinstance(alpha, float) and isinstance(keep, float) else em.math('MULTIPLY', alpha, keep),
            out.inputs['Alpha'])
    out.location = ((em.count // 40 + 1) * 200, 0)
    return group


def load_image(path, colourspace, kind='png', width=0, height=0):
    """A unit texture as written by rc2016.games.rcra.blender_export: PNG, 16-bit linear PNG, or raw float32 RGBA rows top
    first."""
    if kind == 'float':
        import numpy as np
        name = Path(path).stem
        image = bpy.data.images.get(name)
        if image is None or image.get('rc_source') != str(path):
            data = np.fromfile(str(path), dtype='<f4').reshape(height, width, 4)[::-1]   # Blender rows bottom first
            image = bpy.data.images.new(name, width, height, alpha=True, float_buffer=True)
            image.pixels.foreach_set(np.ascontiguousarray(data).ravel())
            image.file_format = 'OPEN_EXR'
            image['rc_source'] = str(path)
            image.pack()
    else:
        image = bpy.data.images.load(str(path), check_existing=True)
    if kind == 'png16':   # sRGB texels decoded to linear when written (blender_export._save): filtered as data
        colourspace = 'Non-Color'
    image.colorspace_settings.name = colourspace
    image.alpha_mode = 'CHANNEL_PACKED'
    return image


class Volume:
    """A 3D texture's padded slice atlas (rc2016.games.rcra.blender_export.write_volume_texture)."""

    def __init__(self, image, depth, columns, width, height):
        self.image, self.depth, self.columns, self.width, self.height = image, depth, columns, width, height


class Cube:
    """A cube map as an equirectangular image (rc2016.games.rcra.blender_export.write_cube_texture), and its alpha image."""

    def __init__(self, image, alpha=None):
        self.image, self.alpha = image, alpha


def _load_row(row):
    if 'cube' in row:
        alpha = row['cube'].get('alpha')
        return Cube(load_image(row['path'], row['colourspace'], row.get('kind', 'png'), row.get('width', 0), row.get('height', 0)),
                    _load_row(alpha) if alpha else None)
    if 'slices' in row:   # a texture array sampled at a run-time slice: one image per slice
        return [_load_row(r) for r in row['slices']]
    if 'volume' in row:
        v = row['volume']
        image = load_image(row['path'], row['colourspace'], row.get('kind', 'png'), row.get('width', 0), row.get('height', 0))
        return Volume(image, v['depth'], v['columns'], v['width'], v['height'])
    return load_image(row['path'], row['colourspace'], row.get('kind', 'png'), row.get('width', 0), row.get('height', 0))


def load_images(rows):
    return {binding: _load_row(row) for binding, row in rows.items()}


def _surface_shader(tree, entry, images, uv_maps, colour_attribute, blended, tangent=False):
    """The surface group feeding the BRDF group; returns the shader socket (alpha clip applied when opaque)."""
    recipe = entry['recipe']
    surface = build_surface_group(recipe, images, uv_maps, colour_attribute, tangent=tangent)
    s = tree.nodes.new('ShaderNodeGroup')
    s.node_tree, s.name, s.location = surface, 'Translated surface', (0, 0)
    b = tree.nodes.new('ShaderNodeGroup')
    b.node_tree, b.name, b.location = build_brdf_group(), 'Game BRDF', (300, 0)
    for socket in ('Albedo', 'F0', 'Gloss', 'Normal', 'Occlusion') + (() if blended else ('Emission',)):
        tree.links.new(s.outputs[socket], b.inputs[socket])
    clip = recipe['nodes'][recipe['terms']['clip']]
    if clip['op'] != 'const' or clip['value'] or 'coverage' in recipe['terms']:
        # alpha-tested (the writer's discard) or partly covered (a fur shell): a transparent mix
        mix = tree.nodes.new('ShaderNodeMixShader')
        clear = tree.nodes.new('ShaderNodeBsdfTransparent')
        mix.location, clear.location = (560, 0), (300, -250)
        tree.links.new(s.outputs['Alpha'], mix.inputs['Fac'])
        tree.links.new(clear.outputs[0], mix.inputs[1])
        tree.links.new(b.outputs['Shader'], mix.inputs[2])
        return mix.outputs[0], surface
    return b.outputs['Shader'], surface


def _forward_shader(tree, entry, images, uv_maps, colour_attribute, lit, tangent=False):
    """The forward group composed over a Transparent BSDF by the entry's blend mode."""
    group = build_forward_group(entry['forward'], images, uv_maps, colour_attribute, tangent=tangent)
    f = tree.nodes.new('ShaderNodeGroup')
    f.node_tree, f.name, f.location = group, 'Translated forward', (0, -400)
    nodes, links = tree.nodes, tree.links
    clear = nodes.new('ShaderNodeBsdfTransparent')
    clear.location = (300, -600)
    mode = entry['blend']
    if mode == 'modulate':
        tint = nodes.new('ShaderNodeMix')
        tint.data_type, tint.location = 'RGBA', (300, -400)
        links.new(f.outputs['Alpha'], tint.inputs['Factor'])
        tint.inputs['A'].default_value = (1.0, 1.0, 1.0, 1.0)
        links.new(f.outputs['Colour'], tint.inputs['B'])
        links.new(tint.outputs['Result'], clear.inputs['Color'])
        return clear.outputs[0], group
    emit = nodes.new('ShaderNodeEmission')
    emit.location = (300, -400)
    links.new(f.outputs['Colour'], emit.inputs['Color'])
    total = nodes.new('ShaderNodeAddShader')
    total.location = (1000, -400)
    links.new(clear.outputs[0], total.inputs[0])
    if mode == 'additive':
        through = nodes.new('ShaderNodeVectorMath')   # 1 + alpha B
        through.operation, through.location = 'MULTIPLY_ADD', (120, -600)
        links.new(f.outputs['Background'], through.inputs[0])
        links.new(f.outputs['Alpha'], through.inputs[1])
        through.inputs[2].default_value = (1.0, 1.0, 1.0)
        links.new(through.outputs[0], clear.inputs['Color'])
        links.new(f.outputs['Alpha'], emit.inputs['Strength'])
        links.new(emit.outputs[0], total.inputs[1])
        return total.outputs[0], group
    through = nodes.new('ShaderNodeMix')   # mix(white, B, alpha) = 1 - alpha + alpha B
    through.data_type, through.location = 'RGBA', (120, -600)
    links.new(f.outputs['Alpha'], through.inputs['Factor'])
    through.inputs['A'].default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(f.outputs['Background'], through.inputs['B'])
    links.new(through.outputs['Result'], clear.inputs['Color'])
    body = emit.outputs[0]
    if lit is not None:
        combine = nodes.new('ShaderNodeAddShader')
        combine.location = (560, -400)
        links.new(lit, combine.inputs[0])
        links.new(emit.outputs[0], combine.inputs[1])
        body = combine.outputs[0]
    weight = nodes.new('ShaderNodeMixShader')   # alpha x body: the first input is left empty (no light)
    weight.location = (800, -400)
    links.new(f.outputs['Alpha'], weight.inputs['Fac'])
    links.new(body, weight.inputs[2])
    links.new(weight.outputs[0], total.inputs[1])
    return total.outputs[0], group


def apply_recipe(material, entry, uv_maps, colour_attribute=None, tangent=False):
    """Replace ``material``'s nodes with the translated material.

    ``entry``: dict(blend, recipe (G-buffer surface), forward (blended materials), images {binding: row}, limitations).
    An opaque entry needs ``recipe``; a blended one needs ``forward`` and uses ``recipe`` for its lit part when present.
    ``tangent``: the mesh carries the decoded tangent (``store_tangent_frame``); else the UV0 frame stands in, labelled."""
    mode = entry.get('blend', 'opaque')
    if mode not in BLEND_MODES:
        raise ValueError('Unknown blend mode ' + str(mode))
    images = load_images(entry['images'])
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    output.location = (1100, 0)
    shader, surface, forward = None, None, None
    if entry.get('recipe'):
        shader, surface = _surface_shader(tree, entry, images, uv_maps, colour_attribute, mode != 'opaque', tangent)
    if mode != 'opaque':
        shader, forward = _forward_shader(tree, entry, images, uv_maps, colour_attribute,
                                          shader if mode == 'blended' else None, tangent)
        if hasattr(material, 'surface_render_method'):   # EEVEE draws it sorted and blended
            material.surface_render_method = 'BLENDED'
    if shader is None:
        raise ValueError('Entry has no surface recipe')
    tree.links.new(shader, output.inputs['Surface'])
    for key, group, prop in (('recipe', surface, RECIPE_PROPERTY), ('forward', forward, FORWARD_PROPERTY)):
        if group is not None:
            material[prop] = group[prop]
        elif prop in material:
            del material[prop]
    material['rc_rcra_blend'] = mode
    material['rc_rcra_limitations'] = '\n'.join(entry.get('limitations', []) + ([] if tangent else [NO_TANGENT_NOTE]))
    material['rc_rcra_tangent'] = bool(tangent)
    return material


def store_tangent_frame(mesh):
    """Rewrite an imported mesh's decoded tangent (point attribute TANGENT_ATTRIBUTE: native xyz and TEXCOORD3.w, which
    the glTF importer makes a float colour attribute) as its coordinates (a, b, c) in Blender's UV0 tangent frame at each
    corner (corner attribute TANGENT_FRAME): T = a t + b (n x t) + c n, with t the MikkTSpace tangent of the first UV map
    and n the corner normal; and w as the point attribute TANGENT_W_ATTRIBUTE. The source attribute is removed, so it is
    not taken for the vertex colour. Returns False, leaving the mesh unchanged, when it has no decoded tangent or no UV
    map."""
    import numpy as np
    source = mesh.attributes.get(TANGENT_ATTRIBUTE)
    if source is None or source.domain != 'POINT' or not mesh.uv_layers or not len(mesh.loops):
        return False
    packed = np.empty(len(mesh.vertices) * 4, np.float32)
    source.data.foreach_get('color' if source.data_type in ('FLOAT_COLOR', 'BYTE_COLOR') else 'vector', packed)
    packed = packed.reshape(-1, 4)
    native, w = packed[:, :3].astype(np.float64), packed[:, 3].copy()
    tangent = np.stack((native[:, 0], -native[:, 2], native[:, 1]), 1)   # game (x, y, z) = Blender (x, z, -y)
    mesh.calc_tangents(uvmap=mesh.uv_layers[0].name)
    count = len(mesh.loops)
    t, n = np.empty(count * 3, np.float32), np.empty(count * 3, np.float32)
    vertex = np.empty(count, np.int32)
    mesh.loops.foreach_get('tangent', t)
    mesh.loops.foreach_get('normal', n)
    mesh.loops.foreach_get('vertex_index', vertex)
    mesh.free_tangents()
    t, n = t.reshape(-1, 3).astype(np.float64), n.reshape(-1, 3).astype(np.float64)
    # Solved in the basis the material rebuilds from, which is skewed where Blender's tangent is not orthogonal to the
    # corner normal (degenerate UV triangles); least squares where it is singular (no UV tangent).
    basis = np.stack((t, np.cross(n, t), n), 2)
    coords = (np.linalg.pinv(basis) @ tangent[vertex][:, :, None])[:, :, 0].astype(np.float32)
    for name in (TANGENT_FRAME, TANGENT_W_ATTRIBUTE):
        if mesh.attributes.get(name) is not None:
            mesh.attributes.remove(mesh.attributes[name])
    mesh.attributes.new(TANGENT_FRAME, 'FLOAT_VECTOR', 'CORNER').data.foreach_set('vector', coords.ravel())
    mesh.attributes.new(TANGENT_W_ATTRIBUTE, 'FLOAT', 'POINT').data.foreach_set('value', w)
    mesh.attributes.remove(mesh.attributes[TANGENT_ATTRIBUTE])
    return True


def translated(material):
    """True for a material apply_recipe built."""
    return bool(material.get(RECIPE_PROPERTY) or material.get(FORWARD_PROPERTY))


def _group_nodes(group):
    """The nodes of a translated group and of its part groups."""
    for node in group.nodes:
        yield node
        if node.type == 'GROUP' and node.node_tree is not None and node.node_tree.name.startswith(group.name + ' part '):
            yield from _group_nodes(node.node_tree)


def _images_missing(material, group):
    problems = []
    for node in _group_nodes(group):
        if node.type in ('TEX_IMAGE', 'TEX_ENVIRONMENT'):
            image = node.image
            if image is None or not (image.packed_file or image.has_data or image.size[0]):
                problems.append('%s: %s has no image data' % (material.name, node.name))
    return problems


def validate_material(material):
    """Problems with a translated material after a reopen; empty when it is intact."""
    if not translated(material):
        return ['%s: no translated recipe' % material.name]
    problems = []
    tree = material.node_tree
    mode = material.get('rc_rcra_blend', 'opaque')
    digest = material.get(RECIPE_PROPERTY)
    if mode == 'opaque' and not digest:
        problems.append('%s: opaque material without a surface recipe' % material.name)
    if digest:
        surface = tree.nodes.get('Translated surface') if tree else None
        brdf = tree.nodes.get('Game BRDF') if tree else None
        if surface is None or surface.node_tree is None or surface.node_tree.get(RECIPE_PROPERTY) != digest:
            problems.append('%s: surface group missing or not its recipe' % material.name)
        if brdf is None or brdf.node_tree is None or brdf.node_tree.get('rc_version') != GROUP_VERSION:
            problems.append('%s: BRDF group missing or out of date' % material.name)
        if surface is not None and surface.node_tree is not None:
            problems += _images_missing(material, surface.node_tree)
            for socket in surface.node_tree.nodes['Group Output'].inputs[:len(SURFACE_OUTPUTS)]:
                if not socket.is_linked and socket.name not in ('Gloss', 'Occlusion', 'Emission', 'Alpha'):
                    problems.append('%s: surface output %s unlinked' % (material.name, socket.name))
    if mode != 'opaque':
        forward = tree.nodes.get('Translated forward') if tree else None
        want = material.get(FORWARD_PROPERTY)
        if not want or forward is None or forward.node_tree is None or forward.node_tree.get(FORWARD_PROPERTY) != want:
            problems.append('%s: forward group missing or not its recipe' % material.name)
        elif not forward.node_tree.nodes['Group Output'].inputs['Colour'].is_linked:
            problems.append('%s: forward colour unlinked' % material.name)
        else:
            problems += _images_missing(material, forward.node_tree)
    return problems


SHELL_LAYER_PROPERTY = 'rc_rcra_fur_layer'


def _fur_mesh(obj, material):
    """A copy of ``obj``'s mesh holding only the faces that use ``material`` (the fur), with their vertices."""
    import bmesh
    slots = {i for i, slot in enumerate(obj.material_slots) if slot.material == material}
    mesh = obj.data.copy()
    mesh.name = obj.data.name + ' fur'
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index not in slots], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def build_fur_shells(entries):
    """Draw the shell layers of every translated fur material (entry ``shells``, rc2016.games.rcra.blender_export
    fur_shells). Each object that uses the material gets, per layer, a linked copy of its fur faces that the layer's
    material shades, moved along the vertex normal by a Displace modifier ahead of the rig (the control map's blue channel,
    point-sampled at UV0, times the layer's ``displace`` metres), so the shells follow the skin. Returns the most layers
    one fur material draws (0 without fur): every one is a transparent surface a ray may cross."""
    most = 0
    for name, entry in entries.items():
        shells = entry.get('shells')
        base = bpy.data.materials.get(name)
        if not shells or base is None or not translated(base) or not entry.get('shell_height'):
            continue
        users = [o for o in list(bpy.data.objects)
                 if o.type == 'MESH' and any(slot.material == base for slot in o.material_slots)]
        if not users:
            continue
        mesh = users[0].data
        uv_maps = [layer.name for layer in mesh.uv_layers]
        if not uv_maps:
            continue
        colour = mesh.color_attributes[0].name if len(mesh.color_attributes) else None
        height = bpy.data.textures.new(name + ' fur shell height', 'IMAGE')
        height.image = _load_row(entry['shell_height'])
        height.use_interpolation, height.extension = False, 'REPEAT'   # point-sampled; a modifier reads the top mip
        materials = []
        for layer in shells:
            material = bpy.data.materials.new('%s fur %02d' % (name, layer['layer']))
            apply_recipe(material, dict(blend='opaque', recipe=layer['recipe'], images=layer['images'],
                                        limitations=layer['limitations']), uv_maps, colour,
                         tangent=TANGENT_FRAME in mesh.attributes)
            material[SHELL_LAYER_PROPERTY] = int(layer['layer'])
            materials.append(material)
        for obj in users:
            fur = _fur_mesh(obj, base)
            scale = sum(obj.matrix_world.to_scale()) / 3.0 or 1.0
            for layer, material in zip(shells, materials):
                copy = obj.copy()
                copy.data = fur
                copy.name = '%s fur %02d' % (obj.name, layer['layer'])
                for collection in obj.users_collection:
                    collection.objects.link(copy)
                for slot in copy.material_slots:
                    slot.link = 'OBJECT'
                    slot.material = material
                displace = copy.modifiers.new('Fur shell %d' % layer['layer'], 'DISPLACE')
                displace.texture, displace.texture_coords, displace.uv_layer = height, 'UV', uv_maps[0]
                displace.direction, displace.space, displace.mid_level = 'NORMAL', 'LOCAL', 0.0
                displace.strength = float(layer['displace']) / scale
                copy.modifiers.move(len(copy.modifiers) - 1, 0)   # before the armature: displaced at rest, then skinned
                copy[SHELL_LAYER_PROPERTY] = int(layer['layer'])
        most = max(most, len(shells) + 1)
        print('FUR SHELLS', name, len(shells), 'layers on', len(users), 'objects', flush=True)
    return most
