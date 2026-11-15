"""sRGB textures reach Cycles as linear texels, so Cycles filters decoded values as the PS4 GPU does.

The Blender test runs the pipeline's own image loader (rc2016.pipeline.geometry._image) on a synthetic sRGB DDS and renders a
lookup at the corner of four contrasting texels; it needs Blender (RC2016_BLENDER, else the default install path).
"""
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rc2016.formats.linear_png import SRGB8_TO_LINEAR16, SRGB_LINEAR16_CONVERSION, srgb8_linear16, srgb_to_linear
from rc2016.formats.texture import dds_bytes
from rc2016.formats.texture_transport import blender_conversion, image_interpretation_identity

BLENDER = os.environ.get('RC2016_BLENDER') or r'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
ROOT = Path(__file__).resolve().parents[1]
# 2 x 2 R8G8B8A8_UNORM_SRGB texels, rows top first: a 0.1 / 0.9 checker, alpha 64 / 200 (alpha is linear data).
CODES = np.array([[26, 230], [230, 26]], np.uint8)
ALPHA = np.array([[64, 200], [200, 64]], np.uint8)

SCRIPT = r'''
import json, sys
import bpy
import numpy as np

job = json.loads(open(sys.argv[sys.argv.index('--') + 1], encoding='utf-8').read())
sys.path.insert(0, job['root'])
from rc2016.pipeline import geometry
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.resolution_x = scene.render.resolution_y = 2
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.render.image_settings.file_format = 'OPEN_EXR'
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
mesh = bpy.data.meshes.new('plane')
mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
plane = bpy.data.objects.new('plane', mesh)
scene.collection.objects.link(plane)
camera = bpy.data.objects.new('camera', bpy.data.cameras.new('camera'))
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 1.0
camera.location = (0, 0, 1)
scene.collection.objects.link(camera)
scene.camera = camera
results = {}
for name, binding in job['bindings'].items():
    image = geometry._image(binding)
    texels = np.empty(len(image.pixels), np.float32)
    image.pixels.foreach_get(texels)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    tex = tree.nodes.new('ShaderNodeTexImage')
    tex.image, tex.interpolation, tex.extension = image, 'Linear', 'REPEAT'
    corner = tree.nodes.new('ShaderNodeCombineXYZ')   # (0.5, 0.5): the corner of all four texels, each weighted 1/4
    corner.inputs[0].default_value = corner.inputs[1].default_value = 0.5
    emission = tree.nodes.new('ShaderNodeEmission')
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    tree.links.new(corner.outputs[0], tex.inputs['Vector'])
    tree.links.new(tex.outputs['Color'], emission.inputs['Color'])
    tree.links.new(emission.outputs[0], output.inputs['Surface'])
    mesh.materials.clear()
    mesh.materials.append(material)
    scene.render.filepath = str(job['folder'] + '/' + name + '.exr')
    bpy.ops.render.render(write_still=True)
    render = bpy.data.images.load(scene.render.filepath)
    pixels = np.empty(len(render.pixels), np.float32)
    render.pixels.foreach_get(pixels)
    results[name] = dict(lookup=pixels.reshape(-1, 4)[:, :3].tolist(), texels=texels.reshape(2, 2, 4)[::-1].tolist(),
                         colorspace=image.colorspace_settings.name, conversion=image.get('blender_conversion'),
                         packed=bool(image.packed_file), marker=image.get('packed_container'))
json.dump(results, open(job['out'], 'w', encoding='utf-8'))
'''


class LinearTexelTests(unittest.TestCase):
    def test_lut_matches_the_float_decode(self):
        codes = np.arange(256)
        np.testing.assert_array_equal(SRGB8_TO_LINEAR16, np.round(srgb_to_linear(codes / 255.0) * 65535.0))
        self.assertEqual(len(set(SRGB8_TO_LINEAR16.tolist())), 256)   # every 8-bit value stays distinct
        self.assertLess(np.abs(SRGB8_TO_LINEAR16 / 65535.0 - srgb_to_linear(codes / 255.0)).max(), 7.7e-6)

    def test_alpha_is_stored_as_it_is(self):
        rgba = np.dstack([CODES, CODES, CODES, ALPHA])
        out = srgb8_linear16(rgba)
        np.testing.assert_array_equal(out[..., 3], ALPHA.astype(np.uint16) * 257)
        np.testing.assert_array_equal(out[..., 0], SRGB8_TO_LINEAR16[CODES])

    def test_srgb_bindings_load_as_linear_non_color(self):
        self.assertEqual(blender_conversion(72, 'sRGB'), SRGB_LINEAR16_CONVERSION)
        self.assertEqual(blender_conversion(71, 'Non-Color'), 'DDS_NATIVE_V1')
        self.assertEqual(blender_conversion(49, 'Non-Color'), 'R8G8_UNORM_DDS_TO_RGBA8_PNG_V1')
        identity = image_interpretation_identity(dict(path='textures/a_c.texture', dds_sha256='AB', colorspace='sRGB'))
        self.assertEqual((identity['colorspace'], identity['conversion']), ('Non-Color', SRGB_LINEAR16_CONVERSION))
        # A binding recorded with the native conversion keeps Blender's sRGB decode.
        native = image_interpretation_identity(dict(path='textures/a_c.texture', dds_sha256='ab', colorspace='sRGB',
                                                    conversion_kind='DDS_NATIVE_V1'))
        self.assertEqual(native['colorspace'], 'sRGB')

    @unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
    def test_cycles_filters_decoded_texels(self):
        """Between contrasting texels the GPU filters decoded values; the pipeline's image makes Cycles do the same, while
        Blender's own sRGB decode (filter first) reads darker."""
        rgba = np.dstack([CODES, CODES, CODES, ALPHA])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            dds = folder / 'srgb_checker.dds'
            dds.write_bytes(dds_bytes(2, 2, 29, rgba.tobytes()))
            binding = dict(path='textures/srgb_checker_c.texture', dds_path=str(dds), format=29, width=2, height=2,
                           role='base_color', colorspace='sRGB', dds_sha256=hashlib.sha256(dds.read_bytes()).hexdigest())
            job = folder / 'job.json'
            job.write_text(json.dumps(dict(root=str(ROOT), folder=folder.as_posix(), out=str(folder / 'out.json'), bindings=dict(
                linear=binding, native=dict(binding, conversion_kind='DDS_NATIVE_V1')))), encoding='utf-8')
            script = folder / 'check.py'
            script.write_text(SCRIPT, encoding='utf-8')
            env = dict(os.environ, BLENDER_USER_RESOURCES=str(folder / 'user'))
            run = subprocess.run([BLENDER, '--background', '--factory-startup', '--python-exit-code', '1', '--python',
                                  str(script), '--', str(job)], capture_output=True, text=True, env=env, encoding='utf-8',
                                 errors='replace', timeout=600)
            self.assertEqual(run.returncode, 0, (run.stdout + run.stderr)[-3000:])
            got = json.loads((folder / 'out.json').read_text(encoding='utf-8'))
            self.assertTrue((folder / 'srgb_checker.linear16.png').is_file())
        linear, native = got['linear'], got['native']
        self.assertEqual((linear['colorspace'], linear['conversion']), ('Non-Color', SRGB_LINEAR16_CONVERSION))
        self.assertTrue(linear['packed'] and linear['marker'])
        texels = np.array(linear['texels'])
        np.testing.assert_allclose(texels[..., :3], srgb_to_linear(rgba[..., :3] / 255.0), atol=8e-6)
        np.testing.assert_allclose(texels[..., 3], ALPHA / 255.0, atol=1e-6)
        want = float(srgb_to_linear(CODES / 255.0).mean())   # decode, then filter: 0.401
        np.testing.assert_allclose(np.array(linear['lookup']), want, atol=2e-5)
        # Blender's sRGB image filters the codes and decodes after (0.216): the error this conversion removes.
        filtered_first = float(srgb_to_linear(CODES.mean() / 255.0))
        self.assertEqual(native['colorspace'], 'sRGB')
        np.testing.assert_allclose(np.array(native['lookup']), filtered_first, atol=2e-3)
        self.assertGreater(want - filtered_first, 0.15)


if __name__ == '__main__':
    unittest.main()
