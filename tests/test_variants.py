"""Model variant kinds the browser's Variants menu hides (nwo.app.variants)."""

import unittest

from nwo.app.variants import VARIANT_KINDS, search_reveals, variant_kind

PATHS = [
    "characters/enemy/enm_bladeball_ps4/enm_bladeball_ps4.model",
    "characters/enemy/enm_bladeball_ps4/enm_bladeball_ps4_topiary.model",
    "characters/enemy/enm_bladeball_ps4/enm_bladeball_ps4_skeleton.model",
    "characters/npc/npc_nefarious/npc_nefarious.model",
    "characters/npc/npc_nefarious/npc_nefarious_topiary.model",
    "characters/npc/npc_nefarious/npc_nefarious_hologram.model",
    "vehicle/veh_thug_dropship/veh_thug_dropship.model",
    "vehicle/veh_thug_dropship/veh_thug_dropship_topiary.model",
    "equipment/weapon/wpn_topiaryblaster/grenade_topiaryblaster.model",
    "visualeffect/model/topiary/topiary_sidewalkflowers_06.model",
    "characters/test/test_topiary_pose_01/test_topiary_pose_01b.model",
    "legacy/rcf/maya/entities/object/vehicle/enemy/pirate_skiff/pirate_skiff_topiary.model",
    "environment/savali/stairs/sav_stairs_v1.model",
    "environment/savali/stairs/sav_stairs_v1_snow.model",
    "environment/x/prop/lantern_anim.model",   # no base lantern.model: stays listed
    "characters/enemy/enm_pirate_robot/enm_pirate_robot.model",
    "characters/enemy/enm_pirate_robot/enm_pirate_robot_art.model",
    "characters/enemy/enm_neo_nefarious_heavy/enm_neo_nefarious_heavy.model",
    "characters/enemy/enm_neo_nefarious_heavy/enm_neo_nefarious_heavy_economy.model",
    "characters/enemy/enm_poolshark/enm_pool_shark_art.model",   # no base: stays listed
]
LISTED = frozenset(p.casefold() for p in PATHS)


def kind(path):
    return variant_kind(path, LISTED)


class VariantKindTests(unittest.TestCase):
    def test_topiary_variants_of_enemies_npcs_and_vehicles(self):
        self.assertEqual(kind(PATHS[1]), "topiary")
        self.assertEqual(kind(PATHS[4]), "topiary")
        self.assertEqual(kind(PATHS[7]), "topiary")

    def test_base_models_weapon_and_plants_stay(self):
        for path in (PATHS[0], PATHS[3], PATHS[6], PATHS[8], PATHS[9], PATHS[12]):
            self.assertIsNone(kind(path), path)

    def test_suffix_without_a_listed_base_stays(self):
        self.assertIsNone(kind("environment/x/prop/lantern_anim.model"))

    def test_folders_and_suffixes(self):
        self.assertEqual(kind(PATHS[10]), "test")
        self.assertEqual(kind(PATHS[11]), "legacy")   # its base pirate_skiff.model is not listed here
        listed = LISTED | {"legacy/rcf/maya/entities/object/vehicle/enemy/pirate_skiff/pirate_skiff.model"}
        self.assertEqual(variant_kind(PATHS[11], listed), "topiary")

    def test_other_suffix_kinds(self):
        self.assertEqual(kind(PATHS[2]), "skeleton")
        self.assertEqual(kind(PATHS[5]), "special")
        self.assertEqual(kind(PATHS[13]), "snow")
        self.assertEqual(kind(PATHS[16]), "art")
        self.assertEqual(kind(PATHS[18]), "economy")
        self.assertIsNone(kind(PATHS[19]))

    def test_case_and_backslashes(self):
        self.assertEqual(kind(PATHS[1].upper().replace("/", "\\")), "topiary")

    def test_defaults_hide_topiary_test_folders_lighter_copies_and_models_without_geometry(self):
        self.assertEqual({k.key for k in VARIANT_KINDS if k.hidden_by_default},
                         {"topiary", "test", "economy", "art", "no_geometry"})

    def test_a_record_flag_marks_models_without_geometry(self):
        from nwo.app.batch import PipelineModelCandidate
        empty = PipelineModelCandidate(model_id="1", archive_path=PATHS[1], no_geometry=True)
        self.assertEqual(variant_kind(PATHS[1], LISTED, empty), "no_geometry")   # before the topiary suffix
        self.assertEqual(variant_kind(PATHS[1], LISTED, PipelineModelCandidate(model_id="2", archive_path=PATHS[1])),
                         "topiary")
        self.assertTrue(search_reveals("no_geometry", ["geometry"]))

    def test_search_word_reveals_a_kind(self):
        self.assertTrue(search_reveals("topiary", ["bladeball", "topiary"]))
        self.assertTrue(search_reveals("legacy", ["legacy"]))
        self.assertTrue(search_reveals("test", ["test"]))
        self.assertTrue(search_reveals("duplicate", ["_static"]))
        self.assertFalse(search_reveals("topiary", ["bladeball"]))
        self.assertFalse(search_reveals("topiary", ["topiaryblaster"]))


if __name__ == "__main__":
    unittest.main()
