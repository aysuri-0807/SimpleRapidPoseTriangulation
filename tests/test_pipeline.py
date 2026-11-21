"""Check the geometry and an end-to-end matching failure mode."""

import unittest

import numpy as np

from dependencies import (
    make_demo_scene,
    pair_score,
    project_points,
    reconstruct_pose,
    score_candidates,
    select_matches,
    triangulate_midpoint,
)


class ReconstructionTests(unittest.TestCase):
    def test_two_calibrated_rays_recover_a_known_3d_point(self) -> None:
        scene = make_demo_scene(noise_px=0)
        point = scene.truth[0, 0]
        a = project_points(scene.cameras["A"], point)[0]
        b = project_points(scene.cameras["B"], point)[0]
        result = triangulate_midpoint(scene.cameras["A"], a, scene.cameras["B"], b)
        np.testing.assert_allclose(result, point, atol=1e-9)

    def test_shuffled_people_match_and_bad_joint_raises_error(self) -> None:
        scene = make_demo_scene()
        scores = score_candidates(scene)
        matches = select_matches(scores, threshold_px=4)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})

        for i, j in matches:
            pose = reconstruct_pose(scene.cameras["A"], scene.detections["A"][i],
                                    scene.cameras["B"], scene.detections["B"][j])
            self.assertLess(np.mean(np.linalg.norm(pose - scene.truth[i], axis=1)), .04)

        corrupted = scene.detections["B"][1].copy()
        corrupted[0, 1] += 40
        self.assertGreater(pair_score(scene.cameras["A"], scene.detections["A"][0],
                                      scene.cameras["B"], corrupted), 4)


if __name__ == "__main__":
    unittest.main()
