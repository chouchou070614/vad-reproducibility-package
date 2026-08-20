import unittest

import numpy as np

from src.decision.frozen_pipeline import (
    branch_score,
    ecdf_tail,
    exact_align,
    fixed_fuse,
    refine_per_video,
)


class FrozenPipelineTest(unittest.TestCase):
    def test_exact_alignment_has_no_padding(self):
        video, frame, r, p = exact_align(
            ["01", "01", "01"], [0, 1, 2], [0.1, 0.2, 0.3],
            ["01", "01"], [1, 2], [0.4, 0.5],
        )
        self.assertEqual(video.tolist(), ["01", "01"])
        self.assertEqual(frame.tolist(), [1, 2])
        np.testing.assert_allclose(r, [0.2, 0.3])
        np.testing.assert_allclose(p, [0.4, 0.5])

    def test_ecdf_matches_locked_formula(self):
        result = ecdf_tail([0.0, 1.0, 2.0], [0.5, 2.5])
        expected_u = np.asarray([(1.0 + 0.5) / 4.0, (3.0 + 0.5) / 4.0])
        np.testing.assert_allclose(result, -np.log(1.0 - expected_u + 1e-8))

    def test_refinement_resets_at_video_boundary(self):
        result = refine_per_video([0.0, 1.0, 10.0, 0.0], [0, 0, 1, 1], "ema", 0.5)
        np.testing.assert_allclose(result, [0.0, 0.5, 10.0, 5.0])

    def test_branch_and_fixed_fusion(self):
        pixel = [0.01, 0.02, 0.04, 0.01, 0.03, 0.05]
        memory = [0.1, 0.2, 0.4, 0.1, 0.2, 0.5]
        score = branch_score(pixel, memory, [0, 0, 0, 1, 1, 1], 0.7)
        self.assertTrue(np.isfinite(score).all())
        np.testing.assert_allclose(fixed_fuse(score, score, 0.4), score)


if __name__ == "__main__":
    unittest.main()
