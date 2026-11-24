import unittest

import numpy as np
import torch

from explanations import action_pair, explain, grid, model_digest, randomized_copy, sarfa_components
from faithfulness import evaluate, intervention_path, metrics, ranked_cells


class FaithfulnessTests(unittest.TestCase):
    def test_fixed_pair_and_changing_competitor(self):
        before, after = np.array([5., 4., 1.]), np.array([3., 1., 4.])
        a, b = action_pair(before)
        self.assertEqual((a, b), (0, 1))
        self.assertEqual(after[a] - after[b], 2.)
        self.assertEqual(metrics(after, a)["margin"], -1.)
        self.assertAlmostEqual(metrics(after, a)["centered"], 1 / 3)
        self.assertTrue(metrics(after, a)["flipped"])
        for name in ["margin", "centered", "flipped"]:
            np.testing.assert_allclose(metrics(after, a)[name], metrics(after + 100, a)[name])
        self.assertEqual(action_pair(np.array([2., 2., 1.])), (0, 1))
        self.assertTrue(metrics(np.array([2., 2., 1.]), 1)["flipped"])

    def test_grid_covers_every_pixel_once(self):
        cover = np.zeros((84, 84))
        for y0, y1, x0, x1 in grid():
            cover[y0:y1, x0:x1] += 1
        np.testing.assert_array_equal(cover, 1)

    def test_synchronous_paths_and_endpoints(self):
        stack = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4) + 1
        base = -np.ones_like(stack)
        cells, counts, order = grid(4, 4, 2), np.array([0, 1, 4]), np.array([2, 0, 3, 1])
        for direction, start, end in [("deletion", stack, base), ("insertion", base, stack)]:
            path = intervention_path(stack, base, cells, order, counts, direction)
            np.testing.assert_array_equal(path[0], start)
            np.testing.assert_array_equal(path[-1], end)
            changed = path[1] != start
            expected = np.zeros_like(changed)
            expected[:, 2:4, 0:2] = True
            np.testing.assert_array_equal(changed, expected)
        np.testing.assert_array_equal(ranked_cells(np.zeros((4, 4)), cells), np.arange(4))

    def test_invalid_order_is_rejected(self):
        stack = np.ones((4, 4, 4), np.float32)
        with self.assertRaises(ValueError):
            intervention_path(stack, stack, grid(4, 4, 2), [0, 0, 1, 2], [0, 4], "deletion")

    def test_sarfa_specificity_and_common_shift(self):
        before = np.array([5., 4., 1.])
        after = np.array([before - 10, [3., 4., 1.]])
        scores, dp, specificity, kl = sarfa_components(before, after, 0)
        self.assertAlmostEqual(scores[0], 0)
        self.assertGreater(scores[1], 0)
        self.assertAlmostEqual(specificity[1], 1)
        self.assertAlmostEqual(kl[1], 0)
        p_before = np.exp(5) / np.exp(before).sum()
        p_after = np.exp(3) / np.exp(after[1]).sum()
        self.assertAlmostEqual(scores[1], 2 * (p_before - p_after) / (1 + p_before - p_after))
        np.testing.assert_allclose(scores, sarfa_components(before + 1000, after + 1000, 0)[0], atol=1e-12)

    def test_randomization_preserves_model_and_rng(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        digest, rng = model_digest(model), torch.random.get_rng_state().clone()
        changed = randomized_copy(model, 7)
        self.assertEqual(model_digest(model), digest)
        self.assertNotEqual(model_digest(changed), digest)
        self.assertEqual(model_digest(changed), model_digest(randomized_copy(model, 7)))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(model[0].weight, changed[0].weight))

    def test_known_linear_ig_and_curve_auc(self):
        torch.set_num_threads(1)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 3, bias=False))
        with torch.no_grad():
            model[1].weight.zero_()
            model[1].weight[0].fill_(1 / 64)
        stack = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        maps, info, _ = explain(model, stack, pair=(0, 1), patch=2)
        self.assertTrue(info["ig"]["converged"])
        self.assertAlmostEqual(maps["ig"].sum(), info["ig"]["output_difference"], places=4)
        cells = grid(4, 4, 2)
        for direction in ["deletion", "insertion"]:
            q, values, summary = evaluate(model, stack, np.zeros_like(stack), cells,
                                          np.arange(4), np.arange(5), direction, 0)
            self.assertEqual(q.shape, (5, 3))
            self.assertAlmostEqual(summary["margin_auc"], np.trapezoid(values["margin"], np.arange(5) / 4))


if __name__ == "__main__":
    unittest.main()
