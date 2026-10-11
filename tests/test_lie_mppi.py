"""Geometry and controller regression tests (CPU, no checkpoint required)."""

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from controllers.lie_mppi import (
    TWO_PI, mppi_batched, pose_distance, riemannian_exp,
    sample_rollouts, sample_unit_tangents,
)


class LieMPPIGeometryTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(17)
        self.dtype = torch.float64

    def directions(self, batch=4, samples=30, horizon=5, planar=False):
        return sample_unit_tangents(batch, samples, horizon, 'cpu', self.dtype,
                                    planar, self.generator)

    def test_every_horizon_sample_has_exact_original_step(self):
        current = torch.randn(4, 6, generator=self.generator, dtype=self.dtype)
        tangents = self.directions()
        rollout = sample_rollouts(current, tangents, 0.015)
        previous = torch.cat([current[:, None, None].expand(-1, 30, 1, -1),
                              rollout[:, :, :-1]], dim=2)
        # Measure with an independent SciPy SO(3) implementation.
        r0 = Rotation.from_rotvec((previous[..., 3:] * TWO_PI).reshape(-1, 3))
        r1 = Rotation.from_rotvec((rollout[..., 3:] * TWO_PI).reshape(-1, 3))
        angle = (r0.inv() * r1).magnitude().reshape(4, 30, 5) / TWO_PI
        dp = (rollout[..., :3] - previous[..., :3]).numpy()
        distance = np.sqrt(np.square(dp).sum(-1) + np.square(angle))
        np.testing.assert_allclose(distance, 0.015, atol=1e-12, rtol=0)
        torch.testing.assert_close(tangents.norm(dim=-1),
                                   torch.ones(4, 30, 5, dtype=self.dtype))

    def test_translation_follows_body_frame_and_rotation_composes(self):
        current = torch.tensor([1., 2., 3., 0., 0., 0.25], dtype=self.dtype)
        tangent = torch.tensor([1., 0., 0., 0.2, 0., 0.], dtype=self.dtype)
        result = riemannian_exp(current, tangent, 0.015)
        np.testing.assert_allclose(result[:3], [1., 2.015, 3.], atol=1e-12)
        reference = (Rotation.from_rotvec(current[3:].numpy() * TWO_PI)
                     * Rotation.from_rotvec(tangent[3:].numpy() * 0.015 * TWO_PI))
        actual = Rotation.from_rotvec(result[3:].numpy() * TWO_PI)
        self.assertLess((reference.inv() * actual).magnitude(), 1e-12)

    def test_half_turn_crossing_preserves_short_geodesic(self):
        current = torch.tensor([0., 0., 0., 0., 0., 0.499], dtype=self.dtype)
        tangent = torch.tensor([0., 0., 0., 0., 0., 1.], dtype=self.dtype)
        result = riemannian_exp(current, tangent, 0.015)
        self.assertLess(float(result[5]), 0)
        self.assertAlmostEqual(float(pose_distance(current, result)), 0.015, places=12)

    def test_equivalent_rotation_vectors_have_zero_distance(self):
        a = torch.tensor([0., 0., 0., 0.5, 0., 0.], dtype=self.dtype)
        b = a.clone()
        b[3] = -0.5
        self.assertLess(float(pose_distance(a, b)), 1e-12)

    def test_zero_tangent_is_identity_and_small_angles_are_finite(self):
        current = torch.zeros(3, 6, dtype=self.dtype)
        current[1, 3] = 0.5
        current[2, 4] = 1e-14
        result = riemannian_exp(current, torch.zeros_like(current), 0.015)
        self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(pose_distance(current, result),
                                   torch.zeros(3, dtype=self.dtype), atol=1e-12, rtol=0)

    def test_weighted_control_stays_within_step_budget(self):
        tangent = self.directions(batch=1, samples=50, horizon=2)[0, :, 0]
        weights = torch.softmax(torch.randn(50, generator=self.generator,
                                           dtype=self.dtype), 0)
        mean = (weights[:, None] * tangent).sum(0)
        current = torch.tensor([1., 0., 0., .4, .1, -.1], dtype=self.dtype)
        result = riemannian_exp(current, mean, 0.015)
        self.assertLessEqual(float(pose_distance(current, result)), 0.015 + 1e-12)
        self.assertAlmostEqual(float(pose_distance(current, result)),
                               0.015 * float(mean.norm()), places=12)

    def test_planar_rollout_preserves_frozen_coordinates(self):
        current = torch.zeros(4, 6, dtype=self.dtype)
        current[:, 2] = 0.123
        current[:, 5] = 0.48
        tangent = self.directions(planar=True)
        rollout = sample_rollouts(current, tangent, 0.015)
        torch.testing.assert_close(rollout[..., 2],
                                   torch.full((4, 30, 5), .123, dtype=self.dtype))
        torch.testing.assert_close(rollout[..., 3:5],
                                   torch.zeros(4, 30, 5, 2, dtype=self.dtype),
                                   atol=1e-12, rtol=0)


class LieMPPIControllerTests(unittest.TestCase):
    @staticmethod
    def cost(pairs):
        return pose_distance(pairs[:, :6], pairs[:, 6:])[:, None]

    def test_forward_and_alternating_reach_goal_and_preserve_input(self):
        pairs = torch.zeros(2, 12)
        pairs[0, 6] = .06
        pairs[1, 6:9] = torch.tensor([-.04, .03, .02])
        original = pairs.clone()
        for alternating in (False, True):
            with self.subTest(alternating=alternating):
                paths, iterations, success, minimum = mppi_batched(
                    self.cost, pairs, steps=100, temperature=150,
                    alternating=alternating,
                    generator=torch.Generator().manual_seed(9))
                self.assertEqual(success, [True, True])
                self.assertTrue(bool((minimum < .01).all()))
                self.assertTrue(all(index < 100 for index in iterations))
                for b, path in enumerate(paths):
                    torch.testing.assert_close(path[0], original[b:b + 1, :6])
                    torch.testing.assert_close(path[-1], original[b:b + 1, 6:])
                    for p, q in zip(path[:-1], path[1:]):
                        self.assertLessEqual(float(pose_distance(p, q)), .015 + 1e-6)
        torch.testing.assert_close(pairs, original)

    def test_already_converged_episode_is_frozen(self):
        pairs = torch.zeros(2, 12)
        pairs[0, 3] = .5
        pairs[0, 9] = -.5
        pairs[1, 6] = .06
        paths, _, success, _ = mppi_batched(
            self.cost, pairs, steps=20, generator=torch.Generator().manual_seed(4))
        self.assertTrue(success[0])
        self.assertEqual(len(paths[0]), 2)
        torch.testing.assert_close(paths[0][0], pairs[0:1, :6])

    def test_seeded_runs_are_reproducible(self):
        pairs = torch.zeros(1, 12)
        pairs[:, 6] = .1
        runs = [mppi_batched(self.cost, pairs, steps=7,
                             generator=torch.Generator().manual_seed(42))
                for _ in range(2)]
        torch.testing.assert_close(torch.cat(runs[0][0][0]), torch.cat(runs[1][0][0]))

    def test_custom_cost_receives_all_horizon_poses(self):
        calls = []

        def candidate_cost(rollout_pairs, current):
            self.assertEqual(tuple(rollout_pairs.shape), (1, 50, 5, 12))
            torch.testing.assert_close(
                pose_distance(current[:, None], rollout_pairs[:, :, 0, :6]),
                torch.full((1, 50), .015), atol=3e-7, rtol=0)
            calls.append(True)
            return self.cost(rollout_pairs.reshape(-1, 12)).reshape(1, 50, 5).mean(-1)

        pairs = torch.zeros(1, 12)
        pairs[:, 6] = .1
        mppi_batched(self.cost, pairs, steps=3, alternating=True,
                     candidate_cost=candidate_cost)
        self.assertEqual(len(calls), 3)

    def test_rejects_step_past_rotation_injectivity_radius(self):
        with self.assertRaises(ValueError):
            mppi_batched(self.cost, torch.zeros(1, 12), step=.5)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
