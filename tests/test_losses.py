"""From rl-at-home/: python -B -m unittest discover -s tests -p test_losses.py -v"""

import math
import unittest

import torch
import torch.nn.functional as F

from losses import clipped_policy_loss, group_advantages, token_logprobs


class TokenLogprobsTests(unittest.TestCase):
    def test_shift_selection_and_masked_gradients(self):
        logits = torch.tensor([
            [[1., 2., 3.], [3., 2., 1.], [99., -99., 10.]],
            [[1., 2., 3.], [3., 2., 1.], [-99., 99., 10.]],
        ], requires_grad=True)
        ids = torch.tensor([[0, 2, 1], [1, 0, 2]])
        actual = token_logprobs(logits, ids)
        log_normalizer = math.log(math.exp(1) + math.exp(2) + math.exp(3))
        expected = torch.tensor([[3., 2.], [1., 1.]]) - log_normalizer
        torch.testing.assert_close(actual, expected)

        # Full-sequence masks are shifted to the same targets as the scores.
        mask = torch.tensor([[False, True, False], [False, False, True]])[:, 1:]
        reference = -F.cross_entropy(
            logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none"
        )
        expected_grad, = torch.autograd.grad(-reference[mask].sum(), logits)
        (-actual[mask].sum()).backward()
        torch.testing.assert_close(logits.grad, expected_grad)
        active = torch.cat([mask, torch.zeros(2, 1, dtype=torch.bool)], dim=1)
        self.assertTrue((logits.grad[~active] == 0).all())
        self.assertTrue((logits.grad[active].abs().sum(-1) > 0).all())

    def test_float32_normalization_for_low_precision_inputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor(
                    [[[1000., 1005., -1000.], [0., 0., 0.]]],
                    dtype=dtype, requires_grad=True,
                )
                actual = token_logprobs(logits, torch.tensor([[0, 2]]))
                self.assertEqual(actual.shape, (1, 1))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertLess(actual.item(), -1000)
                actual.sum().backward()
                self.assertTrue(torch.isfinite(logits.grad).all())


class GroupAdvantagesTests(unittest.TestCase):
    def test_grouping_and_optional_normalization(self):
        rewards = torch.tensor([1., 10., 3., 14.], requires_grad=True)
        original = rewards.detach().clone()
        group_index = torch.tensor([8, 3, 8, 3])
        centered = torch.tensor([-1., -2., 1., 2.])
        for normalize in (True, False):
            with self.subTest(normalize=normalize):
                actual = group_advantages(rewards, group_index, normalize_std=normalize)
                expected = centered
                if normalize:
                    expected = centered / (torch.tensor([1., 2., 1., 2.]) + 1e-6)
                torch.testing.assert_close(actual, expected)
                self.assertFalse(actual.requires_grad)
                torch.testing.assert_close(rewards, original)

    def test_constant_singleton_and_low_precision_rewards(self):
        cases = [
            (torch.tensor([5., 9., 5.]), torch.tensor([0, 1, 0])),
            (torch.tensor([5., 5.], dtype=torch.float16), torch.tensor([0, 0])),
        ]
        for rewards, group_index in cases:
            for normalize in (True, False):
                with self.subTest(dtype=rewards.dtype, normalize=normalize):
                    actual = group_advantages(rewards, group_index, normalize_std=normalize)
                    torch.testing.assert_close(actual, torch.zeros_like(rewards, dtype=torch.float32))


class ClippedPolicyLossTests(unittest.TestCase):
    def test_defaults_ratio_one_and_detached_targets(self):
        current, behavior, proximal = [
            torch.full((2, 1), -1., requires_grad=True) for _ in range(3)
        ]
        advantages = torch.tensor([1., -1.], requires_grad=True)
        loss = clipped_policy_loss(
            current, behavior, proximal, advantages, torch.ones(2, 1, dtype=torch.bool)
        )
        torch.testing.assert_close(loss, torch.tensor(0.))
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[-.5], [.5]]))
        for fixed in (behavior, proximal, advantages):
            self.assertIsNone(fixed.grad)

    def test_correction_clipping_and_masked_gradients(self):
        current = torch.tensor([
            [math.log(.3), math.log(.1), math.log(.2)],
            [math.log(.1), math.log(.3), 1000.],
        ], requires_grad=True)
        proximal = torch.full((2, 3), math.log(.2))
        behavior = torch.tensor([[.2 / 3, .4, .2], [.8, .08, float("nan")]]).log()
        mask = torch.tensor([[True, True, True], [True, True, False]])
        loss = clipped_policy_loss(
            current, behavior, proximal, torch.tensor([1., -1.]), mask, .2, 2.
        )
        # Five selected tokens: -(2*1.2 + .5*.5 + 1 - .25*.8 - 2*1.5) / 5.
        torch.testing.assert_close(loss, torch.tensor(-.09))
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[0., -.05, -.2], [0., .6, 0.]]))

    def test_precision_and_invalid_inputs(self):
        mask = torch.ones(1, 1, dtype=torch.bool)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                current = torch.zeros(1, 1, dtype=dtype, requires_grad=True)
                old = torch.full((1, 1), -12., dtype=dtype)
                loss = clipped_policy_loss(current, old, old, torch.ones(1, dtype=dtype), mask, .2, 2.)
                torch.testing.assert_close(loss, torch.tensor(-1.2))
                loss.backward()
                torch.testing.assert_close(current.grad, torch.zeros_like(current))

        scores = torch.full((1, 1), -1.)
        arguments = [scores, scores, scores, torch.ones(1), mask, .2, 2.]
        for index, value in [
            (0, torch.tensor([[float("nan")]])),
            (1, torch.tensor([[float("inf")]])),
            (2, torch.tensor([[float("inf")]])),
            (2, torch.tensor([[-float("inf")]])),
            (4, ~mask), (5, -.1), (5, 1.),
            (6, 0.), (6, -1.), (6, float("inf")), (6, float("nan")),
        ]:
            with self.subTest(argument=index, value=value):
                invalid = arguments.copy()
                invalid[index] = value
                with self.assertRaises((ValueError, FloatingPointError)):
                    clipped_policy_loss(*invalid)


if __name__ == "__main__":
    unittest.main()
