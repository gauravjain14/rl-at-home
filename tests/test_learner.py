"""From rl-at-home/: python -B -m unittest discover -s tests -p test_learner.py -v"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from learner import score_batch, score_proximal, train_step


class ScoreBatchTests(unittest.TestCase):
    def test_forward_contract_and_gradient_modes(self):
        logits = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5).requires_grad_()
        model = Mock(return_value=SimpleNamespace(logits=logits))
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 4, 0]]),
            "attention_mask": torch.tensor([[True, True, True], [True, True, False]]),
        }
        scores = score_batch(model, batch)
        model.assert_called_once_with(**batch, use_cache=False, return_dict=True)
        expected = -F.cross_entropy(
            logits[:, :-1].transpose(1, 2), batch["input_ids"][:, 1:], reduction="none"
        )
        torch.testing.assert_close(scores, expected)
        scores.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.)
        with torch.no_grad():
            cached = score_batch(model, batch)
        self.assertFalse(cached.requires_grad)
        torch.testing.assert_close(cached, scores.detach())


class TinyPolicy(torch.nn.Module):
    """One trainable next-token distribution per input token; no downloads."""

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(8, 8))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=10 * self.logits[input_ids])


class TrainStepTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyPolicy().eval()
        self.optimizer = SimpleNamespace(
            zero_grad=self.model.zero_grad, step=Mock(side_effect=self.sgd_step)
        )
        self.batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 5, 0]]),
            "attention_mask": torch.tensor([[True, True, True, True], [True, True, True, False]]),
            "loss_mask": torch.tensor([[False, False, True, True], [False, False, True, False]]),
            "rewards": torch.tensor([1., 0.]),
            "group_index": torch.tensor([0, 0]),
        }
        with torch.no_grad():
            self.proximal = score_batch(self.model, self.batch)
        self.batch["behavior_logprobs"] = torch.cat([torch.zeros(2, 1), self.proximal.clone()], dim=1)
        self.step_grad_norms = []

    @torch.no_grad()
    def sgd_step(self):
        self.step_grad_norms.append(self.model.logits.grad.norm().item())
        self.model.logits.add_(self.model.logits.grad, alpha=-.001)

    def test_updates_clear_gradients_and_preserve_proximal(self):
        original_proximal = self.proximal.clone()
        self.model.logits.grad = torch.full_like(self.model.logits, float("inf"))
        for step_count in (1, 2):
            before = self.model.logits.detach().clone()
            metrics = train_step(self.model, self.optimizer, self.batch, self.proximal)
            self.assertEqual(self.optimizer.step.call_count, step_count)
            self.assertFalse(torch.equal(self.model.logits, before))
            torch.testing.assert_close(self.proximal, original_proximal, rtol=0, atol=0)
            for key in ("loss", "grad_norm"):
                self.assertIsInstance(metrics[key], float)
                self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))
            if step_count == 1:
                self.assertGreater(metrics["grad_norm"], 1.)
            self.assertLessEqual(self.step_grad_norms[-1], 1.00001)
            self.assertIsNone(self.model.logits.grad)

    def test_equal_rewards_skip_the_optimizer(self):
        self.batch["rewards"] = torch.tensor([1., 1.])
        before = self.model.logits.detach().clone()
        metrics = train_step(self.model, self.optimizer, self.batch, self.proximal)
        self.assertEqual(metrics, {"loss": 0.0, "grad_norm": 0.0})
        self.optimizer.step.assert_not_called()
        torch.testing.assert_close(self.model.logits, before, rtol=0, atol=0)

    def test_nonfinite_gradient_rejects_before_optimizer(self):
        before = self.model.logits.detach().clone()
        hook = self.model.logits.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        self.addCleanup(hook.remove)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            train_step(self.model, self.optimizer, self.batch, self.proximal)
        self.optimizer.step.assert_not_called()
        torch.testing.assert_close(self.model.logits, before, rtol=0, atol=0)

    def test_microbatches_match_full_batch_with_unequal_lengths(self):
        batch = {key: value[[0, 1, 0]] for key, value in self.batch.items()}
        proximal = self.proximal[[0, 1, 0]]
        results = []
        for size in (3, 1, 2):
            model = TinyPolicy()
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            gradients = []
            optimizer.register_step_pre_hook(lambda opt, args, kwargs: gradients.append(model.logits.grad.clone()))
            metrics = train_step(model, optimizer, batch, proximal, micro_batch_size=size)
            results.append((model.logits.detach().clone(), gradients[0], metrics))
        expected_weights, expected_grad, expected_metrics = results[0]
        for weights, grad, metrics in results[1:]:
            torch.testing.assert_close(weights, expected_weights)
            torch.testing.assert_close(grad, expected_grad)
            self.assertAlmostEqual(metrics["loss"], expected_metrics["loss"], places=6)
            self.assertAlmostEqual(metrics["grad_norm"], expected_metrics["grad_norm"], places=5)


class PPOStepTests(unittest.TestCase):
    def setUp(self):
        rows = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 5, 0]]),
            "attention_mask": torch.tensor([[True, True, True, True], [True, True, True, False]]),
            "loss_mask": torch.tensor([[False, False, True, True], [False, False, True, False]]),
        }
        # two groups of two rows, each with a reward spread
        self.batch = {key: value[[0, 1, 0, 1]] for key, value in rows.items()}
        self.batch["rewards"] = torch.tensor([1., 0., 0., 1.])
        self.batch["group_index"] = torch.tensor([0, 0, 1, 1])
        with torch.no_grad():
            self.proximal = score_batch(TinyPolicy(), self.batch)
        self.batch["behavior_logprobs"] = torch.cat([torch.zeros(4, 1), self.proximal.clone()], dim=1)

    def test_minibatches_step_once_each_against_the_frozen_proximal(self):
        model = TinyPolicy()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        steps = []
        optimizer.register_step_post_hook(lambda *args: steps.append(1))
        with patch("learner.torch.randperm", side_effect=torch.arange):
            metrics = train_step(model, optimizer, self.batch, self.proximal, update_epochs=2, num_minibatches=2)
        self.assertEqual(len(steps), 4)
        # reference: the same four steps, one whole group each. A group's advantages do not
        # depend on the rest of the batch, so train_step on that group alone is the same step.
        reference = TinyPolicy()
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
        ref_metrics = []
        for _ in range(2):
            for rows in ([0, 1], [2, 3]):
                part = {key: value[rows] for key, value in self.batch.items()}
                ref_metrics.append(train_step(reference, ref_optimizer, part, self.proximal[rows]))
        torch.testing.assert_close(model.logits, reference.logits)
        self.assertAlmostEqual(metrics["loss"], sum(m["loss"] for m in ref_metrics) / 4, places=6)
        self.assertAlmostEqual(metrics["grad_norm"], sum(m["grad_norm"] for m in ref_metrics) / 4, places=5)

    def test_score_proximal_matches_one_full_batch_pass(self):
        model = TinyPolicy()
        with torch.no_grad():
            model.logits.normal_()
            full = score_batch(model, self.batch)
        for size in (1, 3, 4):
            proximal = score_proximal(model, self.batch, size)
            self.assertFalse(proximal.requires_grad)
            torch.testing.assert_close(proximal, full)

    def test_invalid_update_settings(self):
        for kwargs in ({"update_epochs": 0}, {"num_minibatches": 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                train_step(TinyPolicy(), Mock(), self.batch, self.proximal, **kwargs)


if __name__ == "__main__":
    unittest.main()
