"""From rl-at-home/: python -B -m unittest -v test_batching"""

import unittest

import torch

from batching import collate_groups
from protocol import RolloutGroup, Trajectory


def group(name, trajectories):
    return RolloutGroup(name, 0, 0, 0, 0.0, trajectories)


class BatchingTests(unittest.TestCase):
    def test_batch_contents_and_dtypes(self):
        groups = [
            group("A", [
                # Real EOS shares ID 0 with padding; 99 is masked response context.
                Trajectory([10, 11], [12, 99, 0], [-0.25, 0.0, -1.5],
                           [True, False, True], 1.0, "stop"),
                Trajectory([10, 11], [13], [-0.75], [True], -0.5, "length"),
            ]),
            group("B", [
                Trajectory([20], [21, 22], [-0.5, -1.0], [True, True], 2.0, "stop"),
            ]),
        ]
        batch = collate_groups(groups, pad_token_id=0)
        expected = {
            "input_ids": (torch.long, [
                [10, 11, 12, 99, 0], [10, 11, 13, 0, 0], [20, 21, 22, 0, 0],
            ]),
            "attention_mask": (torch.bool, [
                [1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0],
            ]),
            "loss_mask": (torch.bool, [
                [0, 0, 1, 0, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0],
            ]),
            "behavior_logprobs": (torch.float32, [
                [0, 0, -0.25, 0, -1.5], [0, 0, -0.75, 0, 0], [0, -0.5, -1, 0, 0],
            ]),
            "rewards": (torch.float32, [1.0, -0.5, 2.0]),
            "group_index": (torch.long, [0, 0, 1]),
        }
        self.assertEqual(batch.keys(), expected.keys())
        for name, (dtype, values) in expected.items():
            with self.subTest(tensor=name):
                torch.testing.assert_close(
                    batch[name], torch.tensor(values, dtype=dtype), rtol=0, atol=0
                )

    def test_rejects_empty_or_misaligned_trajectories(self):
        cases = [
            {"prompt_ids": []},
            {"response_ids": [], "behavior_logprobs": [], "loss_mask": []},
            {"behavior_logprobs": [-0.5]},
            {"loss_mask": [True]},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                fields = dict(prompt_ids=[10], response_ids=[20, 21],
                              behavior_logprobs=[-0.5, -1.0], loss_mask=[True, True],
                              reward=1.0, finish_reason="stop")
                fields.update(changes)
                with self.assertRaises(ValueError):
                    collate_groups([group("A", [Trajectory(**fields)])], pad_token_id=0)


if __name__ == "__main__":
    unittest.main()
