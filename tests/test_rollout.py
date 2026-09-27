import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from batching import collate_groups
from rollout import collect_rollout_groups, generate_rollouts


class ScriptedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, past_key_values, use_cache, return_dict):
        self.calls.append(
            (input_ids.tolist(), past_key_values, use_cache, return_dict,
             torch.is_grad_enabled(), self.training)
        )
        position = 0 if past_key_values is None else past_key_values
        logits = torch.full((*input_ids.shape, 4), -torch.inf)
        logits[:, -1, (2, 3)[position]] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=position + 1)


class RolloutTests(unittest.TestCase):
    def test_tokens_scores_stopping_and_cache_reset(self):
        prompt = torch.tensor([[0, 1]])
        for eos_ids, limit, expected_ids, reason in (
            ((3,), 5, [2, 3], "stop"),
            ((2,), 5, [2], "stop"),
            ((), 2, [2, 3], "length"),
        ):
            with self.subTest(eos_ids=eos_ids):
                policy = ScriptedPolicy()
                result = generate_rollouts(
                    prompt, policy, num_gens=2,
                    max_samples_per_gen=limit, eos_token_ids=eos_ids,
                )
                expected = {
                    "response_ids": expected_ids,
                    "behavior_logprobs": [0.0] * len(expected_ids),
                    "finish_reason": reason,
                }
                self.assertEqual(result, [expected, expected])
                calls_per_response = [([[0, 1]], None, True, True, False, False)]
                if len(expected_ids) == 2:
                    calls_per_response.append(([[2]], 1, True, True, False, False))
                self.assertEqual(policy.calls, calls_per_response * 2)
                self.assertEqual(prompt.tolist(), [[0, 1]])

    def test_collection_scores_and_packages_one_complete_group(self):
        task = {"prompt": "Compose a tune"}
        tokenizer = Mock()
        tokenizer.decode.side_effect = ["first tune", "second tune"]
        score_fn = Mock(side_effect=[1.0, 0.25])
        kwargs = dict(
            task=task, input_prompt=torch.tensor([[0, 1]]),
            behavior_policy=ScriptedPolicy(), tokenizer=tokenizer,
            score_fn=score_fn, group_id=9, attempt_id=2, run_epoch=3,
            source_learner_step=7, num_gens=2, max_samples_per_gen=5,
            eos_token_ids=(3,),
        )
        with patch("rollout.time.time", return_value=123.0):
            group = collect_rollout_groups(**kwargs)
        self.assertEqual(
            (group.group_id, group.attempt_id, group.run_epoch,
             group.source_learner_step, group.created_at),
            (9, 2, 3, 7, 123.0),
        )
        self.assertEqual(len(group.trajectories), 2)
        for trajectory, reward in zip(group.trajectories, [1.0, 0.25]):
            self.assertEqual(vars(trajectory), dict(
                prompt_ids=[0, 1], response_ids=[2, 3],
                behavior_logprobs=[0.0, 0.0], loss_mask=[True, True],
                reward=reward, finish_reason="stop",
            ))
        self.assertEqual(score_fn.call_args_list, [call(task, "first tune"), call(task, "second tune")])
        self.assertEqual(tokenizer.decode.call_args_list, [call([2, 3], skip_special_tokens=True)] * 2)
        batch = collate_groups([group], pad_token_id=0)
        self.assertEqual(batch["group_index"].tolist(), [0, 0])
        self.assertEqual(batch["loss_mask"].tolist(), [[False, False, True, True]] * 2)

        tokenizer.decode.side_effect = None
        tokenizer.decode.return_value = "tune"
        for result, error in [(float("nan"), ValueError), (float("inf"), ValueError),
                              (RuntimeError("Verifier failed"), RuntimeError)]:
            with self.subTest(result=result), self.assertRaises(error):
                score_fn.side_effect = [result]
                collect_rollout_groups(**{**kwargs, "num_gens": 1})


if __name__ == "__main__":
    unittest.main()
