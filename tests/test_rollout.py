import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from batching import collate_groups
from rollout import HFGenerator, collect_rollout_groups, generate_rollouts


class ScriptedPolicy(torch.nn.Module):
    """Row r emits script[r][step] with probability 1 (the last entry repeats), so sampling is
    deterministic. Records the inputs of every forward call."""

    def __init__(self, script, vocab=8):
        super().__init__()
        self.script, self.vocab, self.calls = script, vocab, []

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, use_cache,
                return_dict, logits_to_keep):
        step = 0 if past_key_values is None else past_key_values
        self.calls.append(dict(
            input_ids=input_ids.tolist(), attention_mask=attention_mask.tolist(),
            position_ids=position_ids.tolist(), logits_to_keep=logits_to_keep,
            grad=torch.is_grad_enabled(), training=self.training,
        ))
        logits = torch.full((input_ids.shape[0], 1, self.vocab), -torch.inf)
        for row, tokens in enumerate(self.script[:input_ids.shape[0]]):
            logits[row, -1, tokens[min(step, len(tokens) - 1)]] = 0.0
        return SimpleNamespace(logits=logits, past_key_values=step + 1)


class RolloutTests(unittest.TestCase):
    def test_left_padding_mask_positions_and_decode_inputs(self):
        policy = ScriptedPolicy([[2]] * 4)
        generate_rollouts([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], policy, num_gens=2,
                          max_samples_per_gen=2, eos_token_ids=(), pad_token_id=0)
        first, second = policy.calls
        self.assertEqual(first["input_ids"], [[1, 2, 3], [1, 2, 3], [0, 4, 5], [0, 4, 5]])
        self.assertEqual(first["attention_mask"], [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]])
        self.assertEqual(first["position_ids"], [[0, 1, 2], [0, 1, 2], [0, 0, 1], [0, 0, 1]])
        self.assertEqual(second["input_ids"], [[2]] * 4)
        self.assertEqual(second["attention_mask"], [[1, 1, 1, 1]] * 2 + [[0, 1, 1, 1]] * 2)
        self.assertEqual(second["position_ids"], [[3], [3], [2], [2]])
        for c in policy.calls:
            self.assertEqual((c["logits_to_keep"], c["grad"], c["training"]), (1, False, False))

    def test_lengths_stop_at_first_eos_and_cap_at_limit(self):
        # row 0 stops at its second token; row 1 at its first (its later EOS must not extend it);
        # row 2 never emits EOS and is cut at the limit
        policy = ScriptedPolicy([[2, 7, 3], [7, 3, 7], [3]])
        rows = generate_rollouts([torch.tensor([1])] * 3, policy, num_gens=1,
                                 max_samples_per_gen=5, eos_token_ids=(7,))
        self.assertEqual([(r["response_ids"], r["finish_reason"]) for r in rows],
                         [([2, 7], "stop"), ([7], "stop"), ([3] * 5, "length")])
        self.assertEqual([r["behavior_logprobs"] for r in rows], [[0.0] * 2, [0.0], [0.0] * 5])
        self.assertEqual(len(policy.calls), 5)

    def test_stops_early_once_every_row_is_done(self):
        policy = ScriptedPolicy([[7], [2, 7]])
        rows = generate_rollouts([torch.tensor([1])] * 2, policy, num_gens=1,
                                 max_samples_per_gen=100, eos_token_ids=(7,))
        self.assertEqual([r["response_ids"] for r in rows], [[7], [2, 7]])
        self.assertEqual(len(policy.calls), 16)  # completion is checked every 16 steps

    def test_collection_splits_rows_into_groups_with_unpadded_prompts(self):
        tasks = [{"id": "a"}, {"id": "b"}]
        tokenizer = Mock()
        tokenizer.decode.side_effect = lambda ids, skip_special_tokens: f"text{ids}"
        score_fn = Mock(side_effect=[1.0, 0.0, 0.5, 0.25])
        kwargs = dict(
            tasks=tasks, input_prompts=[torch.tensor([[0, 1, 2]]), torch.tensor([[3, 4]])],
            generator=HFGenerator(ScriptedPolicy([[2, 7], [3, 7], [4, 7], [5, 7]]),
                                  eos_token_ids=(7,), pad_token_id=0),
            tokenizer=tokenizer, score_fn=score_fn, group_id=9, attempt_id=2, run_epoch=3,
            source_learner_step=7, num_gens=2, max_samples_per_gen=5,
        )
        with patch("rollout.time.time", return_value=123.0):
            groups = collect_rollout_groups(**kwargs)
        self.assertEqual(
            [(g.group_id, g.attempt_id, g.run_epoch, g.source_learner_step, g.created_at) for g in groups],
            [(9, 2, 3, 7, 123.0), (10, 2, 3, 7, 123.0)],
        )

        def trajectory(prompt_ids, response_ids, reward):
            return dict(prompt_ids=prompt_ids, response_ids=response_ids, behavior_logprobs=[0.0, 0.0],
                        loss_mask=[True, True], reward=reward, finish_reason="stop", debug_logp=[])

        # prompt k owns rows k*num_gens..; prompt_ids are the unpadded prompt
        self.assertEqual([[vars(t) for t in g.trajectories] for g in groups], [
            [trajectory([0, 1, 2], [2, 7], 1.0), trajectory([0, 1, 2], [3, 7], 0.0)],
            [trajectory([3, 4], [4, 7], 0.5), trajectory([3, 4], [5, 7], 0.25)],
        ])
        self.assertEqual(score_fn.call_args_list, [
            call(tasks[0], "text[2, 7]"), call(tasks[0], "text[3, 7]"),
            call(tasks[1], "text[4, 7]"), call(tasks[1], "text[5, 7]"),
        ])
        batch = collate_groups(groups, pad_token_id=0)
        self.assertEqual(batch["group_index"].tolist(), [0, 0, 1, 1])
        self.assertEqual(batch["loss_mask"].tolist(),
                         [[False] * 3 + [True] * 2] * 2 + [[False] * 2 + [True] * 2 + [False]] * 2)

        tokenizer.decode.side_effect = None
        tokenizer.decode.return_value = "text"
        one = {**kwargs, "tasks": tasks[:1], "input_prompts": kwargs["input_prompts"][:1], "num_gens": 1}
        for result, error in [(float("nan"), ValueError), (float("inf"), ValueError),
                              (RuntimeError("Verifier failed"), RuntimeError)]:
            with self.subTest(result=result), self.assertRaises(error):
                score_fn.side_effect = [result]
                collect_rollout_groups(**one)
        with self.assertRaises(ValueError):
            collect_rollout_groups(**{**kwargs, "tasks": tasks[:1]})  # one task, two prompts


class BatchingTests(unittest.TestCase):
    def test_batched_matches_unbatched_under_greedy(self):
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        )).eval()
        prompts = [torch.tensor([5, 9, 3, 11, 2]), torch.tensor([7, 1, 4])]  # the second gets left-padded
        batched = generate_rollouts(prompts, model, num_gens=2, max_samples_per_gen=12,
                                    eos_token_ids=(), pad_token_id=0, greedy=True)
        alone = [generate_rollouts([p], model, num_gens=1, max_samples_per_gen=12,
                                   eos_token_ids=(), pad_token_id=0, greedy=True)[0] for p in prompts]
        for k, ref in enumerate(alone):
            for row in batched[2 * k:2 * k + 2]:
                self.assertEqual(row["response_ids"], ref["response_ids"])
                torch.testing.assert_close(torch.tensor(row["behavior_logprobs"]),
                                           torch.tensor(ref["behavior_logprobs"]), atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
