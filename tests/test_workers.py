import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from batching import collate_groups
from tasks import gsm8k
from learner import score_batch, score_proximal, train_step
from protocol import RolloutGroup, Trajectory
from test_learner import TinyPolicy
from workers import LearnerWorker, RolloutWorker, SyncWorker


class WorkerTests(unittest.TestCase):
    def test_sync_worker_shares_model_and_tracks_updates(self):
        cls = SyncWorker.__ray_metadata__.modified_class
        model = TinyPolicy()
        tokenizer = Mock(eos_token_id=7)
        for eos, expected in ((None, (7,)), (6, (6,)), ([6, 7], (6, 7))):
            model.generation_config = SimpleNamespace(eos_token_id=eos)
            worker = cls(model, "cpu", pad_token_id=0, tokenizer=tokenizer,
                         score_fn=gsm8k.score, num_gens=2, lr=2e-6)
            self.assertEqual(worker.eos_token_ids, expected)
        self.assertIs(worker.model, model)
        self.assertIs(worker.optimizer.param_groups[0]["params"][0], model.logits)
        self.assertEqual(worker.optimizer.param_groups[0]["lr"], 2e-6)
        group = RolloutGroup(0, 0, 0, 0, 0, [
            Trajectory([1, 2], [3], [-math.log(8)], [True], 1.0, "stop"),
            Trajectory([1, 2], [4], [-math.log(8)], [True], 0.0, "length"),
        ])
        prompt = torch.tensor([[1, 2]])
        task = {"answer": "42"}
        with patch("workers.collect_rollout_groups", return_value=[group]) as collect:
            self.assertEqual(worker.collect_group([task], [prompt], group_id=0, run_epoch=0), [group])
            self.assertEqual(collect.call_args.kwargs["source_learner_step"], 0)
            self.assertEqual(collect.call_args.kwargs["tasks"], [task])
            self.assertEqual(collect.call_args.kwargs["pad_token_id"], 0)
            metrics = worker.train_groups([group])
            self.assertGreater(metrics["grad_norm"], 0)
            self.assertGreater(model.logits.abs().sum().item(), 0)
            self.assertAlmostEqual(metrics["logprob_gap_mean"], 0.0, places=6)
            self.assertAlmostEqual(metrics["sampled_nll"], math.log(8), places=6)
            self.assertEqual(metrics["response_len"], 1.0)
            worker.collect_group([task], [prompt], group_id=1, run_epoch=0)
            for key, expected in (("behavior_policy", model), ("tokenizer", tokenizer),
                                  ("score_fn", gsm8k.score), ("source_learner_step", 1),
                                  ("num_gens", 2), ("eos_token_ids", (6, 7))):
                self.assertEqual(collect.call_args.kwargs[key], expected)
        tokenizer.decode.return_value = r"\boxed{42}"
        with patch("workers.generate_rollouts", return_value=[{"response_ids": [3, 7]}]):
            self.assertEqual(worker.evaluate([(task, {"input_ids": prompt})]), {
                "accuracy": 1.0, "truncated_fraction": 0.0, "num_examples": 1, "policy_version": 1,
            })
        model.save_pretrained = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            self.assertEqual(worker.save_checkpoint(path), str(path))
            model.save_pretrained.assert_called_once_with(path)
            tokenizer.save_pretrained.assert_called_once_with(path)
            saved = torch.load(path / "trainer.pt", weights_only=True)
            self.assertEqual(saved["learner_step"], 1)
            self.assertTrue(saved["optimizer"]["state"])

    def test_sixteen_groups_of_eight_match_full_batch_update(self):
        model = TinyPolicy()
        model.generation_config = SimpleNamespace(eos_token_id=7)
        cls = SyncWorker.__ray_metadata__.modified_class
        worker = cls(model, "cpu", 0, tokenizer=Mock(), score_fn=gsm8k.score)
        self.assertEqual(worker.num_gens, 8)
        worker.optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        groups = [RolloutGroup(g, 0, 0, 0, 0, [
            Trajectory([1, 2], [3, 4] if j % 2 else [5],
                       [-math.log(8)] * (2 if j % 2 else 1),
                       [True] * (2 if j % 2 else 1), float(j <= g % 7), "stop")
            for j in range(8)
        ]) for g in range(16)]
        expected_model = TinyPolicy()
        batch = collate_groups(groups, 0)
        with torch.no_grad():
            proximal = score_batch(expected_model, batch)
        expected = train_step(expected_model, torch.optim.SGD(expected_model.parameters(), lr=.001),
                              batch, proximal, micro_batch_size=128)
        with patch.object(model, "forward", wraps=model.forward) as forward:
            actual = worker.train_groups(groups)
        self.assertEqual(len(forward.call_args_list), 256)
        self.assertTrue(all(call.kwargs["input_ids"].shape[0] == 1 for call in forward.call_args_list))
        torch.testing.assert_close(model.logits, expected_model.logits)
        self.assertAlmostEqual(actual["loss"], expected["loss"], places=6)
        self.assertAlmostEqual(actual["grad_norm"], expected["grad_norm"], places=5)

    def test_sync_evaluation_writes_answers_and_parser_failures(self):
        cls = SyncWorker.__ray_metadata__.modified_class
        model = TinyPolicy()
        model.generation_config = SimpleNamespace(eos_token_id=7)
        tokenizer = Mock()
        worker = cls(model, "cpu", 0, tokenizer=tokenizer, score_fn=gsm8k.score,
                     answer_parser=gsm8k.parse)
        rollouts = [{"response_ids": [3, 7]}, {"response_ids": [4, 4]}]
        tokenizer.decode.side_effect = [r"\boxed{42}", "No final answer"]
        task = {"question_id": 9, "answer": "42", "prompt": [{"role": "user", "content": "Question"}]}
        inputs = [(task, {"input_ids": torch.tensor([[1, 2]])})] * 2
        with tempfile.TemporaryDirectory() as directory, \
                patch("workers.generate_rollouts", return_value=rollouts) as generate:
            path = Path(directory) / "eval.json"
            metrics = worker.evaluate(inputs, 2, output_path=path)
            data = json.loads(path.read_text())
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["parse_failure_fraction"], .5)
        self.assertEqual(metrics["truncated_fraction"], .5)
        self.assertEqual(data["samples"][0]["parsed_answer"], "42")
        self.assertIsNone(data["samples"][1]["parsed_answer"])
        self.assertEqual(data["samples"][1]["finish_reason"], "length")
        # one batched greedy call over all prompts
        generate.assert_called_once()
        self.assertEqual([p.tolist() for p in generate.call_args.args[0]], [[1, 2], [1, 2]])
        self.assertEqual({k: generate.call_args.kwargs[k] for k in ("num_gens", "max_samples_per_gen", "greedy")},
                         {"num_gens": 1, "max_samples_per_gen": 2, "greedy": True})

    def test_update_snapshot_sync_metrics_and_checkpoint(self):
        learner_cls = LearnerWorker.__ray_metadata__.modified_class
        rollout_cls = RolloutWorker.__ray_metadata__.modified_class
        learner = learner_cls(TinyPolicy(), "cpu", pad_token_id=0)
        rollout = rollout_cls.__new__(rollout_cls)
        rollout.model, rollout.device, rollout.policy_version = TinyPolicy(), "cpu", 0
        initial = learner.export_weights()
        group = RolloutGroup(0, 0, 0, 0, 0, [
            Trajectory([1, 2], [3], [-math.log(8)], [True], 1.0, "stop"),
            Trajectory([1, 2], [4], [-math.log(8)], [True], 0.0, "length"),
        ])
        metrics = learner.train_groups([group])
        self.assertEqual(learner.learner_step, 1)
        self.assertEqual(metrics["reward_mean"], 0.5)
        self.assertEqual(metrics["equal_reward_group_fraction"], 0.0)
        self.assertEqual(metrics["truncated_fraction"], 0.5)
        self.assertEqual(metrics["policy_lag_max"], 0)
        self.assertGreater(metrics["grad_norm"], 0)
        self.assertEqual(initial["state_dict"]["logits"].abs().sum().item(), 0)
        self.assertGreater(learner.model.logits.abs().sum().item(), 0)
        with patch("workers.torch.cuda.synchronize") as synchronize:
            self.assertEqual(rollout.load_weights(learner.export_weights()), 1)
            self.assertEqual(synchronize.call_count, 2)
            torch.testing.assert_close(rollout.model.logits, learner.model.logits)
            with self.assertRaises(ValueError):
                rollout.load_weights(initial)
            with self.assertRaises(RuntimeError):
                rollout.load_weights({"learner_step": 2, "state_dict": {}})
            self.assertEqual(rollout.policy_version, 1)
        for trajectory in group.trajectories:
            trajectory.reward = 1.0
        self.assertEqual(learner.train_groups([group])["equal_reward_group_fraction"], 1.0)
        learner.model.save_pretrained = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            self.assertEqual(learner.save_checkpoint(path), str(path))
            learner.model.save_pretrained.assert_called_once_with(path)
            saved = torch.load(path / "trainer.pt", weights_only=True)
            self.assertEqual(saved["learner_step"], 2)
            self.assertTrue(saved["optimizer"]["state"])

    def test_evaluation_scores_only_generated_tokens_without_sampling(self):
        cls = RolloutWorker.__ray_metadata__.modified_class
        worker = cls.__new__(cls)
        worker.device, worker.policy_version, worker.eos_token_ids = "cpu", 7, (9,)
        worker.model = Mock()
        worker.model.generate.side_effect = [torch.tensor([[1, 2, 3, 9]]), torch.tensor([[1, 2, 4, 4]])]
        worker.tokenizer = Mock(pad_token_id=0)
        worker.tokenizer.decode.side_effect = [r"\boxed{42}", "#### 4"]
        worker.score_fn = gsm8k.score
        inputs = [({"answer": "42"}, {"input_ids": torch.tensor([[1, 2]])})] * 2
        self.assertEqual(worker.evaluate(inputs, 2), {
            "accuracy": 0.5, "truncated_fraction": 0.5, "num_examples": 2, "policy_version": 7,
        })
        worker.tokenizer.decode.assert_any_call([3, 9], skip_special_tokens=True)
        for call in worker.model.generate.call_args_list:
            self.assertFalse(call.kwargs["do_sample"])
            self.assertEqual(call.kwargs["max_new_tokens"], 2)

    def test_workers_pass_update_settings_to_the_learner(self):
        group = RolloutGroup(0, 0, 0, 0, 0, [
            Trajectory([1, 2], [3], [-math.log(8)], [True], 1.0, "stop"),
            Trajectory([1, 2], [4], [-math.log(8)], [True], 0.0, "length"),
        ])
        settings = {"micro_batch_size": 2, "ppo_epochs": 3, "num_minibatches": 2}
        model = TinyPolicy()
        model.generation_config = SimpleNamespace(eos_token_id=7)
        sync = SyncWorker.__ray_metadata__.modified_class(
            model, "cpu", 0, tokenizer=Mock(), score_fn=gsm8k.score, **settings)
        learner = LearnerWorker.__ray_metadata__.modified_class(TinyPolicy(), "cpu", 0, **settings)
        for worker in (sync, learner):
            with self.subTest(worker=type(worker).__name__), \
                    patch("workers.score_proximal", wraps=score_proximal) as proximal, \
                    patch("workers.train_step", wraps=train_step) as step:
                worker.train_groups([group])
                self.assertEqual(proximal.call_args.args[2], 2)
                self.assertEqual({k: step.call_args.kwargs[k] for k in settings}, settings)

    def test_rollout_worker_collects_batched_groups(self):
        cls = RolloutWorker.__ray_metadata__.modified_class
        worker = cls.__new__(cls)
        worker.device, worker.policy_version, worker.num_gens, worker.eos_token_ids = "cpu", 3, 4, (9,)
        worker.model, worker.tokenizer, worker.score_fn = TinyPolicy(), Mock(pad_token_id=5), gsm8k.score
        group = RolloutGroup(7, 0, 1, 3, 0, [])
        with patch("workers.collect_rollout_groups", return_value=[group]) as collect:
            self.assertEqual(worker.collect_group([{"answer": "1"}], [torch.tensor([[1, 2]])],
                                                  group_id=7, run_epoch=1), [group])
        kwargs = collect.call_args.kwargs
        self.assertEqual(
            {k: kwargs[k] for k in ("tasks", "group_id", "run_epoch", "source_learner_step",
                                    "num_gens", "eos_token_ids", "pad_token_id")},
            {"tasks": [{"answer": "1"}], "group_id": 7, "run_epoch": 1, "source_learner_step": 3,
             "num_gens": 4, "eos_token_ids": (9,), "pad_token_id": 5},
        )
        self.assertEqual([p.tolist() for p in kwargs["input_prompts"]], [[[1, 2]]])

    def test_gradient_checkpointing_gives_the_same_update(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(0)
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                            num_hidden_layers=2, num_attention_heads=2,
                                            num_key_value_heads=2, max_position_embeddings=64))
        group = RolloutGroup(0, 0, 0, 0, 0, [
            Trajectory([1, 2, 3], [4, 5], [-1.0, -1.0], [True, True], 1.0, "stop"),
            Trajectory([1, 2, 3], [6], [-1.0], [True], 0.0, "length"),
        ])
        cls = SyncWorker.__ray_metadata__.modified_class
        updated = []
        for checkpointing in (False, True):
            model = copy.deepcopy(base)
            worker = cls(model, "cpu", 0, tokenizer=Mock(eos_token_id=31), score_fn=gsm8k.score,
                         gradient_checkpointing=checkpointing)
            self.assertEqual(model.is_gradient_checkpointing, checkpointing)
            worker.train_groups([group])
            self.assertEqual(model.training, checkpointing)  # train mode only when checkpointing
            updated.append([p.detach().clone() for p in model.parameters()])
        for plain, checkpointed in zip(*updated):
            torch.testing.assert_close(plain, checkpointed)

    def test_load_trainer_state_restores_what_save_checkpoint_wrote(self):
        cls = SyncWorker.__ray_metadata__.modified_class
        group = RolloutGroup(0, 0, 0, 0, 0, [
            Trajectory([1, 2], [3], [-math.log(8)], [True], 1.0, "stop"),
            Trajectory([1, 2], [4], [-math.log(8)], [True], 0.0, "length"),
        ])

        def make_worker():
            model = TinyPolicy()
            model.generation_config = SimpleNamespace(eos_token_id=7)
            model.save_pretrained = Mock()
            return cls(model, "cpu", 0, tokenizer=Mock(), score_fn=gsm8k.score)

        trained, resumed = make_worker(), make_worker()
        trained.train_groups([group])
        trained.train_groups([group])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            trained.save_checkpoint(path)
            self.assertEqual(resumed.load_trainer_state(path), 2)
        self.assertEqual(resumed.learner_step, 2)
        saved, loaded = trained.optimizer.state_dict(), resumed.optimizer.state_dict()
        self.assertEqual(saved["param_groups"], loaded["param_groups"])
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(saved["state"][0][key], loaded["state"][0][key])
