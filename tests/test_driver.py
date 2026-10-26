import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import driver
from protocol import RolloutGroup


class DriverTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, count, *, eval_fail_at=None, **kwargs):
        events, version = [], 0

        async def collect(**args):
            # one batched call per step; groups are numbered from group_id
            self.assertEqual(args["input_prompts"], [[[task]] for task in args["tasks"]])
            self.assertEqual(args["max_samples_per_gen"], 512)
            groups = []
            for i in range(len(args["tasks"])):
                events.append(("collect", args["group_id"] + i, version))
                groups.append(RolloutGroup(args["group_id"] + i, 0, args["run_epoch"], version, 0, []))
            await asyncio.sleep(0)
            return groups

        async def train(groups):
            nonlocal version
            self.assertTrue(all(group.source_learner_step == version for group in groups))
            self.assertTrue(all(group.run_epoch == kwargs.get("run_epoch", 0) for group in groups))
            events.append(("train", [group.group_id for group in groups], version))
            await asyncio.sleep(0)
            version += 1
            return {"batch_size": len(groups)}

        async def evaluate(*args, **options):
            events.append(("evaluate", version))
            self.assertIs(args[0], kwargs["eval_inputs"])
            if version == eval_fail_at:
                raise RuntimeError("evaluation failed")
            scores = {"policy_version": version, "accuracy": 0.9 if version == 2 else 0.5}
            if options.get("output_path"):
                Path(options["output_path"]).write_text(json.dumps(scores))
            return scores

        async def save(path):
            events.append(("save", version))
            Path(path).mkdir(parents=True, exist_ok=False)
            return path

        worker = SimpleNamespace(**{
            name: SimpleNamespace(remote=fn) for name, fn in (
                ("collect_group", collect), ("train_groups", train),
                ("evaluate", evaluate), ("save_checkpoint", save),
            )
        })  # Deliberately has no export_weights or load_weights.
        size = kwargs.get("groups_per_batch", 16)
        prepared = [(i, {"input_ids": [[i]]}) for i in range(count)]
        inputs = [prepared[i:i + size] for i in range(0, count, size)]
        with patch("builtins.print"):
            result = await asyncio.wait_for(driver.main(worker, train_batches=iter(inputs), **kwargs), 2)
        return result, events

    async def test_sequential_batches_partial_empty_and_step_limit(self):
        for count, limit, batches in (
            (0, 10, []), (5, 10, [[0, 1], [2, 3], [4]]), (5, 2, [[0, 1], [2, 3]]),
            (0, None, []), (5, None, [[0, 1], [2, 3], [4]]),
        ):
            with self.subTest(count=count, limit=limit):
                result, events = await self.drive(
                    count, num_train_steps=limit, groups_per_batch=2, run_epoch=7,
                )
                expected = []
                for version, ids in enumerate(batches):
                    expected.extend(("collect", group_id, version) for group_id in ids)
                    expected.append(("train", ids, version))
                self.assertEqual(events, expected)
                self.assertEqual(result["learner_step"], len(batches))
                self.assertEqual(result["metrics"], {"batch_size": len(batches[-1])} if batches else {})

    async def test_evaluation_and_checkpoint_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            result, events = await self.drive(
                3, num_train_steps=2, groups_per_batch=1,
                eval_inputs=[({"question_id": 7}, {})], checkpoint_dir=root,
            )
            self.assertEqual(result["checkpoint_dir"], str(root / "step-000002"))
            self.assertEqual(json.loads((root / "validation_ids.json").read_text()), [7])
        self.assertEqual(events, [
            ("evaluate", 0), ("collect", 0, 0), ("train", [0], 0),
            ("collect", 1, 1), ("train", [1], 1), ("save", 2), ("evaluate", 2),
        ])
        self.assertEqual(result["eval_before"]["policy_version"], 0)
        self.assertEqual(result["eval_after"]["policy_version"], 2)

    async def test_periodic_final_retention_and_logging(self):
        for count in (6, 7):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                run = SimpleNamespace(log=Mock())
                result, events = await self.drive(
                    count, num_train_steps=None, groups_per_batch=1, checkpoint_every=2,
                    eval_inputs=[({"question_id": 7}, {})], checkpoint_dir=root, wandb_run=run,
                )
                steps = [2, 4, 6] + ([7] if count == 7 else [])
                expected = [("evaluate", 0)]
                for step in steps:
                    expected.extend([("save", step), ("evaluate", step)])
                self.assertEqual([event for event in events if event[0] in ("save", "evaluate")], expected)
                kept = {2, *steps[-2:]}
                self.assertEqual({path.name for path in root.glob("step-*")},
                                 {f"step-{step:06d}" for step in kept})
                self.assertEqual(result["best_checkpoint_dir"], str(root / "step-000002"))
                self.assertEqual(result["eval_after"]["policy_version"], count)
                self.assertEqual(len(list(root.glob("eval-*.json"))), len(steps) + 1)
                logged_evals = [call.args[0]["learner_step"] for call in run.log.call_args_list
                                if "eval/accuracy" in call.args[0]]
                self.assertEqual(logged_evals, [0, *steps])

    async def test_default_sixteen_prompt_batch_and_partial_tail(self):
        result, events = await self.drive(17, num_train_steps=None)
        trained = [event[1] for event in events if event[0] == "train"]
        self.assertEqual(trained, [list(range(16)), [16]])
        self.assertEqual(result["learner_step"], 2)

    async def test_checkpoint_survives_evaluation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                await self.drive(2, num_train_steps=None, groups_per_batch=1, checkpoint_every=2,
                                 eval_inputs=[({"question_id": 7}, {})], checkpoint_dir=root,
                                 eval_fail_at=2)
            self.assertTrue((root / "step-000002" / "run_config.json").is_file())

    async def test_failures_stop_the_run(self):
        for stage in ("collect_group", "train_groups"):
            with self.subTest(stage=stage):
                worker = SimpleNamespace(**{
                    name: SimpleNamespace(remote=AsyncMock(return_value=[object()]))
                    for name in ("collect_group", "train_groups", "save_checkpoint")
                })
                getattr(worker, stage).remote.side_effect = RuntimeError(stage)
                with self.assertRaisesRegex(RuntimeError, stage):
                    await driver.main(worker, train_batches=[[({}, {"input_ids": None})]])
                worker.save_checkpoint.remote.assert_not_awaited()
                if stage == "collect_group":
                    worker.train_groups.remote.assert_not_awaited()
