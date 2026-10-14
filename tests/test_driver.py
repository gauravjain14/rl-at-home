import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import driver
from protocol import RolloutGroup


class DriverTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, count, **kwargs):
        events, version = [], 0

        async def collect(**args):
            self.assertEqual(args["input_prompt"], [[args["task"]]])
            self.assertEqual(args["max_samples_per_gen"], 512)
            events.append(("collect", args["group_id"], version))
            await asyncio.sleep(0)
            return RolloutGroup(args["group_id"], 0, args["run_epoch"], version, 0, [])

        async def train(groups):
            nonlocal version
            self.assertTrue(all(group.source_learner_step == version for group in groups))
            self.assertTrue(all(group.run_epoch == kwargs.get("run_epoch", 0) for group in groups))
            events.append(("train", [group.group_id for group in groups], version))
            await asyncio.sleep(0)
            version += 1
            return {"batch_size": len(groups)}

        async def evaluate(*args):
            events.append(("evaluate", version))
            return {"policy_version": version}

        async def save(path):
            events.append(("save", version))
            return path

        worker = SimpleNamespace(**{
            name: SimpleNamespace(remote=fn) for name, fn in (
                ("collect_group", collect), ("train_groups", train),
                ("evaluate", evaluate), ("save_checkpoint", save),
            )
        })  # Deliberately has no export_weights or load_weights.
        size = kwargs.get("groups_per_batch", 1)
        prepared = [(i, {"input_ids": [[i]]}) for i in range(count)]
        inputs = [prepared[i:i + size] for i in range(0, count, size)]
        with patch.object(driver, "prepare_tasks", return_value=iter(inputs)) as prepare, \
                patch("builtins.print"):
            result = await asyncio.wait_for(driver.main(worker, **kwargs), 2)
        prepare.assert_called_once_with(batch_size=size, task_name="gsm8k")
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
        result, events = await self.drive(
            3, num_train_steps=2, eval_inputs=["held-out"], checkpoint_dir="test-checkpoint",
        )
        self.assertEqual(events, [
            ("evaluate", 0), ("collect", 0, 0), ("train", [0], 0),
            ("collect", 1, 1), ("train", [1], 1), ("evaluate", 2), ("save", 2),
        ])
        self.assertEqual(result["eval_before"]["policy_version"], 0)
        self.assertEqual(result["eval_after"]["policy_version"], 2)
        self.assertEqual(result["checkpoint_dir"], "test-checkpoint")

    async def test_failures_stop_the_run(self):
        for stage in ("collect_group", "train_groups"):
            with self.subTest(stage=stage):
                worker = SimpleNamespace(**{
                    name: SimpleNamespace(remote=AsyncMock(return_value=object()))
                    for name in ("collect_group", "train_groups", "save_checkpoint")
                })
                getattr(worker, stage).remote.side_effect = RuntimeError(stage)
                with patch.object(driver, "prepare_tasks", return_value=[[({}, {"input_ids": None})]]):
                    with self.assertRaisesRegex(RuntimeError, stage):
                        await driver.main(worker, checkpoint_dir="unused")
                worker.save_checkpoint.remote.assert_not_awaited()
                if stage == "collect_group":
                    worker.train_groups.remote.assert_not_awaited()
