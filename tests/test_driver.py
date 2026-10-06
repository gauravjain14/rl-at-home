import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import driver
from protocol import RolloutGroup


async def collect(*, task, input_prompt, group_id, run_epoch, max_samples_per_gen):
    return RolloutGroup(group_id, 0, run_epoch, 0, 0, [])


class DriverTests(unittest.IsolatedAsyncioTestCase):
    async def drive(self, count, generate, train, **kwargs):
        prepared = [(i, {"input_ids": [[i]]}) for i in range(count)]
        inputs = [prepared[i:i + 2] for i in range(0, count, 2)]
        version = 0

        async def update(groups):
            nonlocal version
            metrics = await train(groups)
            version += 1
            return metrics

        async def load(snapshot):
            self.assertEqual(snapshot["learner_step"], version)
            return version

        rollout_worker = Mock(
            collect_group=Mock(remote=generate), load_weights=Mock(remote=AsyncMock(side_effect=load)),
            evaluate=Mock(remote=AsyncMock(side_effect=lambda *args: {"policy_version": version})),
        )
        learner_worker = Mock(
            train_groups=Mock(remote=update),
            export_weights=Mock(remote=lambda: {"learner_step": version}),
            save_checkpoint=Mock(remote=AsyncMock(side_effect=lambda path: path)),
        )
        with patch.object(driver, "prepare_tasks", return_value=iter(inputs)), \
                patch("builtins.print"):
            result = await asyncio.wait_for(
                driver.main(rollout_worker, learner_worker, **kwargs), 2,
            )
        self.assertEqual(rollout_worker.load_weights.remote.await_count, version + 1)
        return result

    async def test_group_delivery_partial_batch_and_empty_input(self):
        for count in (0, 5):
            with self.subTest(count=count):
                batches = []

                async def generate(**kwargs):
                    self.assertEqual(kwargs["input_prompt"], [[kwargs["task"]]])
                    self.assertEqual(kwargs["max_samples_per_gen"], 512)
                    return await collect(**kwargs)

                async def train(groups):
                    batches.append([group.group_id for group in groups])
                    self.assertTrue(all(group.run_epoch == 7 for group in groups))
                    return {"batch_size": len(groups)}

                result = await self.drive(
                    count, generate, train, groups_per_batch=2, run_epoch=7,
                )
                self.assertEqual(batches, [[0, 1], [2, 3], [4]] if count else [])
                self.assertEqual(result["learner_step"], len(batches))
                self.assertEqual(result["metrics"], {"batch_size": 1} if count else {})
                self.assertEqual(result["buffer_stats"]["accepted"], count)

    async def test_evaluation_uses_initial_and_final_synced_versions(self):
        result = await self.drive(
            2, collect, AsyncMock(return_value={"loss": 0.5}),
            eval_inputs=["held-out prompt"], checkpoint_dir="test-checkpoint",
        )
        self.assertEqual(result["eval_before"]["policy_version"], 0)
        self.assertEqual(result["eval_after"]["policy_version"], 2)
        self.assertEqual(result["checkpoint_dir"], "test-checkpoint")

    async def test_overlap_and_policy_staleness(self):
        training, generated = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            if kwargs["group_id"] == 1:
                await training.wait()
                generated.set()
            return await collect(**kwargs)  # Every group uses policy step zero.

        async def train(groups):
            training.set()
            await generated.wait()  # Generation must progress during training.
            return {"loss": 0.5}

        result = await self.drive(4, generate, train, groups_per_batch=1)
        self.assertEqual(result["learner_step"], 3)
        self.assertEqual(result["buffer_stats"]["rejected_stale"], 1)

    async def test_limit_errors_and_cancellation_clean_up(self):
        for stop in ("limit", "generate_error", "train_error", "cancel"):
            with self.subTest(stop=stop):
                generating, training = asyncio.Event(), asyncio.Event()
                generation_closed, training_closed = asyncio.Event(), asyncio.Event()
                error = RuntimeError(stop)

                async def generate(**kwargs):
                    if kwargs["group_id"] == 0:
                        return await collect(**kwargs)
                    generating.set()
                    try:
                        if stop == "generate_error":
                            await training.wait()
                            raise error
                        await asyncio.Event().wait()
                    finally:
                        generation_closed.set()

                async def train(groups):
                    await generating.wait()
                    training.set()
                    try:
                        if stop == "train_error":
                            raise error
                        if stop in ("cancel", "generate_error"):
                            await asyncio.Event().wait()
                        return {"loss": 1.0}
                    finally:
                        training_closed.set()

                runner = asyncio.create_task(self.drive(
                    2, generate, train, num_train_steps=1, groups_per_batch=1,
                ))
                try:
                    if stop == "cancel":
                        await asyncio.wait_for(training.wait(), 2)
                        runner.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await runner
                    elif stop.endswith("error"):
                        with self.assertRaises(ExceptionGroup) as raised:
                            await runner
                        self.assertIn(error, raised.exception.exceptions)
                    else:
                        result = await runner
                        self.assertEqual(result["learner_step"], 1)
                    self.assertTrue(generation_closed.is_set())
                    self.assertTrue(training_closed.is_set())
                finally:
                    runner.cancel()
                    await asyncio.gather(runner, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
