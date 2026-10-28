"""Python 3.11+. From rl-at-home/: python -B -m unittest -v test_buffer"""

import asyncio
import unittest

from buffer import BufferClosed, RolloutBuffer
from protocol import RolloutGroup


def group(name="A", step=10, epoch=2):
    return RolloutGroup(name, 0, epoch, step, 0, [])


class BufferTests(unittest.IsolatedAsyncioTestCase):
    async def pending(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.addAsyncCleanup(self.cancel, task)
        await asyncio.sleep(0)  # Let the task reach its wait; no timed sleep.
        self.assertFalse(task.done())
        return task

    async def cancel(self, task):
        task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)

    def test_configuration(self):
        for capacity, lag in ((0, 2), (-1, 2), (1, -1)):
            with self.subTest(capacity=capacity, lag=lag), self.assertRaises(ValueError):
                RolloutBuffer(capacity, lag)
        RolloutBuffer(1, 0)

    async def test_admission(self):
        # learner step, lag budget, source step, execution, rejection reason
        cases = [
            (10, 2, 10, 2, None), (10, 2, 8, 2, None),
            (10, 2, 7, 2, "rejected_stale"),
            (10, 2, 11, 2, "rejected_future"),
            (10, 2, 10, 1, "rejected_run"), (10, 2, 10, 3, "rejected_run"),
            (10, 2, 7, 1, "rejected_run"),  # Count one reason, not two.
            (10, 0, 10, 2, None), (10, 0, 9, 2, "rejected_stale"),
            (11, 2, 8, 2, "rejected_stale"),  # Lag at consumption.
        ]
        for learner, limit, source, epoch, reason in cases:
            with self.subTest(case=(learner, limit, source, epoch)):
                async with asyncio.timeout(2):
                    buffer = RolloutBuffer(2, limit)
                    candidate, valid = group(step=source, epoch=epoch), group("B", learner)
                    await buffer.put(candidate)
                    await buffer.put(valid)
                    self.assertIs(await buffer.take(learner, 2), valid if reason else candidate)
                    self.assertEqual(buffer.buffer, [] if reason else [valid])
                    expected = dict.fromkeys(buffer.stats, 0)
                    expected["accepted"] = 1
                    if reason:
                        expected[reason] = 1
                    self.assertEqual(buffer.stats, expected)

    async def test_fifo_and_overflow(self):
        async with asyncio.timeout(2):
            buffer = RolloutBuffer(2, 2)
            entries = [group(name) for name in ("A", "B", "C")]
            for entry in entries:
                await buffer.put(entry)
            self.assertEqual(buffer.buffer, entries[1:])
            for entry in entries[1:]:
                self.assertIs(await buffer.take(10, 2), entry)
            self.assertEqual(buffer.buffer, [])
            self.assertEqual(buffer.stats["dropped_capacity"], 1)
            self.assertEqual(buffer.stats["accepted"], 2)

    async def test_waiting_rechecks_after_empty_or_invalid_notifications(self):
        async with asyncio.timeout(2):
            buffer = RolloutBuffer(3, 2)
            consumer = await self.pending(buffer.take(10, 2))
            async with buffer.condition:
                buffer.condition.notify_all()
            await asyncio.sleep(0)
            self.assertFalse(consumer.done())
            for invalid in (group(epoch=1), group(step=11), group(step=7)):
                await buffer.put(invalid)
            await asyncio.sleep(0)
            self.assertFalse(consumer.done())
            self.assertEqual(buffer.buffer, [])
            valid = group()
            await buffer.put(valid)
            self.assertIs(await consumer, valid)
            self.assertEqual(buffer.stats, dict(accepted=1, dropped_capacity=0,
                             rejected_run=1, rejected_future=1, rejected_stale=1))

    async def test_cancellation_leaves_buffer_usable(self):
        async with asyncio.timeout(2):
            buffer = RolloutBuffer(1, 2)
            consumer = await self.pending(buffer.take(10, 2))
            consumer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await consumer
            self.assertFalse(buffer.condition.locked())
            valid = group()
            await buffer.put(valid)
            self.assertIs(await buffer.take(10, 2), valid)

    async def test_close_preserves_queue_rejects_puts_and_drains(self):
        async with asyncio.timeout(2):
            buffer = RolloutBuffer(3, 2)
            entries = [group("A"), group("B"), group("stale", step=7)]
            for entry in entries:
                await buffer.put(entry)
            await buffer.close()
            await buffer.close()  # Repeated close must be harmless.
            before = buffer.stats.copy()
            with self.assertRaises(BufferClosed):
                await buffer.put(group("late"))
            self.assertEqual(buffer.buffer, entries)
            self.assertEqual(buffer.stats, before)
            for entry in entries[:2]:
                self.assertIs(await buffer.take(10, 2), entry)
            with self.assertRaises(BufferClosed):
                await buffer.take(10, 2)  # Reject the stale tail, then stop.
            self.assertEqual(buffer.buffer, [])
            self.assertEqual(buffer.stats["rejected_stale"], 1)

    async def test_single_consumption_and_close_wakes_remaining_consumers(self):
        async with asyncio.timeout(2):
            buffer = RolloutBuffer(1, 2)
            consumers = [await self.pending(buffer.take(10, 2)) for _ in range(3)]
            valid = group()
            await buffer.put(valid)
            await asyncio.sleep(0)
            done = [task for task in consumers if task.done()]
            self.assertEqual(len(done), 1)
            self.assertIs(await done[0], valid)
            await buffer.close()
            for task in consumers:
                if task is not done[0]:
                    with self.assertRaises(BufferClosed):
                        await task
            self.assertEqual(buffer.stats["accepted"], 1)

    async def test_close_before_queued_put(self):
        async with asyncio.timeout(2):
            buffer = RolloutBuffer(1, 2)
            async with buffer.condition:  # Force close to queue before put.
                closing = await self.pending(buffer.close())
                putting = await self.pending(buffer.put(group()))
            await closing
            with self.assertRaises(BufferClosed):
                await putting
            self.assertEqual(buffer.buffer, [])


if __name__ == "__main__":
    unittest.main()
