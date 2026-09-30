import asyncio
from typing import List
from protocol import RolloutGroup


class BufferClosed(RuntimeError):
    pass

class RolloutBuffer:
    def __init__(self, capacity: int, max_lag: int):
        # capacity implies how many rollouts can be held
        # before the trainer picks it up. If full, old
        # entries are removed
        if capacity <= 0:
            raise ValueError("Capacity must be > 0")
        
        if max_lag < 0:
            raise ValueError("Max lag must be > 0")
        
        self.buffer: List[RolloutGroup] = []
        self.capacity = capacity
        self.max_lag = max_lag
        self.condition = asyncio.Condition()
        self.closed = False

        self.stats = {
            "accepted": 0,
            "dropped_capacity": 0,
            "rejected_run": 0,
            "rejected_future": 0,
            "rejected_stale": 0,
        }

    async def close(self) -> None:
        async with self.condition:
            self.closed = True
            self.condition.notify_all()

    async def put(self, group: RolloutGroup) -> None:
        async with self.condition:
            if self.closed:
                raise BufferClosed("Cannot put to a closed buffer")

            if (len(self.buffer) >= self.capacity):
                self.buffer.pop(0)
                self.stats["dropped_capacity"] += 1    
            self.buffer.append(group)
            self.condition.notify()

    async def take(self, learner_step: int, run_epoch: int) -> RolloutGroup:
        async with self.condition:
            # Still trying to understand why this behavior is needed by Astra says it is.
            while True:
                await self.condition.wait_for(
                    lambda: len(self.buffer) > 0 or self.closed,
                )
                if not self.buffer:
                    raise BufferClosed("Cannot take from a closed buffer")

                ret_entry = self.buffer.pop(0)
                lag = learner_step - ret_entry.source_learner_step
                if ret_entry.run_epoch != run_epoch:
                    self.stats["rejected_run"] += 1
                    continue
                if lag < 0:
                    self.stats["rejected_future"] += 1
                    continue
                if lag > self.max_lag:
                    self.stats["rejected_stale"] += 1
                    continue

                self.stats["accepted"] += 1
                return ret_entry