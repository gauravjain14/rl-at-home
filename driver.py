"""
orchestrator of the behavior, learner, environments and verifier
"""

import asyncio
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from learner import score_batch
from batching import collate_groups
from rollout import collect_rollout_groups
from buffer import RolloutBuffer, BufferClosed

model_name = "Qwen/Qwen3.5-4B"

# Each worker can load its own model; importing the driver does not load one.
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model

# the role of the driver is to prepare batches for generation/rollouts.
# get rewards on these rollouts - this includes running the rollouts in
# environments if any, get rewards and compute the loss.
# take the outputs from the rollout and put it in a buffer for the learner
# to do the backward pass (or compute the loss if more information was needed)

async def learner_step(inp_buffer, learner_worker, num_train_steps=10,
                       *, run_epoch=0, groups_per_batch=1):
    # learner runs. Start with 0th step
    step, metrics = 0, {}
    while step < num_train_steps:
        groups = []
        try:
            for _ in range(groups_per_batch):
                group = await inp_buffer.take(learner_step=step, run_epoch=run_epoch)
                groups.append(group)
        except BufferClosed:
            pass  # Train the final partial batch, if any.

        if not groups:
            break

        # The worker collates, scores the proximal policy, and updates once.
        metrics = await learner_worker.train_groups.remote(groups)
        step += 1
        print({"learner_step": step, **metrics})

    return {"learner_step": step, "metrics": metrics}


mimo_score_fn = None

# hardcoded for mimo task
def prepare_tasks(batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    music_tasks = load_dataset(
        "XiaomiMiMo/MiMo-V2.6-RL-oss",
        name="music",
        split="train",
    )
    inputs = []
    for task in music_tasks:
        inp_prompt = tokenizer.apply_chat_template(
            task["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )  # The rollout worker moves input_ids onto its own GPU.
        
        inputs.append((task, inp_prompt))

        if len(inputs) == batch_size:
            yield inputs
            inputs = []

    if inputs:
        yield inputs

# generation pipeline that keeps running separate from the trainer
async def generate_rollouts(inputs,
                            rollout_worker,
                            rollout_buffer,
                            run_epoch=0,
                            max_samples_per_gen=1024):
    group_id = 0
    try:
        for input_batch in inputs:
            # the music task from MiMo doesn't need an environment
            for task,inp_prompt in input_batch:
                # collect_group wraps collect_rollout_groups inside the worker.
                # That worker supplies its actual loaded source_learner_step.
                # probably this might be a common entry point for other rollouts?
                rollout_groups = await rollout_worker.collect_group.remote(
                    task=task,
                    input_prompt=inp_prompt["input_ids"],
                    group_id=group_id,
                    run_epoch=run_epoch,
                    max_samples_per_gen=max_samples_per_gen
                )

                await rollout_buffer.put(rollout_groups)
                group_id += 1
    finally:
        await rollout_buffer.close()


# orchestrator
async def main(rollout_worker,
               learner_worker,
               *,
               num_train_steps=100,
               groups_per_batch=16,
               run_epoch=0):
    if num_train_steps < 1 or groups_per_batch < 1:
        raise ValueError("Step count and groups_per_batch must be positive")

    rollout_buffer = RolloutBuffer(capacity=16, max_lag=2)

    async with asyncio.TaskGroup() as tasks:
        rollout_producer = tasks.create_task(
            generate_rollouts(
                prepare_tasks(batch_size=16),
                rollout_worker,
                rollout_buffer,
                run_epoch=run_epoch
            )
        )
        learner_consumer = tasks.create_task(
            learner_step(
                rollout_buffer,
                learner_worker,
                num_train_steps=num_train_steps,
                run_epoch=run_epoch,
                groups_per_batch=groups_per_batch,
            )
        )

        result = await learner_consumer
        # Stop this local request loop; remote-job shutdown belongs to workers.
        rollout_producer.cancel()

    result["buffer_stats"] = rollout_buffer.stats.copy()
    return result
