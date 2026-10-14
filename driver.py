"""
orchestrator of the behavior, learner, environments and verifier
"""
import torch
import asyncio
import json
import ray
import time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from learner import score_batch
from batching import collate_groups
from rollout import collect_rollout_groups
from buffer import RolloutBuffer, BufferClosed
from workers import RolloutWorker, LearnerWorker, SyncWorker
from mimo_music.pipeline import compute_score
from gsm8k import gsm8k_score_fn

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Each worker can load its own model; importing the driver does not load one.
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="auto"
    )
    return model

# the role of the driver is to prepare batches for generation/rollouts.
# get rewards on these rollouts - this includes running the rollouts in
# environments if any, get rewards and compute the loss.
# take the outputs from the rollout and put it in a buffer for the learner
# to do the backward pass (or compute the loss if more information was needed)

async def learner_step(inp_buffer, learner_worker, rollout_worker,
                       num_train_steps=10,
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
        # step finished, send weight updates.
        ckpt_ref = learner_worker.export_weights.remote()
        step = await rollout_worker.load_weights.remote(ckpt_ref)
        print({"learner_step": step, **metrics})

    return {"learner_step": step, "metrics": metrics}


mimo_score_fn = None

# Task data stays on CPU until the rollout worker consumes it.
def prepare_tasks(batch_size=16, task_name="gsm8k", *, split="train", limit=None):
    if batch_size < 1 or (limit is not None and limit < 0):
        raise ValueError("Invalid task batch size or limit")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = []

    if task_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        if split == "train":
            dataset = dataset.shuffle(seed=42)
        # apply expected chat format on the dataset
        print(dataset[0])
        for index, row in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            # just maintains music-style task formating
            # I don't think this approach is scalable though
            task = {
                "prompt": [{
                    "role": "user",
                    "content": (
                        row["question"] 
                        + "\nSolve step by step. On the last line, write #### followed by your numerical answer."
                    )
                }],
                "answer": row["answer"].rsplit("####", 1)[-1].strip(),
            }
            inp_prompt = tokenizer.apply_chat_template(
                task["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs.append((task, inp_prompt))
            if len(inputs) == batch_size:
                yield inputs
                inputs = []
    elif task_name == "music":
        music_tasks = load_dataset(
            "XiaomiMiMo/MiMo-V2.6-RL-oss",
            name="music",
            split=split,
        )

        for index, task in enumerate(music_tasks):
            if limit is not None and index >= limit:
                break
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
    else:
        raise ValueError(f"Unknown task: {task_name}")

    if inputs:
        yield inputs

# generation pipeline that keeps running separate from the trainer
async def generate_rollouts(inputs,
                            rollout_worker,
                            rollout_buffer,
                            run_epoch=0,
                            max_samples_per_gen=512):
    group_id = 0
    try:
        for input_batch in inputs:
            # Single-turn tasks are scored after generation.
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


# sync workers
async def main(sync_worker,
               *,
               num_train_steps=100,
               groups_per_batch=1,
               run_epoch=0,
               task_name="gsm8k",
               max_samples_per_gen=512,
               eval_inputs=None,
               checkpoint_dir=None,
               wandb_run=None):
    if groups_per_batch < 1 or (
        num_train_steps is not None and num_train_steps < 1
    ):
        raise ValueError("Step count and groups_per_batch must be positive")

    before = None
    if eval_inputs is not None:
        before = await sync_worker.evaluate.remote(eval_inputs, max_samples_per_gen)
        print({"eval_before": before})
        if wandb_run is not None:
            wandb_run.log({
                "learner_step": 0,
                **{f"eval/{key}": value for key, value in before.items()},
            })

    step = 0
    group_id = 0
    metrics = {}
    for input_batch in prepare_tasks(batch_size=groups_per_batch, task_name=task_name):
        groups = []
        for task, inp_prompt in input_batch:
            group = await sync_worker.collect_group.remote(
                task=task,
                input_prompt=inp_prompt["input_ids"],
                group_id=group_id,
                run_epoch=run_epoch,
                max_samples_per_gen=max_samples_per_gen,
            )
            groups.append(group)
            group_id += 1

        metrics = await sync_worker.train_groups.remote(groups)
        step += 1
        print({"learner_step": step, **metrics})
        if wandb_run is not None:
            wandb_run.log({
                "learner_step": step,
                **{f"train/{key}": value for key, value in metrics.items()},
            })
        if num_train_steps is not None and step >= num_train_steps:
            break

    result = {"learner_step": step, "metrics": metrics}
    if eval_inputs is not None:
        result["eval_before"] = before
        result["eval_after"] = await sync_worker.evaluate.remote(eval_inputs, max_samples_per_gen)
        print({"eval_after": result["eval_after"]})
        if wandb_run is not None:
            wandb_run.log({
                "learner_step": step,
                **{f"eval/{key}": value for key, value in result["eval_after"].items()},
            })
    if checkpoint_dir is not None:
        result["checkpoint_dir"] = await sync_worker.save_checkpoint.remote(str(checkpoint_dir))
    return result


# orchestrator
# async def main(rollout_worker,
#                learner_worker,
#                *,
#                num_train_steps=100,
#                groups_per_batch=1,
#                run_epoch=0,
#                task_name="gsm8k",
#                max_samples_per_gen=512,
#                eval_inputs=None,
#                checkpoint_dir=None):
#     if num_train_steps < 1 or groups_per_batch < 1:
#         raise ValueError("Step count and groups_per_batch must be positive")

#     rollout_buffer = RolloutBuffer(capacity=16, max_lag=2)

#     # Start both workers from the exact same learner snapshot.
#     await rollout_worker.load_weights.remote(learner_worker.export_weights.remote())
#     before = None
#     if eval_inputs is not None:
#         before = await rollout_worker.evaluate.remote(eval_inputs, max_samples_per_gen)
#         print({"eval_before": before})

#     async with asyncio.TaskGroup() as tasks:
#         rollout_producer = tasks.create_task(
#             generate_rollouts(
#                 prepare_tasks(batch_size=16, task_name=task_name),
#                 rollout_worker,
#                 rollout_buffer,
#                 run_epoch=run_epoch,
#                 max_samples_per_gen=max_samples_per_gen,
#             )
#         )
#         learner_consumer = tasks.create_task(
#             learner_step(
#                 rollout_buffer,
#                 learner_worker,
#                 rollout_worker=rollout_worker,
#                 num_train_steps=num_train_steps,
#                 run_epoch=run_epoch,
#                 groups_per_batch=groups_per_batch,
#             )
#         )

#         result = await learner_consumer
#         # Stop this local request loop; remote-job shutdown belongs to workers.
#         rollout_producer.cancel()

#     result["buffer_stats"] = rollout_buffer.stats.copy()
#     if eval_inputs is not None:
#         result["eval_before"] = before
#         result["eval_after"] = await rollout_worker.evaluate.remote(eval_inputs, max_samples_per_gen)
#         print({"eval_after": result["eval_after"]})
#     if checkpoint_dir is not None:
#         result["checkpoint_dir"] = await learner_worker.save_checkpoint.remote(str(checkpoint_dir))
#     return result


if __name__ == "__main__":
    import wandb

    ray.init()
    try:
        if ray.cluster_resources().get("GPU", 0) < 1:
            raise RuntimeError("SyncWorker requires one available GPU.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, attn_implementation="sdpa",
        )
        sync_worker = SyncWorker.remote(
            model=model, device="cuda:0", pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer, score_fn=gsm8k_score_fn
        )
        del model  # Release the driver's CPU copy after submitting it to Ray.
        # learner_worker = LearnerWorker.remote(
        #     model=model, device="cuda:0", pad_token_id=tokenizer.pad_token_id,
        # )
        # del model
        # rollout_worker = RolloutWorker.remote(
        #     model_name=model_name, score_fn=gsm8k_score_fn, num_gens=4,
        # )
        eval_inputs = [
            entry for batch in prepare_tasks(task_name="gsm8k", split="test", limit=8)
            for entry in batch
        ]
        checkpoint_dir = Path("checkpoints") / f"gsm8k-{time.time_ns()}"
        with wandb.init(
            entity="scratchpad",
            project="nanoRL",
            config={"model": model_name, "task": "gsm8k"},
        ) as run:
            run.define_metric("learner_step")
            run.define_metric("train/*", step_metric="learner_step")
            run.define_metric("eval/*", step_metric="learner_step")
            result = asyncio.run(main(
                sync_worker=sync_worker,
                num_train_steps=None,
                task_name="gsm8k", eval_inputs=eval_inputs, checkpoint_dir=checkpoint_dir,
                wandb_run=run,
            ))
        tokenizer.save_pretrained(checkpoint_dir)
        with (checkpoint_dir / "run_metrics.json").open("w") as output:
            json.dump(result, output, indent=2)
        print(result)
    finally:
        ray.shutdown()
