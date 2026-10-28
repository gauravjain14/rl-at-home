"""
orchestrator of the behavior, learner, environments and verifier
"""
import torch
import asyncio
import itertools
import json
import ray
import shutil
import sys
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from learner import score_batch
from batching import collate_groups
from rollout import collect_rollout_groups, HFGenerator
from buffer import RolloutBuffer, BufferClosed
from workers import RolloutWorker, LearnerWorker, SyncWorker
from tasks import get_task

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
                    tasks=[task],
                    input_prompts=[inp_prompt["input_ids"]],
                    group_id=group_id,
                    run_epoch=run_epoch,
                    max_samples_per_gen=max_samples_per_gen
                )

                await rollout_buffer.put(rollout_groups[0])
                group_id += 1
    finally:
        await rollout_buffer.close()


# sync workers
async def main(sync_worker,
               *,
               train_batches,
               num_train_steps=100,
               groups_per_batch=16,
               run_epoch=0,
               task_name="gsm8k",
               max_samples_per_gen=512,
               eval_inputs=None,
               checkpoint_dir=None,
               wandb_run=None,
               checkpoint_every=100,
               run_config=None,
               start_step=0):
    if groups_per_batch < 1 or checkpoint_every < 1 or (
        num_train_steps is not None and num_train_steps < 1
    ):
        raise ValueError("Step count, groups_per_batch and checkpoint_every must be positive")

    config = dict(run_config or {})
    config.update(task=task_name, groups_per_batch=groups_per_batch,
                  max_samples_per_gen=max_samples_per_gen, num_train_steps=num_train_steps,
                  checkpoint_every=checkpoint_every, step_unit="training_batches",
                  run_epoch=run_epoch)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        (checkpoint_dir / "run_config.json").write_text(json.dumps(config, indent=2))
        if eval_inputs is not None:
            (checkpoint_dir / "validation_ids.json").write_text(json.dumps(
                [task["question_id"] for task, _ in eval_inputs], indent=2,
            ))

    async def evaluate(step):
        if eval_inputs is None:
            return None
        output_path = (str(checkpoint_dir / f"eval-{step:06d}.json")
                       if checkpoint_dir is not None else None)
        scores = await sync_worker.evaluate.remote(
            eval_inputs, max_samples_per_gen, output_path=output_path,
        )
        print({"learner_step": step, "eval": scores})
        if wandb_run is not None:
            wandb_run.log({
                "learner_step": step,
                **{f"eval/{key}": value for key, value in scores.items()},
            })
        return scores

    saved_checkpoints = []
    best_checkpoint, latest_checkpoint = None, None
    best_accuracy = float("-inf")

    async def checkpoint_and_evaluate(step):
        nonlocal best_checkpoint, latest_checkpoint, best_accuracy
        if checkpoint_dir is not None:
            path = checkpoint_dir / f"step-{step:06d}"
            await sync_worker.save_checkpoint.remote(str(path))
            (path / "run_config.json").write_text(json.dumps(config, indent=2))
            saved_checkpoints.append(path)
            latest_checkpoint = path
            print({"learner_step": step, "checkpoint": str(path)})
        scores = await evaluate(step)
        if checkpoint_dir is not None:
            if scores is not None and scores["accuracy"] > best_accuracy:
                best_accuracy = scores["accuracy"]
                best_checkpoint = latest_checkpoint
            keep = set(saved_checkpoints[-2:]) | {best_checkpoint}
            for old_path in saved_checkpoints[:]:
                if old_path not in keep:
                    shutil.rmtree(old_path)
                    saved_checkpoints.remove(old_path)
            (checkpoint_dir / "checkpoints.json").write_text(json.dumps({
                "latest": latest_checkpoint.name,
                "best": best_checkpoint.name if best_checkpoint is not None else None,
                "best_accuracy": best_accuracy if best_checkpoint is not None else None,
            }, indent=2))
        return scores

    # a resumed run was already evaluated at start_step
    before = await evaluate(0) if start_step == 0 else None
    last_eval, last_checkpoint_step = before, start_step

    step = start_step
    group_id = 0
    metrics = {}
    for input_batch in train_batches:
        groups = await sync_worker.collect_group.remote(
            tasks=[task for task, _ in input_batch],
            input_prompts=[inp_prompt["input_ids"] for _, inp_prompt in input_batch],
            group_id=group_id,
            run_epoch=run_epoch,
            max_samples_per_gen=max_samples_per_gen,
        )
        group_id += len(groups)

        metrics = await sync_worker.train_groups.remote(groups)
        step += 1
        print({"learner_step": step, **metrics})
        if wandb_run is not None:
            wandb_run.log({
                "learner_step": step,
                **{f"train/{key}": value for key, value in metrics.items()},
            })
        if step % checkpoint_every == 0:
            last_eval = await checkpoint_and_evaluate(step)
            last_checkpoint_step = step
        if num_train_steps is not None and step >= num_train_steps:
            break

    if step > 0 and step != last_checkpoint_step:
        last_eval = await checkpoint_and_evaluate(step)
    result = {"learner_step": step, "metrics": metrics}
    if eval_inputs is not None:
        result["eval_before"] = before
        result["eval_after"] = last_eval
    if latest_checkpoint is not None:
        result["checkpoint_dir"] = str(latest_checkpoint)
    if best_checkpoint is not None:
        result["best_checkpoint_dir"] = str(best_checkpoint)
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
        # resume: python driver.py <task> <checkpoints/.../step-N>
        resume_from = str(Path(sys.argv[2]).resolve()) if len(sys.argv) > 2 else None
        model = AutoModelForCausalLM.from_pretrained(
            resume_from or model_name,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        # Single source for settings used by both the code and the logged config.
        num_gens, learning_rate, groups_per_batch, checkpoint_every = 8, 1e-6, 16, 25
        seed, max_samples_per_gen, num_train_steps = 42, 512, None
        micro_batch_size, update_epochs, num_minibatches = 1, 1, 1
        gradient_checkpointing = False
        # swap in another engine here: any class with generate() and sync()
        generator_cls = HFGenerator
        task_name = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
        if task_name == "music":
            max_samples_per_gen, gradient_checkpointing = 2048, True
        task = get_task(task_name)
        sync_worker = SyncWorker.remote(
            model=model, device="cuda:0", pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer, score_fn=task.score, num_gens=num_gens,
            answer_parser=task.parse, lr=learning_rate, seed=seed,
            micro_batch_size=micro_batch_size, update_epochs=update_epochs,
            num_minibatches=num_minibatches,
            gradient_checkpointing=gradient_checkpointing, generator_cls=generator_cls,
        )
        del model  # Release the driver's CPU copy after submitting it to Ray.
        start_step = ray.get(sync_worker.load_trainer_state.remote(resume_from)) if resume_from else 0
        # learner_worker = LearnerWorker.remote(
        #     model=model, device="cuda:0", pad_token_id=tokenizer.pad_token_id,
        # )
        # del model
        # rollout_worker = RolloutWorker.remote(
        #     model_name=model_name, score_fn=gsm8k_score_fn, num_gens=4,
        # )
        eval_inputs = [
            entry for batch in task.batches(tokenizer, split="validation")
            for entry in batch
        ]
        checkpoint_dir = Path("checkpoints") / f"{task_name}-{time.time_ns()}"
        run_config = {
            "model": model_name, "task": task_name, "num_gens": num_gens,
            "groups_per_batch": groups_per_batch, "micro_batch_size": micro_batch_size,
            "update_epochs": update_epochs, "num_minibatches": num_minibatches,
            "gradient_checkpointing": gradient_checkpointing,
            "generator": generator_cls.__name__,
            "learning_rate": learning_rate,
            "dtype": "float32", "temperature": 1.0, "seed": seed,
            "validation_size": getattr(task, "validation_size", None),
            "split_seed": getattr(task, "split_seed", None),
            "max_samples_per_gen": max_samples_per_gen, "checkpoint_every": checkpoint_every,
            "num_train_steps": num_train_steps, "step_unit": "training_batches",
            "resumed_from": resume_from, "start_step": start_step,
        }
        # a resumed run continues the wandb run its checkpoint was saved from
        wandb_run_id = (json.loads((Path(resume_from) / "run_config.json").read_text()).get("wandb_run_id")
                        if resume_from else None)
        with wandb.init(
            entity="scratchpad",
            project="nanoRL-release",
            name="gsm8k-qwen2.5-0.5b-full-ft",
            config=run_config,
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
            allow_val_change=True,
        ) as run:
            run_config["wandb_run_id"] = run.id
            run.define_metric("learner_step")
            run.define_metric("train/*", step_metric="learner_step")
            run.define_metric("eval/*", step_metric="learner_step")
            result = asyncio.run(main(
                sync_worker=sync_worker,
                # the task order is fixed by its seed, so skipping the trained batches resumes exactly
                train_batches=itertools.islice(task.batches(tokenizer, batch_size=groups_per_batch),
                                               start_step, None),
                num_train_steps=num_train_steps, groups_per_batch=groups_per_batch,
                max_samples_per_gen=max_samples_per_gen,
                task_name=task_name, eval_inputs=eval_inputs, checkpoint_dir=checkpoint_dir,
                wandb_run=run, checkpoint_every=checkpoint_every, run_config=run_config,
                start_step=start_step,
            ))
        with (checkpoint_dir / "run_metrics.json").open("w") as output:
            json.dump(result, output, indent=2)
        print(result)
    finally:
        ray.shutdown()
