"""
orchestrator of the behavior, learner, environments and verifier
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from batching import collate_groups
from rollout import collect_rollout_groups

model_name = "Qwen/Qwen3.5-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# the role of the driver is to prepare batches for generation/rollouts.
# get rewards on these rollouts - this includes running the rollouts in
# environments if any, get rewards and compute the loss.
# take the outputs from the rollout and put it in a buffer for the learner
# to do the backward pass (or compute the loss if more information was needed)

def learner_step(inp_buffer, num_train_steps=10):
    # learner runs. Start with 0th step
    source_learner_step, keep_going = 0, True
    for i in range(num_train_steps):
        # get the forward pass on the logits
        
        pass


music_tasks = load_dataset(
    "XiaomiMiMo/MiMo-V2.6-RL-oss",
    name="music",
    split="train",
)

def prepare_tasks(batch_size=16):
    inputs = []
    for task in music_tasks:
        inp_prompt = tokenizer.apply_chat_template(
            task["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        
        inputs.append((task, inp_prompt))

        if len(inputs) == batch_size:
            yield inputs
            inputs = []

    if inputs:
        yield inputs

# generation pipeline that keep running separate from the trainer
def generate_rollouts(inputs, behavior_policy, score_fn,
                    group_id, run_epoch, source_learner_step,
                    max_samples_per_gen=1024):
    rollout_buffer = []

    # the music task from MiMo doesn't need an environment
    for task,inp_prompt in inputs:
        # get the behavior_policy - which is at the time we
        # launched these rollouts.
        rollout_groups = collect_rollout_groups(
            task=task,
            input_prompt=inp_prompt,
            behavior_policy=behavior_policy,
            score_fn=score_fn,
            group_id=group_id,
            run_epoch=run_epoch,
            source_learner_step=source_learner_step,
            max_samples_per_gen=max_samples_per_gen
        )

        rollout_trajectories = rollout_groups.trajectories
        batch = collate_groups(
            rollout_groups,
            pad_token_id=0
        )

        rollout_buffer.append(batch)

    return rollout_buffer


# orchestrator
current_policy = None  # get the policy from the learner
group_id = 0

for i in range(100):
    group_id += 1

    for inp in prepare_tasks(batch_size=16):
        print(len(inp))