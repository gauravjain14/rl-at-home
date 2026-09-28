"""
orchestrator of the behavior, learner, environments and verifier
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from rollout import collect_rollout_groups

model_name = "Qwen/Qwen3.5-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

music_tasks = load_dataset(
    "XiaomiMiMo/MiMo-V2.6-RL-oss",
    name="music",
    split="train",
)


# the role of the driver is to prepare batches for generation/rollouts.
# get rewards on these rollouts - this includes running the rollouts in
# environments if any, get rewards and compute the loss.
# take the outputs from the rollout and put it in a buffer for the learner
# to do the backward pass (or compute the loss if more information was needed)