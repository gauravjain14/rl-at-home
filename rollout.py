import math
import time

import torch

from protocol import Trajectory, RolloutGroup


@torch.no_grad()
def generate_rollouts(input_prompt,
                    behavior_policy,
                    num_gens=1,
                    max_samples_per_gen=8192,
                    eos_token_ids=()):

    behavior_policy.eval()
    rollouts = []
    for j in range(num_gens):
        input_ids = input_prompt
        past_key_values = None
        response_ids, behavior_logprobs = [], []
        finish_reason = "length"
        for i in range(max_samples_per_gen):
            output = behavior_policy(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            # assuming output [B, S, Vocab]
            logprobs = output.logits[:, -1, :].float().log_softmax(dim=-1)
            next_token = torch.multinomial(logprobs.exp(), num_samples=1)
            response_ids.append(next_token.item())
            sampled_logprobs = logprobs.gather(dim=-1, index=next_token)
            behavior_logprobs.append(sampled_logprobs.item())
            if next_token.item() in eos_token_ids:
                finish_reason = "stop"
                break

            past_key_values = output.past_key_values
            input_ids = next_token
        
        rollouts.append({
            "response_ids": response_ids,
            "behavior_logprobs": behavior_logprobs,
            "finish_reason": finish_reason
        })

    return rollouts


def collect_rollout_groups(
        task, input_prompt, behavior_policy,
        tokenizer, score_fn,
        *, group_id, run_epoch, source_learner_step,
        attempt_id=0, num_gens=4, max_samples_per_gen=128,
        eos_token_ids=(),
) -> RolloutGroup:
    """Generate and score responses to one unpadded prompt shaped [1, P].

    score_fn(task, response_text) returns a scalar reward.
    source_learner_step identifies the generator's loaded weights; the caller
    must keep those weights unchanged throughout this call.
    """
    if input_prompt.ndim != 2 or input_prompt.shape[0] != 1 or input_prompt.shape[1] == 0:
        raise ValueError("Expected one nonempty prompt with shape [1, P]")
    if num_gens < 1 or max_samples_per_gen < 1:
        raise ValueError("Generation count and response token limit must be positive")

    created_at = time.time()
    prompt_ids = input_prompt[0].tolist()
    rollouts = generate_rollouts(
        input_prompt,
        behavior_policy,
        num_gens=num_gens,
        max_samples_per_gen=max_samples_per_gen,
        eos_token_ids=eos_token_ids,
    )

    trajectories = []
    for rollout in rollouts:
        response_ids = rollout["response_ids"]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        reward = float(score_fn(task, response_text))
        if not math.isfinite(reward):
            raise ValueError("Scorer returned a non-finite reward")
        trajectories.append(Trajectory(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            behavior_logprobs=rollout["behavior_logprobs"],
            loss_mask=[True] * len(response_ids),
            reward=reward,
            finish_reason=rollout["finish_reason"],
        ))

    return RolloutGroup(
        group_id=group_id,
        attempt_id=attempt_id,
        run_epoch=run_epoch,
        source_learner_step=source_learner_step,
        created_at=created_at,
        trajectories=trajectories,
    )


# test generate_rollouts
# check the output, thinking and rollout.
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "Qwen/Qwen3.5-4B"

# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

# music_tasks = load_dataset(
#     "XiaomiMiMo/MiMo-V2.6-RL-oss",
#     name="music",
#     split="train",
# )

# for i in range(10):
#     task = music_tasks[i]

#     inp_prompt = tokenizer.apply_chat_template(
#         task["prompt"],
#         tokenize=True,
#         add_generation_prompt=True,
#         enable_thinking=True,
#         return_tensors="pt",
#         return_dict=True,
#     ).to("cuda")

#     print(f"Input ids ", inp_prompt["input_ids"])
#     inp_prompt_detok = tokenizer.decode(inp_prompt["input_ids"])
#     print(f"inp prompt {inp_prompt_detok}")

#     rollout = generate_rollouts(
#         inp_prompt["input_ids"],
#         model,
#         eos_token_ids=(model.generation_config.eos_token_id,)
#     )

#     print(rollout)
#     print(f"")
#     detokenize = tokenizer.decode(rollout[0]['response_ids'])
#     print(detokenize)