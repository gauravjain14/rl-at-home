import math
import time

import torch

from protocol import Trajectory, RolloutGroup

# keep full-vocab log-probs of every step on each trajectory (~0.6 MB per row per step)
DEBUG_LOGP = False

class HFGenerator:
    def __init__(self, model, eos_token_ids=(), pad_token_id=0):
        self.model = model
        self.eos_token_ids = eos_token_ids
        self.pad_token_id = pad_token_id
    
    def generate(self, input_prompts,
                    num_gens=1,
                    max_samples_per_gen=8192,
                    greedy=False):
        return generate_rollouts(
            input_prompts=input_prompts,
            behavior_policy=self.model,
            num_gens=num_gens,
            max_samples_per_gen=max_samples_per_gen,
            eos_token_ids=self.eos_token_ids,
            pad_token_id=self.pad_token_id,
            greedy=greedy
        )
    
    def sync(self, model):
        # not needed yet.
        pass


@torch.no_grad()
def generate_rollouts(input_prompts,
                    behavior_policy,
                    num_gens=1,
                    max_samples_per_gen=8192,
                    eos_token_ids=(),
                    pad_token_id=0,
                    greedy=False):

    behavior_policy.eval()
    device = input_prompts[0].device
    rows = len(input_prompts) * num_gens
    max_prompt_len = max(len(p) for p in input_prompts)
    input_ids = torch.full((rows, max_prompt_len), pad_token_id, device=device)
    print(input_ids.shape)
    attention_mask = torch.zeros((rows, max_prompt_len), dtype=torch.long, device=device)
    for k, p in enumerate(input_prompts):
        # attention mask for every prompt
        num_pad_tokens = max_prompt_len - len(p)
        input_ids[k * num_gens : (k+1) * num_gens, num_pad_tokens:] = p
        attention_mask[k * num_gens : (k+1) * num_gens,num_pad_tokens:] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    eos = torch.tensor(eos_token_ids, dtype=torch.long, device=device)
    done = torch.zeros(rows, dtype=torch.bool, device=device)
    lengths = torch.full((rows,), max_samples_per_gen, device=device)
    tokens, token_logprobs = [], []
    past_key_values = None
    debugging_logp = []

    for i in range(max_samples_per_gen):
        outputs = behavior_policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1, # prevent the mem blowup hello.
        )
        # assuming output [B, S, Vocab]
        if DEBUG_LOGP:
            debug_logprobs = outputs.logits.float().log_softmax(dim=-1)
            debugging_logp.append(debug_logprobs.cpu())
        logprobs = outputs.logits[:, -1, :].float().log_softmax(dim=-1)
        if greedy:
            next_token = logprobs.argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(logprobs.exp(), num_samples=1)
        tokens.append(next_token.squeeze(-1))
        token_logprobs.append(logprobs.gather(dim=-1, index=next_token).squeeze(-1))
        is_eos = ~done & torch.isin(next_token.squeeze(-1), eos)
        lengths = torch.where(is_eos, i + 1, lengths)
        done |= is_eos
        # this just checks for completion every 16 iterations. There should
        # be a better way tbh.
        if i % 16 == 15 and done.all():
            break
        past_key_values = outputs.past_key_values
        input_ids = next_token
        attention_mask = torch.cat([attention_mask,
                                attention_mask.new_ones(rows, 1)], dim=1)
        position_ids = position_ids[:, -1:] + 1

    tokens = torch.stack(tokens, dim=1).tolist()
    token_logprobs = torch.stack(token_logprobs, dim=1).tolist()
    finished = done.tolist()
    return [{
        "debug_logp": [step[j].clone() for step in debugging_logp],
        "response_ids": tokens[j][:n],
        "behavior_logprobs": token_logprobs[j][:n],
        "finish_reason": "stop" if finished[j] else "length",
    } for j, n in enumerate(lengths.tolist())]

def collect_rollout_groups(
        tasks, input_prompts, generator,
        tokenizer, score_fn,
        *, group_id, run_epoch, source_learner_step,
        attempt_id=0, num_gens=4, max_samples_per_gen=128
) -> RolloutGroup:
    """Generate and score responses to one unpadded prompt shaped [1, P].

    score_fn(task, response_text) returns a scalar reward.
    source_learner_step identifies the generator's loaded weights; the caller
    must keep those weights unchanged throughout this call.
    """
    if len(tasks) == 0 or len(tasks) != len(input_prompts):
        raise ValueError("Expected one prompt per task")
    if num_gens < 1 or max_samples_per_gen < 1:
        raise ValueError("Generation count and response token limit must be positive")

    created_at = time.time()
    # not padded here
    prompts = [inp_prompt[0] for inp_prompt in input_prompts]
    rollouts = generator.generate(
        prompts,
        num_gens=num_gens,
        max_samples_per_gen=max_samples_per_gen,
    )

    groups = []
    for k, (task, prompt) in enumerate(zip(tasks, prompts)):
        prompt_ids = prompt.tolist()
        trajectories = []
        for rollout in rollouts[k * num_gens:(k + 1) * num_gens]:
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
                logp=rollout.get("debug_logp"),
            ))

        groups.append(RolloutGroup(
            group_id=group_id + k,
            attempt_id=attempt_id,
            run_epoch=run_epoch,
            source_learner_step=source_learner_step,
            created_at=created_at,
            trajectories=trajectories,
        ))
    return groups


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