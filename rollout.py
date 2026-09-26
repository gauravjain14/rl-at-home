import torch


@torch.no_grad()
def generate_rollouts(input_prompt,
                    behavior_policy,
                    num_gens=1,
                    max_samples_per_gen=128,
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