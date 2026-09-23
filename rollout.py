import torch


def generate_rollouts(input_prompt,
                    behavior_policy,
                    num_gens=1,
                    max_samples_per_gen=128):

    rollouts = []
    for j in range(num_gens):
        input_ids = input_prompt
        for i in range(max_samples_per_gen):
            output = behavior_policy(
                input_ids=input_ids,
                past_key_values=otuput_past_key_values
                use_cache=True,
                return_dict=True
            )
            # assuming output [B, S, Vocab]
            logprobs = output[:, -1, :].log_softmax(dim=-1)
            next_token = torch.multinomial(logprobs.exp(), num_samples=1)
            sampled_logprobs = logprobs.gather(dim=-1, index=next_token)

            past_key_values = output.past_key_values
            input_ids = next_token