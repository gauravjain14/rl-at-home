# connect model forward to the token scorer
import torch
from losses import token_logprobs, group_advantages, clipped_policy_loss


def train_step(model, optimizer, batch, proximal_logprobs, micro_batch_size=1):
    """
        current logprobs - rollout tokens, logprobs from the current trainer policy
        behavior logprobs - logprobs recorded by the generator when sampling those tokens
        proximal logprobs - rollout tokens scored by the frozen proximal learner policy
        advantage - advantage from the rollouts
        loss_mask - mask of the rollouts
    """
    optimizer.zero_grad()
    
    for i in range(0, len(batch), micro_batch_size):
        micro_batch = {key: value[i:i+micro_batch_size] for key, value in batch.items()}
        # current policy logprobs on the trainer
        current_logprobs = score_batch(model, micro_batch)
        advantages = group_advantages(
            micro_batch["rewards"],
            micro_batch["group_index"],
        )
        loss = clipped_policy_loss(
            current_logprobs=current_logprobs,
            behavior_logprobs=micro_batch["behavior_logprobs"][:, 1:],
            proximal_logprobs=proximal_logprobs[i:i+micro_batch_size],
            advantages=advantages,
            loss_mask=micro_batch["loss_mask"][:, 1:],
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True)
    optimizer.step()

    return {"loss": loss.item(), "grad_norm": grad_norm.item()}

def score_batch(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    return token_logprobs(outputs.logits, batch["input_ids"])
