# connect model forward to the token scorer
import torch
from losses import token_logprobs, group_advantages, clipped_policy_loss


def train_step(model, optimizer, batch, proximal_logprobs):
    """
        current logprobs - rollout tokens, logprobs from the current trainer policy
        behavior logprobs - logprobs recorded by the generator when sampling those tokens
        proximal logprobs - logits from the rollouts, logprobs from the proximal policy
        advantage - advantage from the rollouts
        loss_mask - mask of the rollouts
    """
    optimizer.zero_grad()

    current_logprobs = score_batch(model, batch)
    advantages = group_advantages(
        batch["rewards"],
        batch["group_index"],
    )
    loss = clipped_policy_loss(
        current_logprobs=current_logprobs,
        behavior_logprobs=batch["behavior_logprobs"][:, 1:],
        proximal_logprobs=proximal_logprobs,
        advantages=advantages,
        loss_mask=batch["loss_mask"][:, 1:],
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