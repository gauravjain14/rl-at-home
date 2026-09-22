import math
import torch
import torch.nn.functional as F


def token_logprobs(logits, input_ids):
    # find the logprobs for the tokens in the input_ids
    log_probs = F.log_softmax(
        logits[:, :-1], dim=-1, dtype=torch.float32)
    target_ids = input_ids[:, 1:]
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def group_advantages(rewards, group_index, normalize_std=True, eps=1e-6):
    rewards = rewards.float()
    advantages = torch.zeros_like(rewards)

    # assumes group_index: [B] one group id per response in the batch
    for group_id in group_index.unique():
        selected = group_index == group_id
        values = rewards[selected]
        centered = values - values.mean()
        if normalize_std:
            # correction=0 means divide by N instead of N-1
            centered = centered / (centered.std(correction=0) + eps)
        advantages[selected] = centered

    return advantages


# what are the things required
# decoupled policy comes from the verl stuff. It is ppo but with an additional
# factor that separates out behavior policy from proximal policy
def clipped_policy_loss(
        current_logprobs,
        behavior_logprobs,
        proximal_logprobs,
        advantages,
        loss_mask,
        clip_eps=0.2,
        correction=2.0,
):
    """
    Current is the live trainer policy: it scores the recorded tokens with gradients
        enabled, and its weights change with optimizer steps.
    Behavior is generator policy and proximal policy is something that stays fixed
        during the chosen update window.
    """
    if not (
      0 <= clip_eps < 1
      and math.isfinite(correction)
      and correction > 0
    ):
      raise ValueError("Invalid clipping configuration")

    # all flattened arrays where loss_mask is True
    current_logprobs_masked = current_logprobs.float()[loss_mask]
    behavior_logprobs_masked = behavior_logprobs.detach().float()[loss_mask]
    proximal_logprobs_masked = proximal_logprobs.detach().float()[loss_mask]

    # advantage is of the shape B and needs to be applied to each token in the sequence
    adv = advantages.detach().float().unsqueeze(-1)
    adv_masked = adv.expand_as(current_logprobs)[loss_mask]

    for values in (
        current_logprobs_masked,
        behavior_logprobs_masked,
        proximal_logprobs_masked,
        adv_masked,
    ):
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite policy-loss inputs")

    ratio = (current_logprobs_masked - proximal_logprobs_masked).exp()
    correction = (proximal_logprobs_masked - behavior_logprobs_masked).clamp(
          max=math.log(correction)
      ).exp()

    unclipped = ratio * adv_masked
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_masked
    loss = -(correction * torch.minimum(unclipped, clipped)).mean()

    if not torch.isfinite(ratio).all() or not torch.isfinite(loss):
        raise FloatingPointError("Non-finite policy ratio or loss")

    return loss
