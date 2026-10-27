# connect model forward to the token scorer
import torch
from losses import token_logprobs, group_advantages, clipped_policy_loss


def train_step(model, optimizer, batch, proximal_logprobs, micro_batch_size=1,
               ppo_epochs=1, num_minibatches=1):
    """
        current logprobs - rollout tokens, logprobs from the current trainer policy
        behavior logprobs - logprobs recorded by the generator when sampling those tokens
        proximal logprobs - rollout tokens scored by the frozen proximal learner policy
        advantage - advantage from the rollouts
        loss_mask - mask of the rollouts
        ppo_epochs x num_minibatches optimizer steps; micro-batches only bound memory
    """
    if ppo_epochs < 1 or num_minibatches < 1:
        raise ValueError("ppo_epochs and num_minibatches must be positive")
    optimizer.zero_grad()
    batch_size = batch["input_ids"].shape[0]
    # compute the advantages for the batch to have them for the entire group
    advantages = group_advantages(
        batch["rewards"],
        batch["group_index"],
    )
    # all-equal groups carry no signal; skipping keeps Adam momentum from moving weights
    if not advantages.any():
        return {"loss": 0.0, "grad_norm": 0.0}
    minibatch_size = -(-batch_size // num_minibatches)
    losses, grad_norms = [], []

    for epoch in range(ppo_epochs):
        order = torch.randperm(batch_size) if num_minibatches > 1 else torch.arange(batch_size)
        for start in range(0, batch_size, minibatch_size):
            rows = order[start:start + minibatch_size]
            total_tokens = batch["loss_mask"][rows, 1:].sum()
            total_loss = 0.0

            for i in range(0, len(rows), micro_batch_size):
                micro_rows = rows[i:i+micro_batch_size]
                micro_batch = {key: value[micro_rows] for key, value in batch.items()}
                # current policy logprobs on the trainer
                current_logprobs = score_batch(model, micro_batch)

                loss = clipped_policy_loss(
                    current_logprobs=current_logprobs,
                    behavior_logprobs=micro_batch["behavior_logprobs"][:, 1:],
                    proximal_logprobs=proximal_logprobs[micro_rows],
                    advantages=advantages[micro_rows],
                    loss_mask=micro_batch["loss_mask"][:, 1:],
                )
                micro_tokens = micro_batch["loss_mask"][:, 1:].sum()
                weighted_loss = loss * (micro_tokens / total_tokens)
                weighted_loss.backward()
                total_loss += weighted_loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True)
            optimizer.step()

            # free the gradient memory here
            optimizer.zero_grad()
            losses.append(total_loss)
            grad_norms.append(grad_norm.item())

    return {"loss": sum(losses) / len(losses), "grad_norm": sum(grad_norms) / len(grad_norms)}


def score_proximal(model, batch, micro_batch_size=1):
    # frozen learner scores, taken before any update on this batch
    with torch.no_grad():
        proximal_rows = []
        for i in range(0, batch["input_ids"].shape[0], micro_batch_size):
            micro_batch = {
                key: value[i:(i+micro_batch_size)]
                for key, value in batch.items()
            }
            proximal_rows.append(score_batch(model, micro_batch))
        return torch.cat(proximal_rows, dim=0)

def score_batch(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    return token_logprobs(outputs.logits, batch["input_ids"])
