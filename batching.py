import torch
from torch.nn.utils.rnn import pad_sequence
from protocol import RolloutGroup, Trajectory


def collate_groups(groups, pad_token_id):
    """ flattent the groups into one row per trajectory.
    input_ids = [B, T] (prompt + response and right padded)
    attention_mask = [B, T] (prompt + response and right padded)
    loss_mask = [B, T] (false for prompt and padding)
    behavior_logprobs = [B, T]
    rewards = [B]
    group_index = [B]
    """

    rewards = []
    group_indices = []
    id_rows = []
    attention_mask_rows = []
    loss_mask_rows = []
    logprobs_rows = []

    for group_idx, group in enumerate(groups):
        for trajectory in group.trajectories:
            if not trajectory.prompt_ids or not trajectory.response_ids:
                raise ValueError("Trajectory has no prompt or response")
            
            if not(
                len(trajectory.response_ids)
                == len(trajectory.behavior_logprobs)
                == len(trajectory.loss_mask)
            ):
                raise ValueError("response, logprobs, and loss should have equal lengths")
            
            prompt_length = len(trajectory.prompt_ids)
            ids = trajectory.prompt_ids + trajectory.response_ids
            attention = [True] * len(ids)
            loss_mask = [False] * prompt_length + trajectory.loss_mask
            logprobs = [0.0] * prompt_length + trajectory.behavior_logprobs

            rewards.append(trajectory.reward)
            id_rows.append(torch.tensor(ids, dtype=torch.long))
            attention_mask_rows.append(torch.tensor(attention, dtype=torch.bool))
            loss_mask_rows.append(torch.tensor(loss_mask, dtype=torch.bool))
            logprobs_rows.append(torch.tensor(logprobs, dtype=torch.float))
            group_indices.append(group_idx)

    input_ids = pad_sequence(id_rows, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask_rows, batch_first=True, padding_value=False)
    loss_mask = pad_sequence(loss_mask_rows, batch_first=True, padding_value=False)
    behavior_logprobs = pad_sequence(logprobs_rows, batch_first=True, padding_value=0.0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "behavior_logprobs": behavior_logprobs,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "group_index": torch.tensor(group_indices, dtype=torch.long),
    }
