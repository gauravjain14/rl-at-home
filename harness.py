import time
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from protocol import RolloutGroup, Trajectory
from batching import collate_groups
from learner import score_batch

torch.manual_seed(7)
torch.set_printoptions(precision=3, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Token IDs: PAD=0, A=1, B=2, C=3, D=4, E=5
names = ["PAD", "A", "B", "C", "D", "E"]

model = Qwen2ForCausalLM(Qwen2Config(
    vocab_size=len(names), hidden_size=16, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    max_position_embeddings=16, pad_token_id=0,
)).to(device).eval()

# source learner is step the step at which the trainer was when
# the rollout group was first launched.
rollout_group = RolloutGroup(
    group_id=0,
    attempt_id=0,
    run_epoch=0,
    source_learner_step=0,
    created_at=time.time(),
    trajectories=[
        Trajectory([1, 2], [3, 4], [-1.2, -0.7], [True, True], 1.0, "stop"),
        Trajectory([1, 2], [5], [-2.0], [True], 0.0, "stop"),
    ],
)
some_trajectories = [
    Trajectory([1, 2], [3, 4], [-1.2, -0.7], [True, True], 1.0, "stop"),
    Trajectory([1, 2], [5], [-2.0], [True], 0.0, "stop"),
]
batch = collate_groups([rollout_group], pad_token_id=0)

batch = {key: value.to(device) for key, value in batch.items()}

# score the batch (this is using the learner policy)
with torch.no_grad():
    current_logprobs = score_batch(model, batch)

# behavior logprobs are stored as token id and the probs of picking that
# unlike outputs from model that are token id and probs of picking the next
behavior_logprobs = batch["behavior_logprobs"][:, 1:]
print(behavior_logprobs.shape)
