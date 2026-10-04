import torch
import ray
from transformers import AutoModelForCausalLM, AutoTokenizer

from rollout import collect_rollout_groups
from batching import collate_groups
from learner import train_step, score_batch


@ray.remote(num_cpus=1, num_gpus=1)
class RolloutWorker:
    def __init__(self, model_name, score_fn, num_gens=4):
        if not callable(score_fn):
            raise ValueError("score_fn must be callable")

        self.device = "cuda:0"
        # initialize a copy of the model here which is then weight sync'd
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map={"": self.device},
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.score_fn = score_fn
        self.policy_version = 0
        self.num_gens = num_gens

        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_token_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())


    def collect_group(self, task, input_prompt, *, group_id, run_epoch,
                      max_samples_per_gen=1024):
        rollout_groups = collect_rollout_groups(
            task=task,
            input_prompt=input_prompt.to(self.device),
            behavior_policy=self.model,
            tokenizer=self.tokenizer,
            score_fn=self.score_fn,
            group_id=group_id,
            run_epoch=run_epoch,
            source_learner_step=self.policy_version,
            num_gens=self.num_gens,
            max_samples_per_gen=max_samples_per_gen,
            eos_token_ids=self.eos_token_ids,
        )
        return rollout_groups


@ray.remote(num_cpus=1, num_gpus=1)
class LearnerWorker:
    def __init__(self, model, device, pad_token_id, learner_step=0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.pad_token_id = pad_token_id
        self.learner_step = learner_step
 
    def train_groups(self, groups):
        batch = collate_groups(groups, pad_token_id=self.pad_token_id)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        
        # proximal policy my friends.
        with torch.no_grad():
            proximal_logprobs = score_batch(self.model, batch)

        metrics = train_step(
            model=self.model,
            optimizer=self.optimizer,
            batch=batch,
            proximal_logprobs=proximal_logprobs
        )

        self.learner_step += 1
        return metrics
