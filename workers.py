import json
import torch
import ray
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from rollout import collect_rollout_groups, generate_rollouts
from batching import collate_groups
from learner import train_step, score_batch


@ray.remote(num_cpus=1, num_gpus=1)
class SyncWorker:
    # what is needed for the sync worker?
    # model, optimizer, etc.
    # the rollout worker collects the rollout groups, load
    # weights and gets the policy version. rollout
    # also runs the evals
    def __init__(self, model, device, pad_token_id, learner_step=0,
                 *, tokenizer, score_fn, num_gens=8, answer_parser=None, lr=1e-6,
                 seed=42):
        if not callable(score_fn):
            raise ValueError("score_fn must be callable")

        torch.manual_seed(seed)
        self.model = model.to(device=device, dtype=torch.float32).eval()
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.pad_token_id = pad_token_id
        self.learner_step = learner_step
        self.tokenizer = tokenizer
        self.score_fn = score_fn
        self.answer_parser = answer_parser
        self.num_gens = num_gens

        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_token_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())

    def train_groups(self, groups):
        # collate the groups and do a trainer step
        batch = collate_groups(groups, pad_token_id=self.pad_token_id)
        batch = {key: value.to(self.device) for key, value in batch.items()}

        with torch.no_grad():
            proximal_rows = []
            for i in range(batch["input_ids"].shape[0]):
                micro_batch = {
                    key: value[i:(i+1)]
                    for key, value in batch.items()
                }
                proximal_rows.append(score_batch(self.model, micro_batch))
            proximal_logprobs = torch.cat(proximal_rows, dim=0)

        metrics = train_step(
            model=self.model,
            optimizer=self.optimizer,
            batch=batch,
            proximal_logprobs=proximal_logprobs
        )
        trajectories = [trajectory for group in groups for trajectory in group.trajectories]
        mask = batch["loss_mask"][:, 1:]
        behavior = batch["behavior_logprobs"][:, 1:][mask]
        metrics.update({
            "reward_mean": batch["rewards"].mean().item(),
            "equal_reward_group_fraction": sum(
                len({trajectory.reward for trajectory in group.trajectories}) == 1
                for group in groups) / len(groups),
            "truncated_fraction": sum(
                trajectory.finish_reason == "length" for trajectory in trajectories) / len(trajectories),
            "policy_lag_max": max(self.learner_step - group.source_learner_step for group in groups),
            "logprob_gap_mean": (proximal_logprobs[mask] - behavior).abs().mean().item(),
            # mean NLL of sampled tokens -
            "sampled_nll": -behavior.mean().item(),
            "response_len": mask.sum(1).float().mean().item(),
        })
        self.learner_step += 1
        return metrics

    def collect_group(self, tasks, input_prompts, *, group_id, run_epoch,
                      max_samples_per_gen=1024):
        inputs = [input_prompt.to(self.device) for input_prompt in input_prompts]
        rollout_groups = collect_rollout_groups(
            tasks=tasks,
            input_prompts=inputs,
            behavior_policy=self.model,
            tokenizer=self.tokenizer,
            score_fn=self.score_fn,
            group_id=group_id,
            run_epoch=run_epoch,
            source_learner_step=self.learner_step,
            num_gens=self.num_gens,
            max_samples_per_gen=max_samples_per_gen,
            eos_token_ids=self.eos_token_ids,
            pad_token_id=self.pad_token_id,
        )
        return rollout_groups

    @torch.no_grad()
    def evaluate(self, inputs, max_samples_per_gen=512, *, output_path=None):
        if not inputs:
            raise ValueError("Evaluation needs at least one prompt")
        rewards, truncated, parse_failures, samples = [], 0, 0, []
        rollouts = generate_rollouts(
            [inp_prompt["input_ids"][0].to(self.device) for _, inp_prompt in inputs],
            self.model,
            num_gens=1,
            max_samples_per_gen=max_samples_per_gen,
            eos_token_ids=self.eos_token_ids,
            pad_token_id=self.pad_token_id,
            greedy=True,
        )
        for (task, inp_prompt), rollout in zip(inputs, rollouts):
            response_ids = rollout["response_ids"]
            text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            reward = float(self.score_fn(task, text))
            rewards.append(reward)
            hit_limit = not response_ids or response_ids[-1] not in self.eos_token_ids
            truncated += hit_limit
            parsed = self.answer_parser(text) if self.answer_parser is not None else None
            parse_failures += parsed is None
            if output_path is not None:
                samples.append({
                    "question_id": task.get("question_id"), "prompt": task.get("prompt"),
                    "expected_answer": task.get("answer"), "response": text,
                    "parsed_answer": str(parsed) if parsed is not None else None,
                    "reward": reward, "finish_reason": "length" if hit_limit else "stop",
                    "response_tokens": len(response_ids),
                })
        metrics = {
            "accuracy": sum(rewards) / len(rewards),
            "truncated_fraction": truncated / len(rewards),
            "num_examples": len(rewards),
            "policy_version": self.learner_step,
        }
        if self.answer_parser is not None:
            metrics["parse_failure_fraction"] = parse_failures / len(rewards)
        if output_path is not None:
            Path(output_path).write_text(json.dumps({"metrics": metrics, "samples": samples}, indent=2))
        return metrics

    def save_checkpoint(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save({
            "learner_step": self.learner_step,
            "optimizer": self.optimizer.state_dict(),
        }, path / "trainer.pt")
        return str(path)


@ray.remote(num_cpus=1, num_gpus=1)
class RolloutWorker:
    def __init__(self, model_name, score_fn, num_gens=4):
        if not callable(score_fn):
            raise ValueError("score_fn must be callable")

        self.device = "cuda:0"
        torch.manual_seed(42)
        # initialize a copy of the model here which is then weight sync'd
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map={"": self.device},
            attn_implementation="sdpa",
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
    
    @torch.no_grad()
    def load_weights(self, snapshot):
        version = snapshot["learner_step"]
        if version < self.policy_version:
            raise ValueError("Cannot load an older policy")
        # Ordinary Ray actor methods are serial: no collect_group runs here.
        torch.cuda.synchronize(self.device)
        self.model.load_state_dict(snapshot["state_dict"], strict=True)
        torch.cuda.synchronize(self.device)
        self.policy_version = version
        return self.policy_version

    @torch.no_grad()
    def evaluate(self, inputs, max_samples_per_gen=512):
        if not inputs:
            raise ValueError("Evaluation needs at least one prompt")
        rewards, truncated = [], 0
        for task, inp_prompt in inputs:
            prompt = {key: value.to(self.device) for key, value in inp_prompt.items()}
            output = self.model.generate(
                **prompt, do_sample=False, max_new_tokens=max_samples_per_gen,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=list(self.eos_token_ids),
            )
            response_ids = output[0, prompt["input_ids"].shape[1]:].tolist()
            text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            rewards.append(float(self.score_fn(task, text)))
            truncated += not response_ids or response_ids[-1] not in self.eos_token_ids
        return {
            "accuracy": sum(rewards) / len(rewards),
            "truncated_fraction": truncated / len(rewards),
            "num_examples": len(rewards),
            "policy_version": self.policy_version,
        }


@ray.remote(num_cpus=1, num_gpus=1)
class LearnerWorker:
    def __init__(self, model, device, pad_token_id, learner_step=0):
        self.model = model.to(device=device, dtype=torch.float32).eval()
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6, weight_decay=0.0)
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

        trajectories = [trajectory for group in groups 
                        for trajectory in group.trajectories]
        metrics.update({
            "reward_mean": batch["rewards"].mean().item(),
            "equal_reward_group_fraction": sum(
                len({trajectory.reward for trajectory in group.trajectories}) == 1
                for group in groups
            ) / len(groups),
            "truncated_fraction": sum(
                trajectory.finish_reason == "length" for trajectory in trajectories
            ) / len(trajectories),
            "policy_lag_max": max(self.learner_step - group.source_learner_step for group in groups),
        })
        self.learner_step += 1
        return metrics

    def export_weights(self):
        return {
            "learner_step": self.learner_step,
            "state_dict": {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in self.model.state_dict().items()
            },
        }

    def save_checkpoint(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(path)
        torch.save({
            "learner_step": self.learner_step,
            "optimizer": self.optimizer.state_dict(),
        }, path / "trainer.pt")
        return str(path)
