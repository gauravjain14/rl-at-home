

class RolloutGroup:
    def __init__(self, group_id, attempt_id, run_epoch,
            source_learner_step, created_at, trajectories):
        """
            source_learner_step: The learner step represented by the
            generator weights used for this rollout group.
        """
        self.group_id = group_id
        self.attempt_id = attempt_id
        self.run_epoch = run_epoch
        self.created_at = created_at
        self.trajectories = trajectories
        self.source_learner_step = source_learner_step

    
class Trajectory:
    def __init__(self, prompt_ids, response_ids, behavior_logprobs,
                 loss_mask, reward, finish_reason, logp=None):
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        # Generator logprobs of the sampled response tokens.
        self.behavior_logprobs = behavior_logprobs
        # loss_mask - which tokens contribute to the loss.
        self.loss_mask = loss_mask
        self.reward = reward
        self.finish_reason = finish_reason
        self.debug_logp = logp