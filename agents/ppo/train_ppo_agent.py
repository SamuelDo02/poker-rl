def rollout(self, env):
    """
    Rollouts a policy for one round.
    """
    trajectories, payoffs = game_env.run(is_training=True)

def train(self, states):
    for i in tqdm(range(self.num_iters)):
        # make a new dim for num_actors to do rollouts in parallel
        actors_states = states.reshape([states.shape[0], self.num_actors, 1])
        # rollout current policy in environment for T timesteps
        cur_log_probs, prev_log_probs, advantages, states = self.rollout(actors_states, self.num_timesteps)
        ratios = torch.exp(cur_log_probs - prev_log_probs)
        surrogate_loss = self.policy.compute_surrogate_loss(ratios, advantages, self.epsilon)
        for k in self.num_epochs:
            surrogate_loss.backward()
        self.old_policy = self.policy

def main():
    pass

if __name__ == '__main__':
    main()