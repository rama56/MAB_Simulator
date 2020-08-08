def analyse_regret(self):
    # Compute cumulative rewards
    for t in range(1, self.T + 1):
        cumulative_reward_after_t = self.cum_reward_empirical[t - 1] + self.rewards[t]
        self.cum_reward_empirical[t] = cumulative_reward_after_t

    # Compute deltas and theoretical upper bound of regret of UCB1.
    self.best_arm = mh.get_maximum_index(self.true_means)
    mean_of_best_arm = self.true_means[self.best_arm]

    for i in range(self.K):
        self.deltas[i] = mean_of_best_arm - self.true_means[i]

    sum_del_inv, sum_del = mh.get_instance_dependent_values(self.best_arm, self.deltas)

    mult_constant = 8 * sum_del_inv
    addi_constant = mh.func_of_pi(add=1, power=2, mult=1 / 3) * sum_del

    for t in range(1, self.T + 1):
        cum_regret_theo_bound_after_t = mult_constant * mh.natural_logarithm(t) + addi_constant
        self.cum_regret_theo_bound[t] = cum_regret_theo_bound_after_t

    # Compute empirical regret
    for t in range(1, self.T + 1):
        self.cum_optimal_reward[t] = self.cum_optimal_reward[t - 1] + mean_of_best_arm
        # = t * mean_of_best_arm

        self.cum_regret_empirical[t] = self.cum_optimal_reward[t] - self.cum_reward_empirical[t]