# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from rl_games.common.algo_observer import IsaacAlgoObserver


class IsaacAlgoObserverWithSigma(IsaacAlgoObserver):
    """Logs sigma statistics from RL-Games continuous policies."""

    def __init__(self):
        super().__init__()

    def after_steps(self):
        if self.writer is None:
            return
        dataset = getattr(self.algo, "dataset", None)
        if dataset is None or not hasattr(dataset, "values_dict"):
            return
        sigmas = dataset.values_dict.get("sigma")
        if sigmas is None:
            return
        with torch.no_grad():
            sigma_mean = sigmas.mean().item()
        self.writer.add_scalar("policy/sigma_mean", sigma_mean, self.algo.frame)

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            reward_values = []
            for ep_info in self.ep_infos:
                total = sum(
                    float(value)
                    for key, value in ep_info.items()
                    if key.startswith("Episode_Reward/")
                )
                reward_values.append(total)
            if reward_values:
                reward_tensor = torch.tensor(reward_values, device=self.algo.device)
                reward_value = torch.mean(reward_tensor).item()
                self.writer.add_scalar("Episode/total_reward", reward_value, epoch_num)

        if self.writer is not None:
            mean_rewards = getattr(self.algo, "last_mean_rewards", None)
            if mean_rewards is not None:
                if not isinstance(mean_rewards, (list, tuple)):
                    mean_rewards = [float(mean_rewards)]
                    self.algo.last_mean_rewards = mean_rewards
                value = mean_rewards[0]
                self.writer.add_scalar("Episode/total_reward_scaled", float(value), epoch_num)
        super().after_print_stats(frame, epoch_num, total_time)
