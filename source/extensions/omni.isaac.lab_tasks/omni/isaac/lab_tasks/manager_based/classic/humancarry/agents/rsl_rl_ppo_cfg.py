# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class GoalReward:
    name: str = "rew"
    max_value: float = 0.0
    gage_init_std: float = 0.0
    gage_change_rate: float = 0.0


@configclass
class HumancarryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    rewards_expect = {
        # TODO: simple goal achievement aggregation may not work for humancarry,
        # because it collect different reward terms in different periods
        "rew_hand2box": GoalReward(name="rew_hand2box", 
                                   max_value=0.1, gage_init_std=1.0, gage_change_rate=1.0),
        "rew_handonbox": GoalReward(name="rew_handonbox", 
                                    max_value=1.0, gage_init_std=0.8, gage_change_rate=1.0),
        "rew_box2target": GoalReward(name="rew_box2target", 
                                     max_value=0.6, gage_init_std=0.6, gage_change_rate=1.0),
        # "rew_boxontarget": GoalReward(name="rew_boxontarget", 
        #                               max_value=0.8, gage_init_std=0.4, gage_change_rate=1.0),
    }
    # gage_init_std = 0.0
    # gage_change_rate = 0.0
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 50
    experiment_name = "humancarry"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
