# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlOffPolicyRunnerCfg,
    RslRlSacActorCriticCfg,
    RslRlSacAlgorithmCfg,
)



@configclass
class AntonballPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    rewards_expect = {
        "rew_pole_moving": 1.0,
        # "rew_alive": 2.0,
    }
    gage_init_std = 0.0
    num_steps_per_env = 32
    max_iterations = 6000
    save_interval = 50
    experiment_name = "antonball"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic_RND",
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

@configclass
class AntonballSACRunnerCfg(RslRlOffPolicyRunnerCfg):
    rewards_expect = {
        "rew_pole_moving": 1.0,
        # "rew_alive": 2.0,
    }
    num_steps_per_env = 32
    max_iterations = 6000
    save_interval = 50
    experiment_name = "antonball"
    capacity_per_env = 300
    policy = RslRlSacActorCriticCfg(
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
        log_std_min=-5.0,
        log_std_max=5.0,
        use_layer_norm=False,
    )
    algorithm = RslRlSacAlgorithmCfg(
        alpha=1.0,
        num_learning_epochs=5,
        critic_lr=5.0e-4,
        actor_lr=5.0e-4,
        alpha_lr=5.0e-4,
        gamma=0.99,
        batch_size=8192,
        max_grad_norm=1.0,
        target_entropy=-10.0,
        tau=0.005,
        empirical_normalization=True,
    )