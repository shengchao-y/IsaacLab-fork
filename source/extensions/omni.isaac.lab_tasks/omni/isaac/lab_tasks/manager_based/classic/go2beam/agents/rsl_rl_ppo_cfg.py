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
class Go2beamPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    rewards_expect = {
        "rew_progress": 2.0,
        # "rew_alive": 1.0,
        # "rew_orientation": 1.0,
    }
    num_steps_per_env = 32
    max_iterations = 8000
    save_interval = 50
    experiment_name = "go2beam"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
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
class Go2beamSACRunnerCfg(RslRlOffPolicyRunnerCfg):
    rewards_expect = {
        "rew_progress": 2.0,
        # "rew_alive": 1.0,
        # "rew_orientation": 1.0,
    }
    num_steps_per_env = 5
    max_iterations = 50000
    save_interval = 50
    experiment_name = "go2beam"
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
        critic_lr=3.0e-4,
        actor_lr=3.0e-4,
        alpha_lr=3.0e-4,
        gamma=0.99,
        batch_size=2048,
        max_grad_norm=1.0,
        target_entropy=-10.0,
        tau=0.005,
        empirical_normalization=True,
    )