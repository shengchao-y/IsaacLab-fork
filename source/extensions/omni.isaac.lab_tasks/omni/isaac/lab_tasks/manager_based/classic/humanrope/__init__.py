# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion on tight rope environment (similar to OpenAI Gym Humanoid-v2).
"""

import gymnasium as gym

from . import agents, humanrope_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanrope-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": humanrope_env_cfg.HumanropeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanropePPORunnerCfg",
        "rsl_rl_sac_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanropeSACRunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
