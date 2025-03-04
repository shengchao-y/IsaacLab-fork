# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid climb pole environment
"""

import gymnasium as gym

from . import agents, humanpole_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanpole-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": humanpole_env_cfg.HumanpoleEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HumanpolePPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
