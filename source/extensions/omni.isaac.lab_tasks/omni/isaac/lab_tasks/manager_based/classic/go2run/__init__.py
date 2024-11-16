# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2 locomotion environment using unitree Go2 robot.
"""

import gymnasium as gym

from . import agents, go2run_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Go2run-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2run_env_cfg.Go2runEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2runPPORunnerCfg",
        "rsl_rl_sac_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2runSACRunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
