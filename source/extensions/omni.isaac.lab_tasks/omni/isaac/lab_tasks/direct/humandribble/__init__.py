# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid dribbling environment.
"""

import gymnasium as gym

from . import agents
from .humandribble_env import HumandribbleEnv, HumandribbleEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humandribble-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.humandribble:HumandribbleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HumandribbleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumandribblePPORunnerCfg",
        "rsl_rl_sac_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumandribbleSACRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
