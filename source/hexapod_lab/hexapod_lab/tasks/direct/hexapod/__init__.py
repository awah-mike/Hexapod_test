# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Flat-terrain velocity tracking task for the custom hexapod."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Hexapod-Direct-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
    },
)
