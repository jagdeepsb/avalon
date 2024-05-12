
from src.envs import *
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Callable, Tuple, Dict
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, SubprocVecEnv

from src.utils.constants import MODELS_DIR
from src.game.utils import Role
from src.models.ac_models import ActorCriticModel
from src.players.stupid_hardcoded_player import stupid_hardcoded_player_factory
from src.players.random_player import random_player_factory
from torch.distributions import Categorical
from src.game.game_state import AvalonGameState
from src.players.player import AvalonPlayer
from src.game.arena import AvalonArena
from src.belief_models.trivial import GroundTruthBeliefModel
from src.game.utils import GameStage
from copy import deepcopy
from src.players.ppo_player import PPOAvalonPlayer


from stable_baselines3.common.callbacks import BaseCallback


if __name__ == "__main__":
    EXPERIMENT_NAME = "ppo_avalon"
    
    roles = [
        Role.MERLIN,
        Role.RESISTANCE,
        Role.RESISTANCE,
        Role.SPY,
        Role.SPY,
    ]
    
    model = PPO.load(os.path.join(MODELS_DIR, EXPERIMENT_NAME))
    def ppo_player_factory(role: Role, index: int) -> PPOAvalonPlayer:
        return PPOAvalonPlayer(role, index, model.policy, GroundTruthBeliefModel())
    
    # bot_player_factory = stupid_hardcoded_player_factory
    def bot_player_factory(role: Role, index: int) -> AvalonPlayer:
        if role != Role.SPY:
            return stupid_hardcoded_player_factory(role, index)
        else:
            return random_player_factory(role, index)

    arena = AvalonArena(
        roles=roles,
        player_factory_1=ppo_player_factory,
        player_factory_2=bot_player_factory,
        num_games=1000,
        exactly_n_strategy_1=1,
    )

    strategy_names = ["PPO", "BOT"]

    for win_rates, label in zip(arena.get_win_rates_by_role(), strategy_names):
        print(f"{label} win rates:")
        for role, win_rate in win_rates.items():
            print(f"{role}: {win_rate}")
        print()
        
    for win_rate, label in zip(arena.get_overall_win_rates(), strategy_names):
        print(f"{label} win rate: {win_rate}")

    