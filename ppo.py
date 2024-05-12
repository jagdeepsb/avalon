
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

    
    
class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        
        # every 10000 steps, validate the model
        if self.num_timesteps % 10000 == 0:
            def ppo_player_factory(role: Role, index: int) -> PPOAvalonPlayer:
                return PPOAvalonPlayer(role, index, self.model.policy, GroundTruthBeliefModel())
            validate(game_player_roles, ppo_player_factory, bot_player_factory)
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

def validate(        
        roles: List[Role],
        ppo_player_factory: Callable[[Role], AvalonPlayer],
        baseline_player_factory: Callable[[Role], AvalonPlayer],
        num_games: int = 100
    ) -> Tuple[Dict[Role, float], float]:
    """
    Run a series of games between two strategies and compute win rates:
    
    Returns
    - win rates by role: Dict[Role, float]
    - overall win rate: float
    """
    arena = AvalonArena(
        roles=roles,
        player_factory_1=ppo_player_factory,
        player_factory_2=baseline_player_factory,
        num_games=num_games,
        exactly_n_strategy_1=1, # Games should have only 1 PPO player
    )
    ppo_win_rates_by_role = arena.get_win_rates_by_role()[0]
    ppo_win_rate = arena.get_overall_win_rates()[0]
    
    for win_rates, label in zip(arena.get_win_rates_by_role(), ["PPO", "BOT"]):
        print(f"{label} win rates:")
        for role, win_rate in win_rates.items():
            print(f"{role}: {win_rate}")
        print()
        
    for win_rate, label in zip(arena.get_overall_win_rates(), ["PPO", "BOT"]):
        print(f"{label} win rate: {win_rate}")
    
    return ppo_win_rates_by_role, ppo_win_rate


if __name__ == "__main__":
    EXPERIMENT_NAME = "ppo"

    # Config
    game_player_roles = [
        Role.MERLIN,
        Role.RESISTANCE,
        Role.RESISTANCE,
        Role.SPY,
        Role.SPY,
    ]
    bot_player_factory = random_player_factory
    # bot_player_factory = stupid_hardcoded_player_factory

    # Setting up belief model
    belief_model = GroundTruthBeliefModel()
    
    # Setting up env
    # env = gym.make(
    #     'Avalon-v0',
    #     roles=game_player_roles,
    #     belief_model=belief_model,
    #     bot_player_factory=bot_player_factory,
    #     randomize_player_assignments=True,
    #     # verbose=True
    # )
    
    vec_env = make_vec_env(
        'Avalon-v0',
        n_envs=8,
        env_kwargs={
            'roles': game_player_roles,
            'belief_model': belief_model,
            'bot_player_factory': bot_player_factory,
            'randomize_player_assignments': True,
            # 'verbose': True
        }
    )
    
    callback = CustomCallback()
    
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100000, callback=callback)
    model.save("ppo_avalon")