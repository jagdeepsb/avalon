import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Callable, Tuple, Dict

from src.game.utils import Role, RoundStage
from src.models.ac_models import ActorCriticModel
from src.datasets.belief_dataset import BeliefDataset
from src.models.belief_predictor import BeliefPredictor
from src.game.beliefs import all_possible_ordered_role_assignments
from src.envs.avalon_env import AvalonEnv
from src.utils.constants import (
    MODELS_DIR,
    RES_TRAIN_BELIEF_DATASET_PATH,
    RES_VAL_BELIEF_DATASET_PATH,
    SPY_TRAIN_BELIEF_DATASET_PATH,
    SPY_VAL_BELIEF_DATASET_PATH,
)
from src.players.stupid_hardcoded_player import stupid_hardcoded_player_factory
from src.players.random_player import random_player_factory
from src.players.ppo_player import PPOAvalonPlayer
from torch.distributions import Categorical
from src.game.game_state import AvalonGameState
from src.players.player import AvalonPlayer
from src.game.arena import AvalonArena
from src.belief_models.trivial import GroundTruthBeliefModel
from copy import deepcopy

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values.append(next_value)
    gae = 0
    returns = []
    advantages = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
        advantages.insert(0, gae)
    return returns, advantages

def ppo_step(policy, optimizer, states, player_roles, player_indices, actions, log_probs_old, returns, advantage, clip_param=0.2):
    log_probs_new, state_values, dist_entropy = policy.evaluate_actions(states, player_roles, player_indices, actions)
    ratios = torch.exp(log_probs_new - log_probs_old)
    surr1 = ratios * advantage
    surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = 0.5 * (returns - state_values).pow(2).mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * dist_entropy.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

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
    n_classes = len(all_possible_ordered_role_assignments(game_player_roles))

    # Setting up env
    env = AvalonEnv(
        roles=game_player_roles,
        bot_player_factory=bot_player_factory,
        randomize_player_assignments=True
    )

    # Setting up belief model
    belief_model = GroundTruthBeliefModel()

    # Setting up action model
    action_model = ActorCriticModel(belief_model)
    
    # For validation
    def ppo_player_factory(role: Role, index: int) -> PPOAvalonPlayer:
        return PPOAvalonPlayer(role, index, action_model, env)

    # Dummy game loop for demonstration
    states, player_roles, player_indices, actions, rewards = [], [], [], [], []
    for iter in range(1000):
        print(iter)
        done = False
        state, ppo_player_role, ppo_player_index = env.reset()
        ppo_player = PPOAvalonPlayer(ppo_player_role, ppo_player_index, action_model, env)
        while not done:
            action = ppo_player.get_action(state)
            next_state, reward, done, _ = env.step(action)
            states.append(deepcopy(state))
            player_roles.append(ppo_player_role)
            player_indices.append(ppo_player_index)
            actions.append(action)
            rewards.append(reward)
            state = next_state

    actions_dict_proposal = {}
    actions_dict_team_vote = {}
    actions_dict_quest_vote = {}
    num_proposal = 0
    num_team = 0
    num_quest = 0
    for i in range(len(actions)):
        a = actions[i]
        if type(a) == list:
            a = tuple(a)
        s = states[i]
        round_stage = s.round_stage
        if round_stage == RoundStage.TEAM_PROPOSAL:
            num_proposal += 1
            if a in actions_dict_proposal:
                actions_dict_proposal[a] += 1
            else:
                actions_dict_proposal[a] = 1
        if round_stage == RoundStage.TEAM_VOTE:
            num_team += 1
            if a in actions_dict_team_vote:
                actions_dict_team_vote[a] += 1
            else:
                actions_dict_team_vote[a] = 1
        if round_stage == RoundStage.QUEST_VOTE:
            num_quest += 1
            if a in actions_dict_quest_vote:
                actions_dict_quest_vote[a] += 1
            else:
                actions_dict_quest_vote[a] = 1

    print("proposal avg actions")
    for a in actions_dict_proposal:
        print(a, actions_dict_proposal[a] / num_proposal)
    print("team vote avg actions")
    for a in actions_dict_team_vote:
        print(a, actions_dict_team_vote[a] / num_team)
    print("quest vote avg actions")
    for a in actions_dict_quest_vote:
        print(a, actions_dict_quest_vote[a] / num_quest)
