import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from src.game.utils import Role
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
from src.players.ppo_player import PPOAvalonPlayer
from src.utils.belief_from_models import get_belief_for_player_cheap
from torch.distributions import Categorical
from src.game.game_state import AvalonGameState
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

def ppo_step(policy, optimizer, states, actions, log_probs_old, returns, advantage, clip_param=0.2):
    log_probs_new, state_values, dist_entropy = policy.evaluate_actions(states, actions)
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

class GroundTruthBeliefModel:
    def __init__(self, role: Role, index: int):
        self.role = role
        self.index = index

    def __call__(self, obs: AvalonGameState):
        belief = get_belief_for_player_cheap(obs, self.index, 'cpu')
        return belief.distribution


if __name__ == "__main__":
    EXPERIMENT_NAME = "action_debug_16_30_10"

    # Config
    is_spy = False
    roles = [
        Role.MERLIN,
        Role.RESISTANCE,
        Role.RESISTANCE,
        Role.SPY,
        Role.SPY,
    ]
    role = np.random.choice(roles)
    index = np.random.choice(len(roles))

    # Setting up belief model
    if role == Role.SPY:
        is_spy = True
    n_classes = len(all_possible_ordered_role_assignments(roles))

    # Setting up env
    env = AvalonEnv(roles, stupid_hardcoded_player_factory, index, role, True)

    # Setting up belief model
    new_belief_model = GroundTruthBeliefModel(role, index)

    # Setting up action model
    action_model = ActorCriticModel(new_belief_model, role, index)
    ppo_player = PPOAvalonPlayer(role, index, new_belief_model, env)
    optimizer = optim.Adam(action_model.parameters(), lr=0.01)

    # Dummy game loop for demonstration
    states, actions, rewards = [], [], []
    done = False
    state = env.reset()
    while not done:
        action = ppo_player.get_action(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # ACTUALLY TRAIN THE PPO AGENT
    n_episodes = 10000  # number of episodes to train
    max_timesteps = 1000  # max timesteps in one episode
    gamma = 0.99  # discount factor
    tau = 0.95  # factor for GAE
    clip_param = 0.2  # PPO clip parameter

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        states, actions, rewards, log_probs_old, masks, values = [], [], [], [], [], []
        states.append(deepcopy(state))
        while not done:
            action, log_prob, value = ppo_player.get_action_probs_and_value(state)
            next_state, reward, done, _ = env.step(action)

            states.append(deepcopy(state))
            actions.append(action)
            rewards.append(reward)
            log_probs_old.append(log_prob)
            values.append(value)
            masks.append(1 - int(done))

            state = next_state

        log_probs_old = torch.stack(log_probs_old)
        next_value = ppo_player.get_value(next_state)
        returns, advantages = compute_gae(next_value, rewards, masks, values, gamma, tau)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        
        loss = ppo_step(action_model, optimizer, states, actions, log_probs_old, returns, advantages, clip_param)
        print(f"Episode {episode + 1}, Loss: {loss:.5f}")
