from dataclasses import dataclass
from typing import List, Tuple, Any
import random
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions.categorical import Categorical
import torch.optim as optim
import numpy as np
import tyro
import time
import os
from itertools import combinations
import math
from torch.nn.functional import softmax
from src.belief_models.base import BeliefModel


from src.game.utils import (
    TeamVote, QuestVote,
)
from src.game.game_state import AvalonGameState
from src.players.player import AvalonPlayer
from src.models.ac_models import ActorCriticModel
from src.models.belief_predictor import BeliefPredictor
from src.game.utils import (
    Role, QuestResult, RoundStage, GameStage
)
    
class PPOAvalonPlayer(AvalonPlayer):
    """
    Player trained using PPO
    """

    def __init__(
        self,
        role: Role,
        index: int,
        actor: Any,
        belief_model: BeliefModel
    ) -> None:
        super().__init__(role, index)        
        self.role = role
        self.index = index
        self.actor = actor
        self.belief_model = belief_model
    
    def get_team_proposal(self, game_state: AvalonGameState) -> List[int]:
        action = self._get_action_numpy(game_state)
        team_size = game_state.team_size
        
        # get the top team_size indices
        team_inds = action.argsort()[-team_size:][::-1]
        return team_inds.tolist()
    
    def get_team_vote(self, game_state: AvalonGameState) -> TeamVote:
        action = self._get_action_numpy(game_state)
        # only use the first two indices
        return TeamVote.APPROVE if action[0] > action[1] else TeamVote.REJECT
    
    def get_quest_vote(self, game_state: AvalonGameState) -> QuestVote:
        action = self._get_action_numpy(game_state)
        # Good players cannot fail quests
        if self.role != Role.SPY:
            return QuestVote.SUCCESS
        
        # only use the first two indices
        return QuestVote.SUCCESS if action[0] > action[1] else QuestVote.FAIL
    
    def guess_merlin(self, game_state: AvalonGameState) -> int:
        belief = self.belief_model(game_state, self.index)
        top_belief_ind = np.argmax(belief.distribution)
            
        role_assignment = belief.all_assignments[top_belief_ind]
        merlin_index = role_assignment.index(Role.MERLIN)
        return merlin_index
    
    ###########
    # Helpers #
    ###########
    
    def _get_action_numpy(self, game_state: AvalonGameState) -> np.ndarray:
        obs = self._get_obs(game_state)
        action_npy, _ = self.actor.predict(obs)
        return action_npy
    
    def _get_obs(self, game_state: AvalonGameState) -> np.ndarray:
        
        # Role as one hot
        role_one_hot = np.zeros(3)
        role_inds = {Role.MERLIN: 0, Role.RESISTANCE: 1, Role.SPY: 2}
        role_one_hot[role_inds[self.role]] = 1
        
        # Index of agent as one hot
        agent_index_one_hot = np.zeros(5)
        agent_index_one_hot[self.index] = 1
        
        # Beliefs
        beliefs = self.belief_model(game_state, self.index).distribution
        
        # Action type as one hot
        action_type_one_hot = np.zeros(3)
        assert game_state.game_stage == GameStage.IN_PROGRESS, (
            f"Unexpected game stage: {game_state.game_stage}"
        )
        if game_state.round_stage == RoundStage.TEAM_PROPOSAL:
            action_type_one_hot[0] = 1
        elif game_state.round_stage == RoundStage.TEAM_VOTE:
            action_type_one_hot[1] = 1
        elif game_state.round_stage == RoundStage.QUEST_VOTE:
            action_type_one_hot[2] = 1

        # Leader
        leader = np.zeros(5)
        leader[game_state.leader_index] = 1
        
        # Team
        team = np.zeros(5)
        if game_state.round_stage == RoundStage.TEAM_VOTE or game_state.round_stage == RoundStage.QUEST_VOTE:
            team_inds = game_state.teams[-1]
            team[team_inds] = 1

        # Quest progress
        r_wins_progress = np.array([np.count_nonzero(game_state.quest_results == QuestResult.SUCCEEDED) / 3])
        s_wins_progress = np.array([np.count_nonzero(game_state.quest_results == QuestResult.FAILED) / 3])
        
        # Quest number
        quest_num = np.zeros(5)
        quest_num[game_state.quest_num] = 1

        # Turns until hammer
        turns_until_hammer = np.zeros(5)
        for i in range(game_state.turns_until_hammer[0]):
            turns_until_hammer[(game_state.leader_index + i) % 5] = i + 1
        turns_until_hammer = softmax(torch.Tensor(turns_until_hammer), dim = 0).numpy()

        # Concatenate all observations
        obs = np.concatenate([
            role_one_hot, agent_index_one_hot, beliefs, action_type_one_hot, leader, team, r_wins_progress, s_wins_progress, quest_num, turns_until_hammer
        ])
        
        return obs