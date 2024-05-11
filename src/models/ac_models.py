import torch
from torch import nn
from src.game.game_state import AvalonGameState
from src.game.beliefs import all_possible_ordered_role_assignments
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.nn.functional import softmax
from src.game.utils import (
    Role, QuestResult, RoundStage, GameStage
)
from src.belief_models.base import BeliefModel
from src.game.utils import assignment_to_str
import numpy as np
from itertools import combinations
import math

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    """
    Initialize parameters of the network.
    m: torch.nn.Module
    """
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ActorCriticModel(nn.Module):
    def __init__(self, belief_model: BeliefModel):
        """
        Represents an Actor model that takes a Avalon game/belief state
        as input

        TEAM PROPOSAL:
        The action space for the team proposal is a num_players (5) length vector
        where the values at the indices corrosponding to each player are higher
        if the player is more favorable for selection: (5,) <- dist

        The observation space for the team proposal is a 40 length vector:
            role: (3,) <- one hot
            beliefs: (30,) <- distribution
            r_wins_progress: (1,) <- (0, 1]
            s_wins_progess: (1,) <- (0, 1]
            quest_num: (5,) <- one hot

        TEAM VOTE:
        The action space for the team vote is YES or NO: (2,) <- dist

        The observation space for the team vote is a 50 length vector:
            role: (3,) <- one hot
            beliefs: (30,) <- distribution
            who_leader: (5,) <- one hot
            team: (5,) <- binary
            r_wins_progress: (1,) <- (0, 1]
            s_wins_progess: (1,) <- (0, 1]
            turns_until_hammer: (5,) <- distribution

        QUEST VOTE:
        The action space for the quest vote is PASS or FAIL: (2,) <- dist

        The observation space for the quest vote is a 45 length vector
            role: (3,) <- one hot
            beliefs: (30,) <- distribution
            who_leader: (5,) <- one hot
            team: (5,) <- binary
            r_wins_progress: (1,) <- (0, 1]
            s_wins_progess: (1,) <- (0, 1]

        MERLIN VOTE:
        The action space for the merlin vote is a num_players (5) length vector
        where the values at the indices corrosponding to each player are higher
        if the player is more likely to be the merlin (5,) <- dist

        The observation space for the merlin vote is a 30 length vector:
            beliefs: (30,) <- distribution

        Note: no need to train a new model for the merlin vote, just select merlin
        from role permulation with highest belief value

        Parameters
        ----
        role: Role
            Player role in game: (spy, resistance, merlin)
        
        index: int
            Player index in game: [0, num_players-1]

        """
        super().__init__()
        self.belief_model = belief_model

        # Define actor models
        self.team_proposal_actor = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 5)
        )
        self.team_vote_actor = nn.Sequential(
            nn.Linear(50, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
        self.quest_vote_actor = nn.Sequential(
            nn.Linear(45, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

        # Define critic models
        self.team_proposal_critic = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.team_vote_critic = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.quest_vote_critic = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs: AvalonGameState, role: Role, index: int):
        """
        Performs a forward pass through the actor-critic network

        Parameters
        ----
        obs : AvalonGameState
        role : Role
        index : int

        ----

        returns:

        dist : torch.distribution
            The distribution of actions from policy. A Categorical distribution
            for discreet action spaces.
        """
        
        assert obs.player_assignments[index] == role, (
            f"Expected player {index} to have role {role}, in game but got {obs.player_assignments[index]}"
        )
        
        obs_in = self.format_observation(obs=obs, role=role, index=index).float()
        dist, value = None, None

        if obs.game_stage == GameStage.MERLIN_VOTE:
            # top_belief_ind = torch.argmax(obs_in)
            
            beliefs = self.belief_model(obs, index).distribution
            beliefs = torch.tensor(beliefs)
            top_belief_ind = torch.argmax(beliefs)
            
            # Problem spec
            roles = [
                Role.MERLIN,
                Role.RESISTANCE,
                Role.RESISTANCE,
                Role.SPY,
                Role.SPY,
            ]
            role_assignment = all_possible_ordered_role_assignments(roles)[top_belief_ind]
            merlin_index = role_assignment.index(Role.MERLIN)
            x = torch.zeros(5)
            x[merlin_index] = 1
            dist = Categorical(x)
            value = torch.tensor([0])
        if obs.round_stage == RoundStage.TEAM_PROPOSAL:
            x = self.team_proposal_actor(obs_in)
            dist = Categorical(logits=F.log_softmax(x, dim=0))
            value = self.team_proposal_critic(x)
        if obs.round_stage == RoundStage.TEAM_VOTE:
            x = self.team_vote_actor(obs_in)
            dist = Categorical(logits=F.log_softmax(x, dim=0))
            value = self.team_vote_critic(x)
        if obs.round_stage == RoundStage.QUEST_VOTE:
            x = self.quest_vote_actor(obs_in)
            dist = Categorical(logits=F.log_softmax(x, dim=0))
            value = self.quest_vote_critic(x)

        return dist, value

    def format_observation(self, obs: AvalonGameState, role: Role, index: int):
        
        assert obs.player_assignments[index] == role, (
            f"Expected player {index} to have role {role}, in game but got {obs.player_assignments[index]}"
        )
        
        role_one_hot = torch.zeros(3)
        role_inds = {Role.MERLIN: 0, Role.RESISTANCE: 1, Role.SPY: 2}
        role_one_hot[role_inds[role]] = 1
        
        """
        historical_obs = obs.game_state_obs(from_perspective_of_player_index=index)
        beliefs = softmax(self.belief_model(torch.tensor(historical_obs).float().unsqueeze(0)))[0]
        """
        beliefs = self.belief_model(obs, index).distribution
        beliefs = torch.tensor(beliefs)

        leader = torch.zeros(5)
        leader[obs.leader_index] = 1

        team = torch.zeros(5)
        if obs.round_stage == RoundStage.TEAM_VOTE or obs.round_stage == RoundStage.QUEST_VOTE:
            team_inds = obs.teams[-1]
            team[team_inds] = 1

        r_wins_progress = torch.tensor([np.count_nonzero(obs.quest_results == QuestResult.SUCCEEDED) / 3])
        s_wins_progress = torch.tensor([np.count_nonzero(obs.quest_results == QuestResult.FAILED) / 3])
        
        quest_num = torch.zeros(5)
        quest_num[obs.quest_num] = 1

        turns_until_hammer = torch.zeros(5)
        for i in range(obs.turns_until_hammer[0]):
            turns_until_hammer[(obs.leader_index + i) % 5] = i + 1
        turns_until_hammer = softmax(turns_until_hammer, dim = 0)

        formatted_obs = None
        if obs.game_stage == GameStage.MERLIN_VOTE:
            formatted_obs = beliefs
        if obs.round_stage == RoundStage.TEAM_PROPOSAL:
            formatted_obs = torch.cat([
                                role_one_hot,
                                beliefs,
                                r_wins_progress,
                                s_wins_progress,
                                quest_num
                            ])
        if obs.round_stage == RoundStage.TEAM_VOTE:
            formatted_obs = torch.cat([
                                role_one_hot,
                                beliefs,
                                leader,
                                team,
                                r_wins_progress,
                                s_wins_progress,
                                turns_until_hammer
                            ])
        if obs.round_stage == RoundStage.QUEST_VOTE:
            formatted_obs = torch.cat([
                                role_one_hot,
                                beliefs,
                                leader,
                                team,
                                r_wins_progress,
                                s_wins_progress
                            ])
        
        return formatted_obs.to(torch.float32)
    

    def evaluate_actions(self, states, player_roles, player_indices, actions):
        """
        Evaluate the actions taken by the policy given the states.

        Parameters:
        - states: list of torch.Tensor
            List of states, each represented as a torch.Tensor
        - actions: list of torch.Tensor
            List of actions, each represented as a torch.Tensor

        Returns:
        - log_probs: torch.Tensor
            Log probabilities of the actions taken by the policy
        - state_values: torch.Tensor
            Estimated state values for the given states
        - dist_entropy: torch.Tensor
            Entropy of the action distribution
        """
        log_probs = []
        state_values = []
        dist_entropy = 0
        
        for state, action, role, player_index in zip(states, actions, player_roles, player_indices):
            dist, value = self.forward(state, role, player_index) 
            action_tensor = torch.tensor(action)

            if state.round_stage == RoundStage.TEAM_PROPOSAL:
                selected_logits = dist.logits[action]
                action_log_prob = selected_logits.sum()
            else:
                action_log_prob = dist.log_prob(action_tensor)
            log_probs.append(action_log_prob.view(1))
            state_values.append(value)

            dist_entropy += dist.entropy().mean()
        
        return torch.cat(log_probs), torch.cat(state_values), dist_entropy