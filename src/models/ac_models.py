import torch
from torch import nn
from src.game.game_state import AvalonGameState
from src.models.belief_predictor import BeliefPredictor
from src.game.beliefs import all_role_assignments
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.nn.functional import softmax
from src.game.utils import (
    Role, QuestResult, RoundStage, GameStage
)
import numpy as np

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
    def __init__(self, belief_model: BeliefPredictor, role: Role, index: int):
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
        self.role = role
        self.index = index

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
            nn.Linear(40, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.team_vote_critic = nn.Sequential(
            nn.Linear(50, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.quest_vote_critic = nn.Sequential(
            nn.Linear(45, 64),
            nn.ReLU(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs: AvalonGameState):
        """
        Performs a forward pass through the actor-critic network

        Parameters
        ----
        obs : AvalonGameState

        ----

        returns:

        dist : torch.distribution
            The distribution of actions from policy. A Categorical distribution
            for discreet action spaces.
        """
        obs_in = self.format_observation(obs=obs)

        dist, value = None, None

        if obs.game_stage == GameStage.MERLIN_VOTE:
            top_belief_ind = torch.argmax(obs_in)
            # Problem spec
            roles = [
                Role.MERLIN,
                Role.RESISTANCE,
                Role.RESISTANCE,
                Role.SPY,
                Role.SPY,
            ]
            role_assignment = all_role_assignments(roles)[top_belief_ind]
            merlin = role_assignment[0]
            x = torch.zeros(5)
            x[merlin] = 1
            dist = Categorical(x)
            value = torch.tensor([0])
        if obs.round_stage == RoundStage.TEAM_PROPOSAL:
            x = self.team_proposal_actor(obs_in)
            dist = Categorical(logits=F.log_softmax(x, dim=1))
            value = self.team_proposal_critic(x).squeeze(1)
        if obs.round_stage == RoundStage.TEAM_VOTE:
            x = self.team_vote_actor(obs_in)
            dist = Categorical(logits=F.log_softmax(x, dim=1))
            value = self.team_vote_critic(x).squeeze(1)
        if obs.round_stage == RoundStage.QUEST_VOTE:
            x = self.quest_vote_actor(obs_in)
            dist = Categorical(logits=F.log_softmax(x, dim=1))
            value = self.quest_vote_critic(x).squeeze(1)

        return dist, value

    def format_observation(self, obs: AvalonGameState):
        role = torch.zeros(3)
        role_inds = {Role.MERLIN: 0, Role.RESISTANCE: 1, Role.SPY: 2}
        role[role_inds[self.role]] = 1
        
        historical_obs = obs.game_state_obs(from_perspective_of_player_index=self.index)
        beliefs = softmax(self.belief_model(torch.tensor(historical_obs).float().unsqueeze(0)))[0]

        leader = torch.zeros(5)
        leader[obs.leader_index] = 1

        team_inds = obs.teams[-1]
        team = torch.zeros(5)
        team[team_inds] = 1

        r_wins_progress = torch.from_numpy([np.count_nonzero(obs.quest_results == QuestResult.SUCCEEDED) / 3])
        s_wins_progress = torch.from_numpy([np.count_nonzero(obs.quest_results == QuestResult.FAILED) / 3])
        
        quest_num = torch.zeros(5)
        quest_num[obs.quest_num] = 1

        turns_until_hammer = torch.zeros(5)
        for i in range(obs.turns_until_hammer):
            turns_until_hammer[(obs.leader_index + i) % 5] = i + 1
        turns_until_hammer = softmax(turns_until_hammer)

        formatted_obs = None
        if obs.game_stage == GameStage.MERLIN_VOTE:
            formatted_obs = beliefs
        if obs.round_stage == RoundStage.TEAM_PROPOSAL:
            formatted_obs = torch.cat([
                                role,
                                beliefs,
                                r_wins_progress,
                                s_wins_progress,
                                quest_num
                            ])
        if obs.round_stage == RoundStage.TEAM_VOTE:
            formatted_obs = torch.cat([
                                role,
                                beliefs,
                                leader,
                                team,
                                r_wins_progress,
                                s_wins_progress,
                                turns_until_hammer
                            ])
        if obs.round_stage == RoundStage.QUEST_VOTE:
            formatted_obs = torch.cat([
                                role,
                                beliefs,
                                leader,
                                team,
                                r_wins_progress,
                                s_wins_progress
                            ])
        
        return formatted_obs
