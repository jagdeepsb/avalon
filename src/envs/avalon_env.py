import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torch.nn.functional import softmax
import torch

from src.game.simulator import AvalonSimulator
from src.game.game_state import AvalonGameState
from src.game.utils import (
    Role, TeamVote, TeamVoteResult, QuestVote,
    QuestResult, RoundStage, GameStage,
)
from src.players.player import AvalonPlayer
from typing import List, Callable, Tuple, Dict, Optional, Union
from src.game.utils import Role
from src.game.simulator import AvalonSimulator
from src.players.random_player import RandomAvalonPlayer
import random
from copy import deepcopy
from src.belief_models.base import BeliefModel

class AvalonEnv(gym.Env):
    """
    Custom Environment for the game Avalon that follows gym interface.
    """

    def __init__(
        self,
        roles: List[Role],
        belief_model: BeliefModel,
        bot_player_factory: Callable[[Role, int], AvalonPlayer],
        randomize_player_assignments=True,
        player_role: Optional[Role] = None,
        verbose=False
    ):
        """
        Initialize the environment

        Input:
        - roles: List of roles for the game
        - bot_player_factory: Factory function that returns a bot player instance
        - randomize_player_assignments: Whether to randomize the player assignments
        - verbose: Whether to print game state information
        
        """
        super(AvalonEnv, self).__init__()
        
        # Initialize state
        self.roles = roles
        self.belief_model = belief_model
        self.bot_player_factory = bot_player_factory
        self.randomize_player_assignments = randomize_player_assignments
        self._forced_player_role = player_role
        self.verbose = verbose
        
        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=1, shape=(63,), dtype=np.float64)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(13,), dtype=np.float64)

        self.reset()

    def reset(self, seed: Optional[int] = None, options = None) -> np.ndarray:
        """
        Reset the environment:
        - Reset the game state
        - Randomly shuffle player assignments (if randomize_player_assignments=True)
        - Assign the agent to a random role and player index
        
        Returns
        - np.ndarray: The observation for the agent
        """
        
        # Compute player assignments
        player_assignments = self.roles.copy()
        if self.randomize_player_assignments:
            random.shuffle(player_assignments)
            
        # Assign agent role and index
        self.agent_index = np.random.choice(len(player_assignments))
        self.agent_role = player_assignments[self.agent_index]
        if self._forced_player_role is not None:
            while self.agent_role != self._forced_player_role:
                self.agent_index = np.random.choice(len(player_assignments))
                self.agent_role = player_assignments[self.agent_index]
                
        # Create simulator
        self.player_assignments = player_assignments
        self.game_state = AvalonGameState(
            self.player_assignments,
            randomize_player_assignments=False, 
            verbose=self.verbose
        )

        # Initialize bot players
        self.num_players = len(self.player_assignments)
        self.players: dict[int, AvalonPlayer] = {}
        for i, role in enumerate(self.player_assignments):
            if i != self.agent_index:
                self.players[i] = self.bot_player_factory(role, i)
        
        # Move the environment to the first state where the agent's action is required
        self._step_until_agent_action_required()
        
        # If the game has ended without requiring any agent action, reset the environment again
        if self._is_game_over():
            return self.reset()
                
        info = {}
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        The action method where the game dynamics are implemented
        
        returns: observation for the agent (np.ndarray)
        """
        
        # We must be in a state where the agent's action is required
        assert self._is_agent_action_required()
        
        # Action from numpy into the appropriate format (depending on game state)
        action = self._parse_action(action)
        
        # Take the action
        if self.game_state.game_stage == GameStage.IN_PROGRESS:
            if self.game_state.round_stage == RoundStage.TEAM_PROPOSAL:
                self._team_proposal_step(action)
            elif self.game_state.round_stage == RoundStage.TEAM_VOTE:
                self._team_vote_step(action)
            elif self.game_state.round_stage == RoundStage.QUEST_VOTE:
                self._quest_vote_step(action)
        elif self.game_state.game_stage == GameStage.MERLIN_VOTE:
            self._merlin_vote_step(action)
            
        # Move the environment to the next state where the agent's action is required
        self._step_until_agent_action_required()  

        # Calculate reward, done, and any additional info
        reward = 0
        done = False
        truncated = False
        info = {}
        if self.game_state.game_stage == GameStage.SPY_WIN:
            done = True
            if self.player_assignments[self.agent_index] == Role.SPY:
                reward = 1.0
        elif self.game_state.game_stage == GameStage.RESISTANCE_WIN:
            done = True
            if self.player_assignments[self.agent_index] == Role.RESISTANCE or self.player_assignments[self.agent_index] == Role.MERLIN:
                reward = 1.0
                
        if self._is_game_over():
            obs = np.zeros(self.observation_space.shape)
        else:
            obs = self._get_obs()
              
        return obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment
        """
        pass
    
    ####################
    # Parse the Action #
    ####################
    
    def _parse_action(self, action: np.ndarray) -> Union[List[int], TeamVote, QuestVote]:
        """
        Parse the action from the agent, depending on the game state
        action: np.ndarray of shape (5,)
        """
        
        assert self.game_state.game_stage == GameStage.IN_PROGRESS
        assert action.shape == (5,)
        
        if self.game_state.round_stage == RoundStage.TEAM_PROPOSAL:
            team_size = self.game_state.team_size
            # get the top team_size indices
            team_inds = action.argsort()[-team_size:][::-1]
            return team_inds.tolist()
        if self.game_state.round_stage == RoundStage.TEAM_VOTE:
            # only use the first two indices
            return TeamVote.APPROVE if action[0] > action[1] else TeamVote.REJECT
        if self.game_state.round_stage == RoundStage.QUEST_VOTE:
            # Good players cannot fail quests
            if self.agent_role != Role.SPY:
                return QuestVote.SUCCESS
            
            # only use the first two indices
            return QuestVote.SUCCESS if action[0] > action[1] else QuestVote.FAIL
        raise ValueError(f"Unknown round stage: {self.game_state.round_stage}")
        
    ###############
    # Observation #
    ###############
    
    def _get_obs(self) -> np.ndarray:
        """
        Get the observation for the agent
        """
        
        # Role as one hot
        role_one_hot = np.zeros(3)
        role_inds = {Role.MERLIN: 0, Role.RESISTANCE: 1, Role.SPY: 2}
        role_one_hot[role_inds[self.agent_role]] = 1
        
        # Index of agent as one hot
        agent_index_one_hot = np.zeros(5)
        agent_index_one_hot[self.agent_index] = 1
        
        # Beliefs
        beliefs = self.belief_model(self.game_state, self.agent_index).distribution
        
        # Action type as one hot
        action_type_one_hot = np.zeros(3)
        assert self.game_state.game_stage == GameStage.IN_PROGRESS, (
            f"Unexpected game stage: {self.game_state.game_stage}"
        )
        if self.game_state.round_stage == RoundStage.TEAM_PROPOSAL:
            action_type_one_hot[0] = 1
        elif self.game_state.round_stage == RoundStage.TEAM_VOTE:
            action_type_one_hot[1] = 1
        elif self.game_state.round_stage == RoundStage.QUEST_VOTE:
            action_type_one_hot[2] = 1

        # Leader
        leader = np.zeros(5)
        leader[self.game_state.leader_index] = 1
        
        # Team
        team = np.zeros(5)
        if self.game_state.round_stage == RoundStage.TEAM_VOTE or self.game_state.round_stage == RoundStage.QUEST_VOTE:
            team_inds = self.game_state.teams[-1]
            team[team_inds] = 1

        # Quest progress
        r_wins_progress = np.array([np.count_nonzero(self.game_state.quest_results == QuestResult.SUCCEEDED) / 3])
        s_wins_progress = np.array([np.count_nonzero(self.game_state.quest_results == QuestResult.FAILED) / 3])
        
        # Quest number
        quest_num = np.zeros(5)
        quest_num[self.game_state.quest_num] = 1

        # Turns until hammer
        turns_until_hammer = np.zeros(5)
        for i in range(self.game_state.turns_until_hammer[0]):
            turns_until_hammer[(self.game_state.leader_index + i) % 5] = i + 1
        turns_until_hammer = softmax(torch.Tensor(turns_until_hammer), dim = 0).numpy()

        # Concatenate all observations
        obs = np.concatenate([
            role_one_hot,
            agent_index_one_hot,
            beliefs,
            action_type_one_hot,
            leader,
            team,
            r_wins_progress,
            s_wins_progress,
            quest_num,
            turns_until_hammer
        ])
        
        return obs
    
    ###############
    # State Utils #
    ###############
    
    def _is_game_over(self) -> bool:
        """
        Returns true if the game is over
        """
        return self.game_state.game_stage in [GameStage.SPY_WIN, GameStage.RESISTANCE_WIN]
    
    def _is_agent_action_required(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        # If the game is over, no action is required
        if self._is_game_over():
            return False
        
        # Check the game stage to determine if the agent's action is required
        if self.game_state.game_stage == GameStage.IN_PROGRESS:
            if self.game_state.round_stage == RoundStage.TEAM_PROPOSAL:
                return self._team_proposal_requires_agent_action()
            elif self.game_state.round_stage == RoundStage.TEAM_VOTE:
                return self._team_vote_requires_agent_action()
            elif self.game_state.round_stage == RoundStage.QUEST_VOTE:
                return self._quest_vote_requires_agent_action()
        if self.game_state.game_stage == GameStage.MERLIN_VOTE:
            return self._merlin_vote_requires_agent_action()
        raise ValueError(f"Unknown game stage: {self.game_state.game_stage}")
    
    def _team_proposal_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        return self.game_state.leader_index == self.agent_index
    
    def _merlin_vote_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        # We never need to query the model, we just read from the beliefs.
        return False
    
    def _quest_vote_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        quest_team = self.game_state.quest_teams[self.game_state.quest_num]
        return self.agent_index in quest_team
    
    def _team_vote_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        return self.agent_index in self.game_state.team_votes
    
    ############################
    # Stepping the Environment #
    ############################
    
    def _step_until_agent_action_required(self):
        """
        Step until the agent's action is required
        """
        
        # While we don't need the agent's action
        while not self._is_agent_action_required():
            
            # If the game is over, break
            if self._is_game_over():
                break
            
            # Otherwise, step the game (without agent action)
            if self.game_state.game_stage == GameStage.IN_PROGRESS:
                if self.game_state.round_stage == RoundStage.TEAM_PROPOSAL:
                    self._team_proposal_step(None)
                elif self.game_state.round_stage == RoundStage.TEAM_VOTE:
                    self._team_vote_step(None)
                elif self.game_state.round_stage == RoundStage.QUEST_VOTE:
                    self._quest_vote_step(None)
                else:
                    raise ValueError(f"Unknown round stage: {self.game_state.round_stage}")
            elif self.game_state.game_stage == GameStage.MERLIN_VOTE:
                self._merlin_vote_step()
            else:
                raise ValueError(f"Unknown game stage: {self.game_state.game_stage}")
    
    def _team_proposal_step(self, action: Optional[List[int]]):
        """
        Run the team proposal step
        """
        
        if self._team_proposal_requires_agent_action():
            assert action is not None
            team = action
            self.game_state.propose_team(team)
        else:
            assert action is None
            leader = self.players[self.game_state.leader_index]
            team = leader.get_team_proposal(self.game_state)
            self.game_state.propose_team(team)        
    
    def _team_vote_step(self, action: Optional[TeamVote]) -> None:
        """
        Run the team vote step
        """
        
        if self._team_vote_requires_agent_action():
            assert action is not None
        else:
            assert action is None
        
        team_votes = []
        for i in range(self.num_players):
            if i != self.agent_index:
                team_votes.append(self.players[i].get_team_vote(self.game_state))
            else:
                team_votes.append(action)
        self.game_state.vote_on_team(team_votes)
            
    def _quest_vote_step(self, action: Optional[QuestVote]) -> None:
        """
        Run the quest vote step
        """
        
        if self._quest_vote_requires_agent_action():
            assert action is not None
        else:
            assert action is None
        
        quest_team = self.game_state.quest_teams[self.game_state.quest_num]
        quest_votes = []
        if self.agent_index not in quest_team:
            quest_votes = [
                self.players[i].get_quest_vote(self.game_state) for i in quest_team
            ]
        else:
            quest_votes = [
                self.players[i].get_quest_vote(self.game_state) if i != self.agent_index else action for i in quest_team
            ]
        self.game_state.vote_on_quest(quest_votes)
    
    def _merlin_vote_step(self) -> None:
        """
        Run the merlin vote step
        """
        
        # We should never require the agent's action for the Merlin vote
        assert not self._merlin_vote_requires_agent_action()
        
        merlin_guesses = []
        
        # Non agent players
        for i, player in self.players.items():
            if player.role == Role.SPY:
                    merlin_guesses.append(player.guess_merlin(self.game_state))
        
        # Agent player
        if self.agent_role == Role.SPY:
            merlin_guesses.append(self._get_agent_merlin_guess())
        
        # All spies guess merlin. Take the majority vote, if tie, randomly choose between tied players
        player_idxs, counts = np.unique(merlin_guesses, return_counts=True)
        max_votes_mask = np.where(counts == counts.max())[0]
        merlin_guess = np.random.choice(player_idxs[max_votes_mask])
        
        self.game_state.guess_merlin(merlin_guess)
    
    def _get_agent_merlin_guess(self) -> int:
        """
        Get the agent's guess for Merlin, using the belief model
        """
        belief = self.belief_model(self.game_state, self.agent_index)
        top_belief_ind = np.argmax(belief.distribution)
            
        role_assignment = belief.all_assignments[top_belief_ind]
        merlin_index = role_assignment.index(Role.MERLIN)
        return merlin_index
        
    ###########
    # Cleanup #
    ###########    
                
    def close(self):
        """
        Perform any necessary cleanup
        """
        pass

# You can create an instance of the environment to test it
if __name__ == "__main__":
    env = AvalonEnv()
    print("Initial State:", env.reset())
    def player_factory(role: Role, index: int) -> RandomAvalonPlayer:
        return RandomAvalonPlayer(role, index)

    roles = [
        Role.MERLIN,
        Role.RESISTANCE,
        Role.RESISTANCE,
        Role.SPY,
        Role.SPY,
    ]

    simulator = AvalonSimulator(roles, player_factory, verbose=True)
    final_game_state = simulator.run_to_