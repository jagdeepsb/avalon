import gym
from gym import spaces
import numpy as np
from src.game.simulator import AvalonSimulator
from src.game.game_state import AvalonGameState
from src.game.utils import (
    Role, TeamVote, TeamVoteResult, QuestVote,
    QuestResult, RoundStage, GameStage,
)
from src.players.player import AvalonPlayer
from typing import List, Callable, Tuple, Dict
from src.game.utils import Role
from src.game.simulator import AvalonSimulator
from src.players.random_player import RandomAvalonPlayer
import random
from copy import deepcopy

class AvalonEnv(gym.Env):
    """
    Custom Environment for the game Avalon that follows gym interface.
    """

    def __init__(
        self,
        roles: List[Role],
        bot_player_factory: Callable[[Role, int], AvalonPlayer],
        randomize_player_assignments=True,
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
        self.bot_player_factory = bot_player_factory    
        self.randomize_player_assignments = randomize_player_assignments
        self.verbose = verbose

        self.reset()

    def reset(self) -> Tuple[AvalonGameState, Role, int]:
        """
        Reset the environment:
        - Reset the game state
        - Randomly shuffle player assignments (if randomize_player_assignments=True)
        - Assign the agent to a random role and player index
        
        Returns
        - game_state: The initial game state
        - agent_role: The role of the agent
        - agent_index: The index of the agent
        """
        
        # Compute player assignments
        player_assignments = self.roles.copy()
        if self.randomize_player_assignments:
            random.shuffle(player_assignments)
            
        # Assign agent role and index
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
                
        return deepcopy(self.game_state), self.agent_role, self.agent_index
    
    def _step_until_agent_action_required(self):
        """
        Step until the agent's action is required
        """
        
        
    def step(self, action: List[int]):
        """
        The action method where the game dynamics are implemented
        """
        
        require_agent_action = False
        if self.game_state.game_stage == GameStage.IN_PROGRESS:
            if self.game_state.round_stage == RoundStage.TEAM_PROPOSAL:
                require_agent_action = self._team_proposal_step(action)
            elif self.game_state.round_stage == RoundStage.TEAM_VOTE:
                require_agent_action = self._team_vote_step(action)
            elif self.game_state.round_stage == RoundStage.QUEST_VOTE:
                require_agent_action = self._quest_vote_step(action)
        elif self.game_state.game_stage == GameStage.MERLIN_VOTE:
            require_agent_action = self._merlin_vote_step(action)

        # Calculate reward, done, and any additional info
        reward = 0
        done = False
        info = {"did_use_agent_action": require_agent_action}
        if self.game_state.game_stage == GameStage.SPY_WIN:
            done = True
            if self.player_assignments[self.agent_index] == Role.SPY:
                reward = 1.0
        elif self.game_state.game_stage == GameStage.RESISTANCE_WIN:
            done = True
            if self.player_assignments[self.agent_index] == Role.RESISTANCE or self.player_assignments[self.agent_index] == Role.MERLIN:
                reward = 1.0
        return deepcopy(self.game_state), reward, done, info
    
    def _team_proposal_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        return self.game_state.leader_index == self.agent_index

    def _team_proposal_step(self, action: List[int]) -> bool:
        """
        returns true if used the agent's action
        """
        if self.game_state.leader_index != self.agent_index:
            leader = self.players[self.game_state.leader_index]
            team = leader.get_team_proposal(self.game_state)
            self.game_state.propose_team(team)
            return False
        else:
            team = action
            self.game_state.propose_team(team)
            return True
        
    def _team_vote_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        return self.agent_index in self.game_state.team_votes
        
    def _team_vote_step(self, action: List[int]) -> bool:
        """
        returns true if used the agent's action
        """
        team_votes = []
        for i in range(self.num_players):
            if i != self.agent_index:
                team_votes.append(self.players[i].get_team_vote(self.game_state))
            else:
                team_votes.append(action)
        self.game_state.vote_on_team(team_votes)
        return self.agent_index in self.game_state.team_votes
    
    def _quest_vote_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        quest_team = self.game_state.quest_teams[self.game_state.quest_num]
        return self.agent_index in quest_team
        
    def _quest_vote_step(self, action: List[int]) -> bool:
        """
        returns true if used the agent's action
        """
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
        return self.agent_index in quest_team
    
    def _merlin_vote_step(self, action) -> bool:
        """
        returns true if used the agent's action
        """
        merlin_guesses = []
        for i, player in self.players.items():
            if player.role == Role.SPY:
                if i != self.agent_index:
                    merlin_guesses.append(player.guess_merlin(self.game_state))
                else:
                    merlin_guesses.append(action)
        
        # All spies guess merlin. Take the majority vote, if tie, randomly choose between tied players
        player_idxs, counts = np.unique(merlin_guesses, return_counts=True)
        max_votes_mask = np.where(counts == counts.max())[0]
        merlin_guess = np.random.choice(player_idxs[max_votes_mask])
        
        self.game_state.guess_merlin(merlin_guess)
        return self.agent_role == Role.SPY
                
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