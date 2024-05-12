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
from typing import List, Callable, Tuple, Dict, Optional
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
        
        # Move the environment to the first state where the agent's action is required
        self._step_until_agent_action_required()
        
        # If the game has ended without requiring any agent action, reset the environment again
        if self._is_game_over():
            return self.reset()
                
        return deepcopy(self.game_state), self.agent_role, self.agent_index

        
    def step(self, action: List[int]):
        """
        The action method where the game dynamics are implemented
        """
        
        # We must be in a state where the agent's action is required
        assert self._is_agent_action_required()
        
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
        info = {}
        if self.game_state.game_stage == GameStage.SPY_WIN:
            done = True
            if self.player_assignments[self.agent_index] == Role.SPY:
                reward = 1.0
        elif self.game_state.game_stage == GameStage.RESISTANCE_WIN:
            done = True
            if self.player_assignments[self.agent_index] == Role.RESISTANCE or self.player_assignments[self.agent_index] == Role.MERLIN:
                reward = 1.0
              
        
        return deepcopy(self.game_state), reward, done, info
    
    
    ############################
    # Stepping the Environment #
    ############################
    
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
    
    def _team_proposal_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        return self.game_state.leader_index == self.agent_index

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
            

    def _team_vote_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        return self.agent_index in self.game_state.team_votes
        
    def _team_vote_step(self, action: Optional[List[int]]) -> None:
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
    
    def _quest_vote_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        quest_team = self.game_state.quest_teams[self.game_state.quest_num]
        return self.agent_index in quest_team
        
    def _quest_vote_step(self, action: Optional[List[int]]) -> None:
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
    
    def _merlin_vote_requires_agent_action(self) -> bool:
        """
        Returns true if the agent's action is required
        """
        # We never need to query the model, we just read from the beliefs.
        return False
    
    def _merlin_vote_step(self) -> None:
        """
        Run the merlin vote step
        """
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