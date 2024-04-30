from typing import List, Callable
import numpy as np

from src.game_state import AvalonGameState
from src.players.player import AvalonPlayer
from src.utils import (
    Role, TeamVote, TeamVoteResult, QuestVote,
    QuestResult, RoundStage, GameStage,
)
class AvalonSimulator:
    """
    Simulates an avalon game with a set of roles.
    Creates a game state, and set of players players (by calling the player_factory function), 
    and runs the game to completion.
    """
    def __init__(
        self,
        roles: List[Role],
        player_factory: Callable[[Role], AvalonPlayer],
        randomize_player_assignments: bool = True, 
        verbose: bool = False,
    ) -> None:
        """
        roles: List of roles for the players in the game
        player_factory: Function that takes a Role and returns an AvalonPlayer
        """
        self.game_state = AvalonGameState(
            roles,
            randomize_player_assignments=randomize_player_assignments, 
            verbose=verbose
        )
        player_assignment = self.game_state.player_assignments
        self.players: List[AvalonPlayer] = [player_factory(role, i) for i, role in enumerate(player_assignment)]
        
    def run_to_completion(self) -> AvalonGameState:
        """
        Run the game to completion and return the final game state.
        """
        while self.game_state.game_stage not in [GameStage.RESISTANCE_WIN, GameStage.SPY_WIN]:
            self.step()
        return self.game_state
    
    def step(self) -> None:
        """
        Requests actions from each player (given the current game state),
        and updates the game state accordingly.
        """
        
        if self.game_state.game_stage == GameStage.IN_PROGRESS:
            if self.game_state.round_stage == RoundStage.TEAM_PROPOSAL:
                self._team_proposal_step()
            elif self.game_state.round_stage == RoundStage.TEAM_VOTE:
                self._team_vote_step()
            elif self.game_state.round_stage == RoundStage.QUEST_VOTE:
                self._quest_vote_step()
        elif self.game_state.game_stage == GameStage.MERLIN_VOTE:
            self._merlin_vote_step()
            
    def _team_proposal_step(self) -> None:
        leader = self.players[self.game_state.leader_index]
        team = leader.get_team_proposal(self.game_state)
        self.game_state.propose_team(team)
    
    def _team_vote_step(self) -> None:
        team_votes = [player.get_team_vote(self.game_state) for player in self.players]
        self.game_state.vote_on_team(team_votes)
        
    def _quest_vote_step(self) -> None:
        quest_team = self.game_state.quest_teams[self.game_state.quest_num]
        quest_votes = [
            self.players[i].get_quest_vote(self.game_state) for i in quest_team
        ]
        self.game_state.vote_on_quest(quest_votes)
    
    def _merlin_vote_step(self) -> None:
        merlin_guesses = [
            player.guess_merlin(self.game_state) for player in self.players if player.role == Role.SPY
        ]
        
        # All spies guess merlin. Take the majority vote, if tie, randomly choose between tied players
        player_idxs, counts = np.unique(merlin_guesses, return_counts=True)
        max_votes_mask = np.where(counts == counts.max())[0]
        merlin_guess = np.random.choice(player_idxs[max_votes_mask])
        
        self.game_state.guess_merlin(merlin_guess)
                
        