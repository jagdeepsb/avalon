from typing import List

from src.utils import (
    Role, TeamVote, QuestVote,
)
from src.game_state import AvalonGameState

class AvalonPlayer:
    """
    Player interface for the game of Avalon.
    Player strategies are implemented by subclassing this class and implementing the methods.
    """
    def __init__(self, role: Role, index: int) -> None:
        """
        role: The role of the player
        index: The index of the player in the game (into the player_assignments list)
        """
        self.role = role
        self.index = index
        
    def get_team_proposal(self, game_state: AvalonGameState) -> List[int]:
        """
        Get the player's team proposal for the current round.
        Returns a list of player indices.
        """
        raise NotImplementedError
    
    def get_team_vote(self, game_state: AvalonGameState) -> TeamVote:
        """
        Get the player's vote for whether the proposed team should go on the quest.
        """
        team = game_state.teams[game_state.round_num]
        raise NotImplementedError
    
    def get_quest_vote(self, game_state: AvalonGameState) -> QuestVote:
        """
        Get the player's vote for the current quest.
        """
        team = game_state.quest_teams[game_state.quest_num]
        board_state = game_state.quest_results
        raise NotImplementedError
    
    def guess_merlin(self, game_state: AvalonGameState) -> int:
        """
        Guess the index of the player who's role is Merlin.
        """
        raise NotImplementedError
    
    