from typing import List
import random

from src.game.utils import (
    TeamVote, QuestVote, Role
)
from src.game.game_state import AvalonGameState
from src.players.player import AvalonPlayer

def get_mimic_player_factory(game_state: AvalonGameState) -> AvalonPlayer:
    def mimic_player_factory(role: Role, index: int) -> AvalonPlayer:
        return MimicAvalonPlayer(game_state, role, index)
    return mimic_player_factory
    
class MimicAvalonPlayer(AvalonPlayer):
    """
    Player that copies the behavior of a player in an already played game.
    """
    
    def __init__(self, game_state: AvalonGameState, role: Role, index: int) -> None:
        """
        index: The index of the player in the game (into the player_assignments list)
        """
        super().__init__(role, index)
        self.copy_state = game_state
                
        assert role == game_state.player_assignments[index], (
            f"Player role {role} does not match the role in the game state {game_state.player_assignments[index]}. Make sure you pass in `randomize_player_assignments = False` into your simulator/game state."
        )
    
    def get_team_proposal(self, game_state: AvalonGameState) -> List[int]:
        return self.copy_state.teams[game_state.round_num]
    
    def get_team_vote(self, game_state: AvalonGameState) -> TeamVote:
        return self.copy_state.team_votes[game_state.round_num][self.index]
    
    def get_quest_vote(self, game_state: AvalonGameState) -> QuestVote:
        index_into_quest_team = game_state.quest_teams[game_state.quest_num].index(self.index)
        quest_vote = self.copy_state.quest_votes[game_state.quest_num][index_into_quest_team]
        if self.role != Role.SPY:
            assert quest_vote == QuestVote.SUCCESS, (
                f"Resistance player {self.index} voted {quest_vote} on quest {game_state.quest_num}."
            )
        return quest_vote
    
    def guess_merlin(self, game_state: AvalonGameState) -> int:
        return self.copy_state.merlin_guess_player_idx