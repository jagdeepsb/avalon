from typing import List
import random

from src.utils import (
    TeamVote, QuestVote,
)
from src.game_state import AvalonGameState
from src.players.player import AvalonPlayer
    
class RandomAvalonPlayer(AvalonPlayer):
    """
    Player that makes random decisions.
    """
    
    def get_team_proposal(self, game_state: AvalonGameState) -> List[int]:
        return random.sample(range(game_state.n_players), game_state.team_size)
    
    def get_team_vote(self, game_state: AvalonGameState) -> TeamVote:
        return random.choice([TeamVote.APPROVE, TeamVote.REJECT])
    
    def get_quest_vote(self, game_state: AvalonGameState) -> QuestVote:
        return random.choice([QuestVote.SUCCESS, QuestVote.FAIL])
    
    def guess_merlin(self, game_state: AvalonGameState) -> int:
        return random.choice(range(game_state.n_players))