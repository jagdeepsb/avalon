from typing import List
import random

from src.game.utils import (
    TeamVote, QuestVote,
)
from src.game.game_state import AvalonGameState
from src.players.player import AvalonPlayer
from src.game.utils import (
    Role, QuestResult, RoundStage, GameStage
)
    
class RandomAvalonPlayer(AvalonPlayer):
    """
    Player that makes random decisions.
    """
    def get_action(self, game_state: AvalonGameState):
        if game_state.game_stage == GameStage.MERLIN_VOTE:
            return random.choice(range(game_state.n_players))
        elif game_state.round_stage == RoundStage.TEAM_PROPOSAL:
            return random.sample(range(game_state.n_players), game_state.team_size)
        elif game_state.round_stage == RoundStage.TEAM_VOTE:
            return random.choice([TeamVote.APPROVE, TeamVote.REJECT])
        elif game_state.round_stage == RoundStage.QUEST_VOTE:
            return random.choice([QuestVote.SUCCESS, QuestVote.FAIL])
    
    def get_team_proposal(self, game_state: AvalonGameState) -> List[int]:
        return random.sample(range(game_state.n_players), game_state.team_size)
    
    def get_team_vote(self, game_state: AvalonGameState) -> TeamVote:
        return random.choice([TeamVote.APPROVE, TeamVote.REJECT])
    
    def get_quest_vote(self, game_state: AvalonGameState) -> QuestVote:
        return random.choice([QuestVote.SUCCESS, QuestVote.FAIL])
    
    def guess_merlin(self, game_state: AvalonGameState) -> int:
        return random.choice(range(game_state.n_players))