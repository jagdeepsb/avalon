from __future__ import annotations
from typing import List
import random

from src.game.utils import (
    TeamVote, QuestVote,
)
from src.game.game_state import AvalonGameState
from src.players.player import AvalonPlayer
from src.game.utils import Role

def stupid_hardcoded_player_factory(role: Role, index: int) -> AvalonPlayer:
    """
    Create a player with a hardcoded strategy based on their role.
    """
    if role == Role.MERLIN:
        return StupidHardcodedMerlin(index)
    elif role == Role.SPY:
        return StupidHardcodedSpy(index)
    elif role == Role.RESISTANCE:
        return StupidHardcodedResistance(index)
    else:
        raise ValueError(f"Invalid role: {role}")
    
class StupidHardcodedMerlin(AvalonPlayer):
    """
    Merlin that uses their full information to always benefit the resistance.
    """
    def __init__(self, index: int) -> None:
        """
        index: The index of the player in the game (into the player_assignments list)
        """
        super().__init__(Role.MERLIN, index)
        
    def get_team_proposal(self, game_state: AvalonGameState) -> List[int]:
        """
        Use the game state to propose a team without any spies.
        """
        player_assignments = game_state.player_assignments
        merlin_index = player_assignments.index(Role.MERLIN)
        resistance_indices = [
            i for i, role in enumerate(player_assignments) if role == Role.RESISTANCE
        ]
        team = [merlin_index] + random.sample(resistance_indices, game_state.team_size - 1)
        return team
    
    def get_team_vote(self, game_state: AvalonGameState) -> TeamVote:
        """
        Approve the team if it contains no spies.
        """    
        player_assignments = game_state.player_assignments
        team = game_state.teams[game_state.round_num]
        if any(player_assignments[i] == Role.SPY for i in team):
            return TeamVote.REJECT
        return TeamVote.APPROVE
    
    def get_quest_vote(self, game_state: AvalonGameState) -> QuestVote:
        """
        Always approve the quest.
        """
        return QuestVote.SUCCESS
    
class StupidHardcodedSpy(AvalonPlayer):
    """
    Spy that uses their full information to always benefit the spies.
    """
    def __init__(self, index: int) -> None:
        """
        index: The index of the player in the game (into the player_assignments list)
        """
        super().__init__(Role.SPY, index)
        
    def get_team_proposal(self, game_state: AvalonGameState) -> List[int]:
        """
        Use the game state to propose a team with yourself and all non-spies.
        """
        player_assignments = game_state.player_assignments
        non_spy_indices = [
            i for i, role in enumerate(player_assignments) if role != Role.SPY
        ]
        team = [self.index] + random.sample(non_spy_indices, game_state.team_size - 1)
        return team
    
    def get_team_vote(self, game_state: AvalonGameState) -> TeamVote:
        """
        Approve the team if it contains at least one spy
        """    
        player_assignments = game_state.player_assignments
        team = game_state.teams[game_state.round_num]
        if any(player_assignments[i] == Role.SPY for i in team):
            return TeamVote.APPROVE
        return TeamVote.REJECT
    
    def get_quest_vote(self, game_state: AvalonGameState) -> QuestVote:
        """
        Always reject the quest.
        """
        return QuestVote.FAIL
    
    def guess_merlin(self, game_state: AvalonGameState) -> int:
        """
        Guess a random non-spy player as Merlin.
        """
        
        player_assignments = game_state.player_assignments
        non_spy_indices = [
            i for i, role in enumerate(player_assignments) if role != Role.SPY
        ]
        return random.choice(non_spy_indices)
    
class StupidHardcodedResistance(AvalonPlayer):
    """
    Resistance that proposes random teams and votes for them if they're on it.
    Always approves the quest.
    """
    def __init__(self, index: int) -> None:
        """
        index: The index of the player in the game (into the player_assignments list)
        """
        super().__init__(Role.RESISTANCE, index)
        
    def get_team_proposal(self, game_state: AvalonGameState) -> List[int]:
        """
        Propose a team with yourself and other random players.
        """
        other_indices = [i for i in range(game_state.n_players) if i != self.index]
        team = [self.index] + random.sample(other_indices, game_state.team_size - 1)
        return team
    
    def get_team_vote(self, game_state: AvalonGameState) -> TeamVote:
        """
        Don't approve a team unless your on it.
        """
        team = game_state.teams[game_state.round_num]
        if self.index in team:
            return TeamVote.APPROVE
        return TeamVote.REJECT
    
    def get_quest_vote(self, game_state: AvalonGameState) -> QuestVote:
        """
        Always approve the quest.
        """
        return QuestVote.SUCCESS