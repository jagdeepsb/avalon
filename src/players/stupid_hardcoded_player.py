from __future__ import annotations
from typing import List
import random
import numpy as np

from src.game.utils import (
    TeamVote, QuestVote, QuestResult,
    TeamVoteResult
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
        if self.index in non_spy_indices:
            non_spy_indices.remove(self.index)
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
        
        self.player_is_bad_score = []
        self.proc_n_quests = 0
        
    def _update_player_is_bad_score(self, game_state: AvalonGameState):
        """
        Update the player_is_bad_score list with the current game state.
        """
        if len(self.player_is_bad_score) == 0:
            self.player_is_bad_score = [0] * game_state.n_players
        
        while self.proc_n_quests < game_state.quest_num:
            # Quest hasnt finished yet
            if len(game_state.quest_results) <= self.proc_n_quests:
                print(f"Quest {len(game_state.quest_results)}, {self.proc_n_quests}")
                break
            
            # If the last quest failed, increment the score of all players on the team
            if game_state.quest_results[self.proc_n_quests] == QuestResult.FAILED:
                n_fails = game_state.quest_votes[self.proc_n_quests].count(QuestVote.FAIL)
                quest_team = game_state.quest_teams[self.proc_n_quests]
                for player_index in quest_team:
                    self.player_is_bad_score[player_index] += n_fails
            
            # If the last quest succeeded, increment the score of all players not on the team
            elif game_state.quest_results[self.proc_n_quests] == QuestResult.SUCCEEDED:
                quest_team = game_state.quest_teams[self.proc_n_quests]
                
                if len(quest_team) == 2:
                    penalty = 1.5
                elif len(quest_team) == 3:
                    penalty = 3
                    
                for player_index in range(game_state.n_players):
                    if player_index not in quest_team:
                        self.player_is_bad_score[player_index] += penalty
                        
            round_num = self._get_round_num_for_quest_num(self.proc_n_quests, game_state)
            assert game_state.teams[round_num] == game_state.quest_teams[self.proc_n_quests], (game_state.teams[round_num], game_state.quest_teams[self.proc_n_quests])
            if round_num != -1:
                # If the quest failed, increment all players who voted for and decrement all players who voted against
                if game_state.quest_results[self.proc_n_quests] == QuestResult.FAILED:
                    for player_index, vote in enumerate(game_state.team_votes[round_num]):
                        if vote == TeamVote.APPROVE:
                            self.player_is_bad_score[player_index] += 0.5
                        else:
                            self.player_is_bad_score[player_index] -= 0.5
                
                # If the quest succeeded, increment all players who voted against and decrement all players who voted for
                elif game_state.quest_results[self.proc_n_quests] == QuestResult.SUCCEEDED:
                    for player_index, vote in enumerate(game_state.team_votes[round_num]):
                        if vote == TeamVote.APPROVE:
                            self.player_is_bad_score[player_index] -= 0.5
                        else:
                            self.player_is_bad_score[player_index] += 0.5
                        
            # You are a resistance player, so you are good
            self.player_is_bad_score[self.index] = 0
            
            self.proc_n_quests += 1
            
        # print(self.player_is_bad_score)
        # print(game_state.player_assignments)
            
    def _get_round_num_for_quest_num(self, quest_num: int, game_state: AvalonGameState) -> int:
        quest_count = 0
        for i, team_vote_result in enumerate(game_state.team_vote_results):
            if team_vote_result == TeamVoteResult.APPROVED:
                if quest_count == quest_num:
                    return i
                quest_count += 1
        return -1
        
    def get_team_proposal(self, game_state: AvalonGameState) -> List[int]:
        """
        Propose a team with yourself and other random players.
        Sample players that have a low player_is_bad_score.
        """
        
        self._update_player_is_bad_score(game_state)
        
        other_indices = [i for i in range(game_state.n_players) if i != self.index]
        # team = [self.index] + random.sample(other_indices, game_state.team_size - 1)
        
        bad_scores = np.array([self.player_is_bad_score[i] for i in other_indices])
        weights = np.exp(-bad_scores)
        weights /= weights.sum()
        
        # print(weights)
        # print([game_state.player_assignments[i] for i in other_indices])
        
        team = [self.index] + list(np.random.choice(other_indices, game_state.team_size - 1, p=weights, replace=False))
        
        return team
    
    def get_team_vote(self, game_state: AvalonGameState) -> TeamVote:
        """
        Approve teams with a low player_is_bad_score.
        """
        
        self._update_player_is_bad_score(game_state)
        
        team = game_state.teams[game_state.round_num]
        # if self.index in team:
        #     return TeamVote.APPROVE
        
        average_bad_score = np.mean(self.player_is_bad_score)
        team_bad_score = np.mean([self.player_is_bad_score[i] for i in team])
        
        
        # print(team)
        # print(self.player_is_bad_score)
        # print(game_state.player_assignments)
        # print(average_bad_score, team_bad_score)
        # print("approved" if team_bad_score < average_bad_score else "rejected")
        # print()
        if team_bad_score < average_bad_score:
            return TeamVote.APPROVE
        
        return TeamVote.REJECT
    
    def get_quest_vote(self, game_state: AvalonGameState) -> QuestVote:
        """
        Always approve the quest.
        """
        return QuestVote.SUCCESS