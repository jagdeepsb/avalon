from typing import List, Tuple
import random
from src.utils import (
    Role, TeamVote, TeamVoteResult, QuestVote,
    QuestResult, RoundStage, GameStage,
    TEAM_SIZE_BY_N_PLAYERS_AND_QUEST_NUM
)

class AvalonGameState:
    """
    Stores the state of an Avalon game. State can be updated by calling the appropriate methods:
    - propose_team
    - vote_on_team
    - vote_on_quest
    - guess_merlin
    State can be reset by calling the reset method.
    State will throw an error if an invalid action is taken given the current state.
    """
    def __init__(
        self,
        roles: List[Role],
        randomize_player_assignments: bool = True,
        verbose: bool = False,
    ) -> None:
        
        self._game_roles = roles
        self.n_players = len(roles)
        self.verbose = verbose
        
        self.reset(randomize_player_assignments)
        
    def reset(self, randomize_player_assignments: bool = True) -> None:
        """
        Reset the game state. Randomize player assignments if specified.
        """
        
        # Player assignments
        roles = self._game_roles.copy()
        if randomize_player_assignments:
            random.shuffle(roles)
        self.player_assignments = roles
        
        self.round_num = 0
        self.quest_num = 0
        self.leader_index = 0
        
        # Data for each round
        self.turns_until_hammer: List[int] = [4] # For each round, number of turns until hammer (0=hammer)
        
        # Data for teams proposed in each round
        self.teams: List[List[int]] = [] # For each round, list of player indices in team
        self.team_votes: List[List[TeamVote]] = [] # For each round, list of each player's vote
        self.team_vote_results: List[TeamVoteResult] = [] # For each round, result of team vote
        
        # Data for each quest
        self.quest_teams: List[List[int]] = [] # For each quest, list of player indices in team
        self.quest_votes: List[List[QuestVote]] = [] # For each quest, list of each player's vote (for players on the team)
        self.quest_results: List[QuestResult] = [] # For each quest, result of quest
        
        self.game_stage = GameStage.IN_PROGRESS
        self.round_stage = RoundStage.TEAM_PROPOSAL
        
        self.team_size = TEAM_SIZE_BY_N_PLAYERS_AND_QUEST_NUM[self.n_players][self.quest_num+1]
        
        # Who was guessed to be Merlin
        self.merlin_guess_player_idx = None
        
        if self.verbose:
            print(f"[RESET]: Game reset to player roles: {self.player_assignments}\n")
        
    @property
    def get_game_state_as_str(self) -> str:
        if self.game_stage == GameStage.IN_PROGRESS:
            if self.round_stage == RoundStage.TEAM_PROPOSAL:
                return f"Waiting for {self.player_assignments[self.leader_index]} to propose a team"
            elif self.round_stage == RoundStage.TEAM_VOTE:
                return f"Waiting for team votes for team {self.teams[self.round_num]}"
            elif self.round_stage == RoundStage.QUEST_VOTE:
                return f"Waiting for quest votes for quest {self.quest_num} with team {self.quest_teams[self.quest_num]}"
        elif self.game_stage == GameStage.MERLIN_VOTE:
            return "Waiting for spies to vote for Merlin"
        elif self.game_stage == GameStage.RESISTANCE_WIN:
            return "Resistance wins!"
        elif self.game_stage == GameStage.SPY_WIN:
            return "Spies win!"
        
    def propose_team(self, team: List[int]) -> None:
        """
        team: List of player indices in the proposed team
        - Game must be in the IN_PROGRESS stage
        - Round must be in the TEAM_PROPOSAL stage
        - The team must be valid
        """
        
        # Check if game is in the correct stage and the team is valid
        if self.game_stage != GameStage.IN_PROGRESS:
            raise ValueError(f"Cannot propose team when in stage {self.game_stage}")
        if self.round_stage != RoundStage.TEAM_PROPOSAL:
            raise ValueError(f"Cannot propose team when in stage {self.round_stage}")
        if len(team) != self.team_size:
            raise ValueError(f"Team size must be {self.team_size}")
        if len(set(team)) != len(team):
            raise ValueError("Team must have unique players")
        for player_idx in team:
            if player_idx < 0 or player_idx >= self.n_players:
                raise ValueError(f"Player index {player_idx} out of bounds (0-{self.n_players-1})")
        self._check_start_of_round_invariant()
            
        is_hammer = self.turns_until_hammer[self.round_num] == 0
        
        if self.verbose:
            print(f"[PROPOSED TEAM]: {self.leader_index} proposed {team}. Hammer: {is_hammer}")
        
        # Update state
        self.teams.append(team)
        self.round_stage = RoundStage.TEAM_VOTE
        if is_hammer:
            self.vote_on_team([TeamVote.APPROVE] * self.n_players) # Auto-approve if hammer
        
    def vote_on_team(self, votes: List[TeamVote]) -> None:
        """
        votes: List of each player's vote
        - Game must be in the IN_PROGRESS stage
        - Round must be in the TEAM_VOTE stage
        - The votes must be valid
        """
        
        # Check if game is in the correct stage and the votes are valid
        if self.game_stage != GameStage.IN_PROGRESS:
            raise ValueError(f"Cannot vote on team when in stage {self.game_stage}")
        if self.round_stage != RoundStage.TEAM_VOTE:
            raise ValueError(f"Cannot vote on team when in stage {self.round_stage}")
        if len(votes) != self.n_players:
            raise ValueError(f"Number of votes must be {self.n_players}")
        
        did_pass = votes.count(TeamVote.APPROVE) > votes.count(TeamVote.REJECT) # Strict majority
        
        if self.verbose:
            print(f"[TEAM VOTE]: Votes: {votes}, Result: {'Approved' if did_pass else 'Rejected'}")
        
        # Update state
        self.team_votes.append(votes)
        self.team_vote_results.append(TeamVoteResult.APPROVED if did_pass else TeamVoteResult.REJECTED)
        if did_pass:
            self.round_stage = RoundStage.QUEST_VOTE
            self.quest_teams.append(self.teams[self.round_num])
        else:
            self.leader_index = (self.leader_index + 1) % self.n_players
            self.round_num += 1
            self.round_stage = RoundStage.TEAM_PROPOSAL
            self.turns_until_hammer.append(self.turns_until_hammer[self.round_num-1]-1)
            assert self.turns_until_hammer[-1] >= 0, (
                "Something went horribly wrong"
            )
            
    def vote_on_quest(self, votes: List[QuestVote]) -> None:
        """
        votes: List of each player's vote (for players on the team)
        - Game must be in the IN_PROGRESS stage
        - Round must be in the QUEST_VOTE stage
        - The votes must be valid
        """
        
        # Check if game is in the correct stage and the votes are valid
        if self.game_stage != GameStage.IN_PROGRESS:
            raise ValueError(f"Cannot vote on quest when in stage {self.game_stage}")
        if self.round_stage != RoundStage.QUEST_VOTE:
            raise ValueError(f"Cannot vote on quest when in stage {self.round_stage}")
        if len(votes) != self.team_size:
            raise ValueError(f"Number of votes must be {self.team_size}, got {len(votes)}")
        
        did_pass = votes.count(QuestVote.FAIL) == 0 # Fail if any fail votes
        
        if self.verbose:
            print(f"[QUEST VOTE]: Votes: {votes}, Result: {'Succeeded' if did_pass else 'Failed'}\n")
        
        # Update state
        self.quest_votes.append(votes)
        self.quest_results.append(QuestResult.SUCCEEDED if did_pass else QuestResult.FAILED)

        # Check if game is over
        resistance_wins = self.quest_results.count(QuestResult.SUCCEEDED) == 3
        spy_wins = self.quest_results.count(QuestResult.FAILED) == 3
        if resistance_wins:
            self.game_stage = GameStage.MERLIN_VOTE
        elif spy_wins:
            self.game_stage = GameStage.SPY_WIN
        else:
            self.quest_num += 1
            self.round_num += 1
            self.leader_index = (self.leader_index + 1) % self.n_players
            self.round_stage = RoundStage.TEAM_PROPOSAL
            self.turns_until_hammer.append(4)
            self.team_size = TEAM_SIZE_BY_N_PLAYERS_AND_QUEST_NUM[self.n_players][self.quest_num+1]
            
    def guess_merlin(self, guess_player_idx: int) -> None:
        """
        guess_player_idx: Index of the player guessed to be Merlin
        - Game must be in the MERLIN_VOTE stage
        """
        
        # Check if game is in the correct stage and the votes are valid
        if self.game_stage != GameStage.MERLIN_VOTE:
            raise ValueError(f"Cannot vote on Merlin when in stage {self.game_stage}")
        if guess_player_idx < 0 or guess_player_idx >= self.n_players:
            raise ValueError(f"Player index {guess_player_idx} out of bounds (0-{self.n_players-1})")
    
        if self.verbose:
            print(f"[MERLIN GUESS]: Spies guessed {guess_player_idx} to be merlin, who is {self.player_assignments[guess_player_idx]}")
        
        # Update state
        self.merlin_guess_player_idx = guess_player_idx
        if self.player_assignments[guess_player_idx] == Role.MERLIN:
            self.game_stage = GameStage.SPY_WIN
        else:
            self.game_stage = GameStage.RESISTANCE_WIN
            
    def _check_start_of_round_invariant(self,):
        """
        Raises if the start of the round invariant is violated.
        """
        
        assert self.round_num == len(self.teams) == len(self.team_votes) == len(self.team_vote_results), (
            f"There is a mismatch {self.round_num} {len(self.teams)} {len(self.team_votes)} {len(self.team_vote_results)}"
        )
        
        assert self.round_num == len(self.turns_until_hammer)-1, (
            f"There is a mismatch {self.round_num} {len(self.turns_until_hammer)-1}"
        )
        
        assert self.quest_num == len(self.quest_teams) == len(self.quest_votes) == len(self.quest_results), (
            f"There is a mismatch {self.quest_num} {len(self.quest_teams)} {len(self.quest_votes)} {len(self.quest_results)}"
        )
        