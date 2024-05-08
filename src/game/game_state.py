from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from src.game.utils import (
    Role, TeamVote, TeamVoteResult, QuestVote,
    QuestResult, RoundStage, GameStage, ROLES_CAN_SEE,
    TEAM_SIZE_BY_N_PLAYERS_AND_QUEST_NUM
)
from src.game.beliefs import Belief
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
            
    ################
    # Observations #
    ################
    
    def player_role_one_hot_obs(self, player_index: int) -> np.ndarray:
        """
        Returns a one hot encoding of the player's role as a numpy 1D array of shape (Role.one_hot_dim(),).
        """
        return self.player_assignments[player_index].as_one_hot().astype(self.np_type)
    
    def all_player_roles_obs(self, from_perspective_of_player_index: Optional[int] = None) -> np.ndarray:
        """
        Returns the one hot encoding of the player's roles as a numpy 2D array with 
        shape (n_players, Role.one_hot_dim()). 
        
        from_perspective_of_player_index: If provided, the roles of players that are
        hidden to the current player are masked out to 0.
        """
        
        # Observation contains the one hot encoding of each player's role
        player_obs = np.stack([
            self.player_role_one_hot_obs(i) for i in range(self.n_players)
        ], axis=0)
        
        # Mask out roles of players that the current player cannot see
        if from_perspective_of_player_index is not None:
            for i in range(self.n_players):
                if from_perspective_of_player_index == i: # player can see their own role
                    continue
                curr_role = self.player_assignments[from_perspective_of_player_index]
                other_role = self.player_assignments[i]
                if other_role not in ROLES_CAN_SEE[curr_role]:
                    player_obs[i] = np.zeros_like(player_obs[i])
                    
        return player_obs
                
    def game_state_obs(self, from_perspective_of_player_index: Optional[int] = None) -> np.ndarray:
        """
        Returns the entire history of the game as a numpy array with 
        shape (n_players, max_rounds, 7 + Role.one_hot_dim()).
        
        from_perspective_of_player_index: If provided, the information in the game state
        that the current player cannot see is masked out to 0.
        """
        
        game_history = self.game_history_np # (n_players, max_rounds, 7)
        
        player_roles = self.all_player_roles_obs(from_perspective_of_player_index) # (n_players, Role.one_hot_dim())
        player_roles_expanded = np.repeat(player_roles[:, np.newaxis, :], self.max_rounds, axis=1) # (n_players, max_rounds, Role.one_hot_dim())
        
        return np.concatenate([
            game_history,
            player_roles_expanded
        ], axis=2) # (n_players, max_rounds, 7 + Role.one_hot_dim())
        
    def ground_truth_belief_distribution(self,) -> np.ndarray:
        return Belief.trivial_distribution(
            self.player_assignments
        ).distribution
        
    def get_trainable_belief_distribution(self, player_index: int, constrained: bool = False) -> np.ndarray:
        role = self.player_assignments[player_index]
    
        if not constrained:
            return Belief.smoothed_trivial_distribution(self.player_assignments).distribution
    
        if role == Role.SPY:
            # smoothed distribution over all assignments conditioned on knowing who all the spies are
            return Belief.smoothed_trivial_distribution(self.player_assignments).condition_on(
                [Role.SPY if role == Role.SPY else Role.UNKNOWN for role in self.player_assignments]
            ).distribution
        if role == Role.RESISTANCE:
            # smoothed distribution over all assignments conditioned on knowing you are resistance
            return Belief.smoothed_trivial_distribution(self.player_assignments).condition_on(
                [Role.RESISTANCE if i == player_index else Role.UNKNOWN for i in range(len(self.player_assignments))]
            ).distribution
        raise ValueError(f"Player {player_index} is not a spy or resistance, training not supported")
    
    #########################
    # Game History As Numpy #
    #########################
    
    @property
    def np_type(self) -> np.dtype:
        """Get the numpy data type of the observations."""
        return np.float32
    
    @property
    def max_rounds(self) -> int:
        """Get the maximum number of rounds in the game."""
        return 25
    
    @property
    def turns_until_hammer_np(self) -> np.ndarray:
        """
        Get the number of turns until hammer as a numpy array with shape (max_rounds,). 
        Values are in range [0, 1] where
        - 0 means the turn hasnt happened yet
        - 1 means the turn is the hammer turn
        - fractional values count down to the hammer turn
        """
        
        turns = self.turns_until_hammer.copy() + [5] * (self.max_rounds - len(self.turns_until_hammer))
        turns = np.array(turns, dtype=self.np_type)
        turns = (5-turns)/5.0
        return turns
    
    @property
    def who_was_leader_np(self) -> np.ndarray:
        """
        Get the leader in each round as a numpy array with shape (max_rounds, n_players). 
        1 = leader, fractional = counts up to leader.
        """
        
        leader_np = np.zeros((self.n_players, self.max_rounds), dtype=self.np_type)
        for i in range(self.round_num+1):
            for j in range(self.n_players):
                val = 1-(j/self.n_players)
                player_index = (i+j) % self.n_players
                leader_np[player_index, i] = val
        return leader_np
    
    @property
    def teams_np(self) -> np.ndarray:
        """
        Get the teams proposed in each round as a numpy array with shape (n_players, max_rounds). 
        0 = hasnt happened, 0.5 = not on team, 1 = on team.
        """
        
        teams_np = np.zeros((self.n_players, self.max_rounds), dtype=self.np_type)
        for i, team in enumerate(self.teams):
            teams_np[:, i] = 0.5
            teams_np[team, i] = 1
        return teams_np
    
    @property
    def team_votes_np(self) -> np.ndarray:
        """
        Get the votes on the teams proposed in each round as a numpy array with shape
        (n_players, max_rounds). 0 = hasnt happened, 0.5 = reject, 1 = approve.
        """
        
        team_votes_np = np.zeros((self.n_players, self.max_rounds), dtype=self.np_type)
        for i, team_votes in enumerate(self.team_votes):
            for j, vote in enumerate(team_votes):
                team_votes_np[j, i] = 1 if vote == TeamVote.APPROVE else 0.5
        return team_votes_np
    
    @property
    def team_vote_results_np(self) -> np.ndarray:
        """
        Get the results of the team votes in each round as a numpy array with shape (max_rounds,).
        0 = hasnt happened, 0.5 = rejected, 1 = approved.
        """
        
        team_vote_results_np = np.zeros(self.max_rounds, dtype=self.np_type)
        for i, result in enumerate(self.team_vote_results):
            team_vote_results_np[i] = 1 if result == TeamVoteResult.APPROVED else 0.5
        return team_vote_results_np
    
    @property
    def quest_votes_np(self) -> np.ndarray:
        """
        Get the votes on the teams proposed for each quest as a numpy array with shape (max_rounds). 
        value=0.5*(num_fails+1).
        """
        
        quest_votes_np = np.zeros((self.max_rounds,), dtype=self.np_type)
        quest_counter = 0
        for i, result in enumerate(self.team_vote_results):
            if quest_counter == len(self.quest_votes):
                break
            if result == TeamVoteResult.APPROVED:
                num_fails = self.quest_votes[quest_counter].count(QuestVote.FAIL)
                quest_votes_np[i] = 0.5*(num_fails + 1)
                quest_counter += 1
        return quest_votes_np
    
    @property
    def quest_results_np(self) -> np.ndarray:
        """
        Get the results of the quests as a numpy array with shape (max_rounds,). 
        0 = hasnt happened, 0.5 = failed, 1 = succeeded.
        """
        
        quest_results_np = np.zeros(self.max_rounds, dtype=self.np_type)
        quest_counter = 0
        for i, result in enumerate(self.team_vote_results):
            if quest_counter == len(self.quest_results):
                break
            if result == TeamVoteResult.APPROVED:
                quest_results_np[i] = 0.5 if self.quest_results[quest_counter] == QuestResult.FAILED else 1
                quest_counter += 1
        return quest_results_np
    
    @property
    def game_history_np(self) -> np.ndarray:
        """
        Returns the entire history of the game as a numpy array with shape (n_players, max_rounds, 7).
        """
        
        one_dim_obs = np.stack([
            self.turns_until_hammer_np,
            self.team_vote_results_np,
            self.quest_votes_np,
            self.quest_results_np,
        ], axis=1)
        two_dim_obs = np.stack([
            self.who_was_leader_np,
            self.teams_np,
            self.team_votes_np,
        ], axis=2)
        one_dim_copied = np.repeat(one_dim_obs[np.newaxis, :, :], self.n_players, axis=0)
        
        return np.concatenate([
            two_dim_obs,
            one_dim_copied
        ], axis=2)
    
    ###########
    # Helpers #
    ###########
            
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
    
    def assert_equals(self, other: AvalonGameState) -> bool:
        """
        Asserts that this game state is equal to another game state.
        """
        
        assert self. player_assignments == other.player_assignments
        assert self.round_num == other.round_num
        assert self.quest_num == other.quest_num
        assert self.leader_index == other.leader_index
        
        assert self.turns_until_hammer == other.turns_until_hammer
        assert self.teams == other.teams
        assert self.team_votes == other.team_votes
        assert self.team_vote_results == other.team_vote_results
        
        assert self.quest_teams == other.quest_teams
        assert self.quest_votes == other.quest_votes
        assert self.quest_results == other.quest_results
        
        assert self.merlin_guess_player_idx == other.merlin_guess_player_idx
        assert self.game_stage == other.game_stage
        assert self.round_stage == other.round_stage
    
    ###################
    # To Load BC Data #
    ###################
    
    @classmethod
    def from_json(cls, history: dict) -> AvalonGameState:
        """Holy this is stupid af. I'm sorry."""
        
        was_leader = "VHleader"
        was_picked = "VHpicked"
        did_approve = "VHapprove"
        did_reject = "VHreject"
        
        def get_actions(player_name: str, round_num: int) -> str:
            round_count = 0
            for quest_data in history['voteHistory'][player_name]:
                for round_data in quest_data:
                    if round_count == round_num:
                        return round_data
                    round_count += 1
                    
        def get_num_rounds() -> int:
            default_player_name = list(history['voteHistory'].keys())[0]
            round_count = 0
            for quest_data in history['voteHistory'][default_player_name]:
                round_count += len(quest_data)
            return round_count
        
        def get_num_quests() -> int:
            return len(history['missionHistory'])
        
        def get_first_turn_as_leader(player_name: str, max_rounds: int) -> Optional[int]:
            for i in range(max_rounds):
                if was_leader in get_actions(player_name, i):
                    return i
            return None
        
        def assign_indices_to_none_players(player_name_to_idx: Dict[str, Optional[int]]) -> None:
            seen_inidices = set(player_name_to_idx.values())
            missing_indices = [i for i in range(len(player_name_to_idx)) if i not in seen_inidices]
            num_nones = len([idx for idx in player_name_to_idx.values() if idx is None])
            
            assert len(missing_indices) == num_nones
            
            for player_name, idx in player_name_to_idx.items():
                if idx is None:
                    player_name_to_idx[player_name] = missing_indices.pop()
        
        def get_role(player_name: str) -> Role:
            if history["playerRoles"][player_name]["role"] == "Merlin":
                return Role.MERLIN
            elif history["playerRoles"][player_name]["role"] == "Resistance":
                return Role.RESISTANCE
            elif history["playerRoles"][player_name]["role"] == "Spy":
                return Role.SPY
            elif history["playerRoles"][player_name]["role"] == "Assassin":
                return Role.SPY
            else:
                raise ValueError(f"Invalid role {history['playerRoles'][player_name]['role']} for player {player_name}")
            
        def get_game_stage(history: dict) -> GameStage:
            if history['winningTeam'] == "Resistance":
                return GameStage.RESISTANCE_WIN
            elif history['winningTeam'] == "Spy":
                return GameStage.SPY_WIN
            else:
                raise ValueError("Invalid game stage")
            
        def hadAssassination(history: dict) -> bool:
            return "whoAssassinShot" in history
        
        def assassinShotMerlin(history: dict) -> bool:
            return history["whoAssassinShot"] == "Merlin"
        
        def get_merlin_guess(history: dict, player_assignment: List[int]) -> str:
            if not hadAssassination(history):
                return None
            did_shoot_merlin = assassinShotMerlin(history)
            if did_shoot_merlin:
                return player_assignment.index(Role.MERLIN)
            else:
                # We don't know exactly who the assassin guessed (dataset is vague), choose an arbitrary resistance player
                return player_assignment.index(Role.RESISTANCE)
                
            
        # Get number of rounds
        n_rounds = get_num_rounds()
        n_quests = get_num_quests()
        
        # Get player order (name -> index) mapping
        player_name_to_idx: Dict[str, Optional[int]] = {player_name: get_first_turn_as_leader(player_name, n_rounds) for player_name in history['voteHistory'].keys()}
        assign_indices_to_none_players(player_name_to_idx)
        
        # Get player roles
        player_assignment = [Role.UNKNOWN for _ in range(len(player_name_to_idx))]
        for player_name, idx in player_name_to_idx.items():
            player_assignment[idx] = get_role(player_name)
        
        # Create game state
        game_state = AvalonGameState(player_assignment, randomize_player_assignments=False)
        
        # Initialize game state
        game_state.turns_until_hammer = [4] * n_rounds
        
        game_state.teams = [[] for _ in range(n_rounds)]
        game_state.team_votes = [[] for _ in range(n_rounds)]
        game_state.team_vote_results = [TeamVoteResult.REJECTED for _ in range(n_rounds)]
        
        game_state.quest_teams = [[] for _ in range(n_quests)]
        game_state.quest_votes = [[] for _ in range(n_quests)]
        game_state.quest_results = [QuestResult.FAILED for _ in range(n_quests)]
        
        # Fill in game state
        game_state.round_num = n_rounds-1
        game_state.quest_num = n_quests-1
        game_state.leader_index = (n_rounds-1) % len(player_assignment)
        
        game_state.game_stage = get_game_stage(history)
        game_state.round_stage = RoundStage.QUEST_VOTE
        
        game_state.team_size = TEAM_SIZE_BY_N_PLAYERS_AND_QUEST_NUM[len(player_assignment)][game_state.quest_num]
        game_state.merlin_guess_player_idx = get_merlin_guess(history, player_assignment)
        
        game_state.quest_results = [
            QuestResult.FAILED if result == "failed" else QuestResult.SUCCEEDED for result in history['missionHistory']
        ]
        
        for player_name in history["voteHistory"].keys():
            player_index = player_name_to_idx[player_name]
            round_count = 0
            for quest_num, quest_data in enumerate(history['voteHistory'][player_name]):
                for i, round_data in enumerate(quest_data):
                    game_state.turns_until_hammer[round_count] = 4-i
                    if was_picked in round_data:
                        game_state.teams[round_count].append(player_index)
                    if did_approve in round_data:
                        game_state.team_votes[round_count].append(TeamVote.APPROVE)
                    elif did_reject in round_data:
                        game_state.team_votes[round_count].append(TeamVote.REJECT)
                    else: 
                        raise ValueError("Invalid team vote")
                    if i == len(quest_data)-1:
                        game_state.team_vote_results[round_count] = TeamVoteResult.APPROVED
                        game_state.quest_teams[quest_num] = game_state.teams[round_count]
                    else:
                        game_state.team_vote_results[round_count] = TeamVoteResult.REJECTED
                    round_count += 1
                    
        def resolve_who_voted_how(game_state: AvalonGameState, quest_num: int, num_fails: int):
            quest_team = game_state.quest_teams[quest_num]
            player_assignment = game_state.player_assignments
            quest_result = game_state.quest_results[quest_num]
            
            if num_fails == 0:
                assert quest_result == QuestResult.SUCCEEDED
                game_state.quest_votes[quest_num] = [QuestVote.SUCCESS for _ in quest_team]
                return
            
            n_spies = len([player_idx for player_idx in quest_team if player_assignment[player_idx] == Role.SPY])
            assert num_fails <= n_spies, (
                f"Number of fails ({num_fails}) must be less than or equal to number of spies ({n_spies})"
            )
            
            quest_votes = []
            n_more_fails_allowed = num_fails
            for player_idx in quest_team:
                if player_assignment[player_idx] == Role.SPY:
                    if n_more_fails_allowed > 0:
                        quest_votes.append(QuestVote.FAIL)
                        n_more_fails_allowed -= 1
                    else:
                        quest_votes.append(QuestVote.SUCCESS)
                else:
                    quest_votes.append(QuestVote.SUCCESS)
            
            game_state.quest_votes[quest_num] = quest_votes
            
        for quest_num, quest_result in enumerate(game_state.quest_results):
            num_fails = history["numFailsHistory"][quest_num]
            resolve_who_voted_how(game_state, quest_num, num_fails)
        
        # ensure hammer rounds are all approves
        for i in range(n_rounds):
            if game_state.turns_until_hammer[i] == 0:
                game_state.team_votes[i] = [TeamVote.APPROVE] * len(player_assignment)
                # game_state.team_vote_results[i] = TeamVoteResult.APPROVED
        
        return game_state