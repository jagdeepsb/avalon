from typing import List, Callable, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

from src.game_state import AvalonGameState
from src.players.player import AvalonPlayer
from src.utils import (
    Role, TeamVote, TeamVoteResult, QuestVote,
    QuestResult, RoundStage, GameStage,
)
from src.simulator import AvalonSimulator


class AvalonArena():
    def __init__(
        self,
        roles: List[Role],
        player_factory_1: Callable[[Role], AvalonPlayer],
        player_factory_2: Callable[[Role], AvalonPlayer],
        num_games: int,
    ) -> None:
        """
        Pits two avalon strategies against each other in a series of games.
        Reports a series of statistics about the games played.
        roles: list of roles to play with
        player_factory_1: factory that creates players for strategy 1
        player_factory_2: factory that creates players for strategy 2
        num_games: number of games to play
        """
        
        self.roles = roles
        self.player_factory_1 = player_factory_1
        self.player_factory_2 = player_factory_2
        self.num_games = num_games
        
        # (strategy number, number of players from strategy 1, player role, spy win or resistance win)
        self.history: List[Tuple[int, int, Role, GameStage]] = [] 
        self._run()
        
    def get_combined_factory(self, n_strategy_1: int) -> Callable[[Role], AvalonPlayer]:
        """
        Create a factory that assigns the first n_strategy_1 players to strategy 1, and the rest to strategy 2.
        """
        def combined_factory(role: Role, i: int) -> AvalonPlayer:
            if i < n_strategy_1:
                return self.player_factory_1(role, i)
            else:
                return self.player_factory_2(role, i)
        return combined_factory
    
    def _run(self) -> None:
        """
        Run the arena and store statistics of the games played.
        """
        
        n_strategy_1 = 0
        for i in tqdm(range(self.num_games)):
            simulator = AvalonSimulator(
                self.roles,
                self.get_combined_factory(n_strategy_1),
            )
            final_game_state = simulator.run_to_completion()
            for i, player_role in enumerate(simulator.game_state.player_assignments):
                self.history.append((
                    0 if i < n_strategy_1 else 1,
                    n_strategy_1, player_role, final_game_state.game_stage
                ))
            n_strategy_1 = (n_strategy_1 + 1) % (len(self.roles)+1)
            
    def get_win_rates_by_role(
        self,
        n_strategy_1: Optional[int] = None,
    ) -> Tuple[Dict[Role, float], Dict[Role, float]]:
        """
        Get the win rates of the two strategies played in games where there were 
        n_strategy_1 players assigned to strategy 1 and len(roles) - n_strategy_1 
        players assigned to strategy 2.
        
        n_strategy_1: number of players assigned to strategy 1, a number between 0 and len(roles), inclusive. 
                      If None, return the overall win rates of the two strategies.
        """
        
        strategy_wins, strategy_total_games_played = self._get_wins_total_games(n_strategy_1)
            
        # Compute win rates
        strategy_win_rates = []
        for i in range(2):
            strategy_win_rates.append({})
            for role in set(self.roles):
                if strategy_total_games_played[i][role] == 0:
                    strategy_win_rates[i][role] = 0
                else:
                    strategy_win_rates[i][role] = strategy_wins[i][role] / strategy_total_games_played[i][role]
        
        return tuple(strategy_win_rates)


    def get_overall_win_rates(
        self,
        n_strategy_1: Optional[int] = None,
    ) -> Tuple[float]:
        """
        Get overlal the win rates of the two strategies played in games where there were 
        n_strategy_1 players assigned to strategy 1 and len(roles) - n_strategy_1 
        players assigned to strategy 2.
        
        n_strategy_1: number of players assigned to strategy 1, a number between 0 and len(roles), inclusive. 
                      If None, return the overall win rates of the two strategies.
        """
        
        strategy_wins, strategy_total_games_played = self._get_wins_total_games(n_strategy_1)
            
        # Compute win rates
        strategy_win_rates = []
        for i in range(2):
            total_wins = sum(strategy_wins[i].values())
            total_games_played = sum(strategy_total_games_played[i].values())
            if total_games_played == 0:
                strategy_win_rates.append(0)
            else:
                strategy_win_rates.append(total_wins / total_games_played)
        
        return tuple(strategy_win_rates)
    

    def _get_wins_total_games(
        self,
        n_strategy_1: Optional[int] = None,
    ) -> Tuple[List[Dict[Role, float]], List[Dict[Role, float]]]:
        """
        n_strategy_1: number of players assigned to strategy 1, a number between 0 and len(roles), inclusive. 
                      If None, return the overall win rates of the two strategies.
        """
        
        assert n_strategy_1 is None or (n_strategy_1 >= 0 and n_strategy_1 <= len(self.roles))
        
        strategy_wins = []
        strategy_total_games_played = []
        
        # Initialize dicts
        for i in range(2):
            strategy_wins.append({})
            strategy_total_games_played.append({})
            for role in set(self.roles):
                strategy_wins[i][role] = 0
                strategy_total_games_played[i][role] = 0
        
        # Unpack history and count wins and total games played    
        for i, n_s1, player_role, game_stage in self.history:
            if n_strategy_1 is not None and n_s1 != n_strategy_1:
                continue
            
            if game_stage == GameStage.RESISTANCE_WIN:
                if player_role in [Role.RESISTANCE, Role.MERLIN]:
                    strategy_wins[i][player_role] += 1
                strategy_total_games_played[i][player_role] += 1
            elif game_stage == GameStage.SPY_WIN:
                if player_role in [Role.SPY]:
                    strategy_wins[i][player_role] += 1
                strategy_total_games_played[i][player_role] += 1
            
        return strategy_wins, strategy_total_games_played