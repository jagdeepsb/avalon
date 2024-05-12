import torch
import numpy as np

from src.belief_models.base import BeliefModel
from src.game.utils import Role
from src.game.game_state import AvalonGameState
from src.game.beliefs import Belief, all_possible_ordered_role_assignments


class GroundTruthBeliefModel(BeliefModel):
    def __call__(
        self,
        game_state: AvalonGameState,
        player_index: int,
        device: torch.device = torch.device("cpu")
    ) -> Belief:
        """
        Get the belief of the player with the given index.
        This doesn't use any models, and can be used for testing training.
        Use `belief.distribution` to get the probabilities of each role assignment.
        """
        role = game_state.player_assignments[player_index]
        game_progress = game_state.quest_num/5.0
        belief_role_assignments = all_possible_ordered_role_assignments(game_state.player_assignments)
        
        if not role in [Role.RESISTANCE, Role.SPY, Role.MERLIN]:
            raise ValueError(f"Unknown role: {role}")
        
        if role == Role.MERLIN:
            belief = Belief.trivial_distribution(
                game_state.player_assignments
            )
            return belief
        
        probs = game_state.get_trainable_belief_distribution(player_index, constrained=True)
        
        # add some noise to the belief. Less noise as the game progresses
        # weight = game_progress/1.5 if role == Role.RESISTANCE else game_progress/3.0
        
        # No noise for now
        weight = 1.0
        
        probs = weight*probs + (1-weight)*np.random.normal(0, 0.20, len(probs))
        probs = probs / probs.sum()
        
        belief = Belief(belief_role_assignments, probs)
        
        if role == Role.RESISTANCE:
            # condition on the fact that the player is resistance
            belief = belief.condition_on(
                [Role.RESISTANCE if i == player_index else Role.UNKNOWN for i in range(len(game_state.player_assignments))]
            )
            
            # rule out impossible role assignments given the quest history
            belief = belief.condition_on_quest_history(game_state.quest_teams, game_state.quest_votes)
        
        if role == Role.SPY:
            # condition on the fact that you know who the spies are
            belief = belief.condition_on(
                [Role.SPY if role == Role.SPY else Role.UNKNOWN for role in game_state.player_assignments]
            )
        
        return belief