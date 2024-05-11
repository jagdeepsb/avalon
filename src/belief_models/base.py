import torch

from src.game.game_state import AvalonGameState
from src.game.beliefs import Belief

class BeliefModel:
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
        raise NotImplementedError("This method should be implemented by the subclass")