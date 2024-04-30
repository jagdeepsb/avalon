from typing import List, Tuple, Dict
from enum import Enum
import numpy as np

DATA_DIR = '.data'

class Role(Enum):
    MERLIN = 0
    RESISTANCE = 1
    SPY = 2
    
    # Unknown must be the last role
    UNKNOWN = 3
    
    @classmethod
    def one_hot_dim(cls) -> int:
        return len(cls) - 1
    
    def as_one_hot(self) -> np.ndarray:
        """
        Role as one hot encoding. Unknown is all zeros.
        """
        n_roles = len(Role)-1
        one_hot = np.zeros(n_roles)
        one_hot[self.value] = 1
        return one_hot
        
    
class TeamVote(Enum):
    APPROVE = 'Approve'
    REJECT = 'Reject'
    
class TeamVoteResult(Enum):
    APPROVED = 'Approved'
    REJECTED = 'Rejected'
    
class QuestVote(Enum):
    SUCCESS = 'Success'
    FAIL = 'Fail'
    
class QuestResult(Enum):
    SUCCEEDED = 'Succeeded'
    FAILED = 'Failed'
    
class RoundStage(Enum):
    TEAM_PROPOSAL = 'Team Proposal'
    TEAM_VOTE = 'Team Vote'
    QUEST_VOTE = 'Quest Vote'
    
class GameStage(Enum):
    IN_PROGRESS = 'In Progress'
    MERLIN_VOTE = 'Merlin Vote'
    RESISTANCE_WIN = 'Resistance Win'
    SPY_WIN = 'Spy Win'
    
TEAM_SIZE_BY_N_PLAYERS_AND_QUEST_NUM: Dict[int, Dict[int, int]] = {
    5: {1: 2, 2: 3, 3: 2, 4: 3, 5: 3},
    6: {1: 2, 2: 3, 3: 4, 4: 3, 5: 4},
    7: {1: 2, 2: 3, 3: 3, 4: 4, 5: 4},
    8: {1: 3, 2: 4, 3: 4, 4: 5, 5: 5},
    9: {1: 3, 2: 4, 3: 4, 4: 5, 5: 5},
    10: {1: 3, 2: 4, 3: 4, 4: 5, 5: 5}
}

ROLES_CAN_SEE: Dict[Role, List[Role]] = {
    Role.MERLIN: [Role.SPY, Role.MERLIN, Role.RESISTANCE],
    Role.RESISTANCE: [],
    Role.SPY: [Role.SPY],
}