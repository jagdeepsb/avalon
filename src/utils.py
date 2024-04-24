from typing import List, Tuple, Dict
from enum import Enum

class Role(Enum):
    MERLIN = 'Merlin'
    RESISTANCE = 'Resistance'
    SPY = 'Spy'
    UNKNOWN = 'Unknown'
    
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