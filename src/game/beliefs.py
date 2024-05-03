from typing import List, Tuple, Dict, Set
from itertools import permutations
import numpy as np

from src.game.utils import Role

ALL_ROLE_ASSIGNMENTS_CACHE: Dict[Tuple[Role], List[Tuple[Role]]] = {}

def all_role_assignments(
    roles: List[Role],
) -> List[Tuple[Role]]:
    """
    Get all possible role assignments for a game of Avalon with the given roles.
    """
    
    sorted_roles = sorted(roles, key=lambda x: x.value)
    if tuple(sorted_roles) in ALL_ROLE_ASSIGNMENTS_CACHE:
        return ALL_ROLE_ASSIGNMENTS_CACHE[tuple(sorted_roles)]
    
    seen_assigments: Set[Tuple[Role]] = set()
    all_unique_assignments = []
    for assignment in permutations(sorted_roles):
        if assignment not in seen_assigments:
            all_unique_assignments.append(assignment)
            seen_assigments.add(assignment)
            
    ALL_ROLE_ASSIGNMENTS_CACHE[tuple(sorted_roles)] = all_unique_assignments
    
    return all_unique_assignments

def num_possible_assignments(roles: List[Role]) -> int:
    """
    Get the number of possible role assignments for a game of Avalon with the given roles.
    """
    return len(all_role_assignments(roles))

def make_trivial_role_assignment_distribution(
    role_assignment: List[Role],
) -> np.ndarray:
    """
    Returns a (num_possible_assignments(roles),) array where the i-th element
    is 1 if it corresponds the assignment provided by `role_assignment` and 0 otherwise.
    """
    tuple_assignment = tuple(role_assignment)
    all_assignments = all_role_assignments(role_assignment)
    distribution = np.zeros(len(all_assignments))
    for i, assignment in enumerate(all_assignments):
        if assignment == tuple_assignment:
            distribution[i] = 1.0
    assert np.allclose(distribution.sum(), 1.0)
    return distribution