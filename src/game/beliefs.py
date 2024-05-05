from __future__ import annotations
from typing import List, Tuple, Dict, Set
from itertools import permutations
import numpy as np

from src.game.utils import Role

ALL_ROLE_ASSIGNMENTS_CACHE: Dict[Tuple[Role], List[Tuple[Role]]] = {}


def all_possible_ordered_role_assignments(
    roles: List[Role],
) -> List[Tuple[Role]]:
    """
    Get all possible role assignments for a game of Avalon with the given roles.
    Assignments are ordered and unique.
    
    roles: A list of roles to assign to players, in any order.
    returns: A list of all possible role assignments. Order is always the same, regardless the order of the input roles.
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

class Belief:
    """
    Defines a distribution over all possible role assignments.
    """
    
    def __init__(
        self,
        role_assignments: List[Tuple[Role]],
        probabilities: np.ndarray,
    ) -> None:
        """
        role_assignments: A list of all possible role assignments.
        probabilities: A (n,) numpy array of probabilities for each possible role assignment. Must have the same length as `role_assignments`.
        """
        self.probabilities = probabilities
        self.all_assignments = role_assignments
        
        assert len(probabilities) == len(self.all_assignments), (
            f"Length of probabilities {len(probabilities)} does not match the number of possible assignments {len(self.all_assignments)}."
        )
        
    @property
    def distribution(self) -> np.ndarray:
        return self.probabilities
        
    def get_probability(self, role_assignment: List[Role]) -> float:
        """
        Get the probability of a given role assignment under this belief distribution.
        """
        
        tuple_assignment = tuple(role_assignment)
        for p, assignment in zip(self.probabilities, self.all_assignments):
            if assignment == tuple_assignment:
                return p
        raise ValueError(f"Role assignment {role_assignment} not found in belief distribution.")
        
    def marginalize(self, role_assignment: List[Role]) -> float:
        """
        Get the marginal probability of a given role assignment under this belief distribution.
        """
        
        def is_match(assignment: Tuple[Role]) -> bool:
            for r1, r2 in zip(role_assignment, assignment):
                if r1 == Role.UNKNOWN: # catch all
                    continue
                if r1 != r2:
                    return False
            return True
        
        return sum([p for p, assignment in zip(self.probabilities, self.all_assignments) if is_match(assignment)])
    
    def get_top_k_assignments(self, k: int) -> List[Tuple[Role, float]]:
        """
        Get the top k assignments with the highest probability.
        """
        
        sorted_indices = np.argsort(self.probabilities)[::-1]
        return [(self.all_assignments[i], self.probabilities[i]) for i in sorted_indices[:k]]
        
    @classmethod
    def trivial_distribution(cls, role_assignment: List[Role]) -> Belief:
        """
        Create a trivial distribution where the probability is 1 for the given role assignment and 0 for all others.
        """
        
        tuple_assignment = tuple(role_assignment)
        all_assignments = all_possible_ordered_role_assignments(role_assignment)
        distribution = np.zeros(len(all_assignments))
        for i, assignment in enumerate(all_assignments):
            if assignment == tuple_assignment:
                distribution[i] = 1.0
        assert np.allclose(distribution.sum(), 1.0)
        
        return cls(all_assignments, distribution)
    
    @classmethod
    def smoothed_triangle_distribution(cls, role_assignment: List[Role]) -> Belief:
        """
        Create a smoothed distribution where the probability mass for each role assignment is proportional
        to some monotonic function of how many roles match the given role assignment.
        """
        
        def count_match(assignment: Tuple[Role]) -> int:
            return sum([1 for r1, r2 in zip(role_assignment, assignment) if r1 == r2])**2
        
        all_assignments = all_possible_ordered_role_assignments(role_assignment)
        distribution = np.zeros(len(all_assignments))
        for i, assignment in enumerate(all_assignments):
            distribution[i] = count_match(assignment)
        distribution /= distribution.sum()
        assert np.allclose(distribution.sum(), 1.0)
        
        return cls(all_assignments, distribution)