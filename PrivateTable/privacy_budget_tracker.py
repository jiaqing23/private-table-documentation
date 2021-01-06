"""
PrivacyBudgetTracker classes.
"""

from abc import ABC
from typing import List

import numpy as np

from privacy_budget import PrivacyBudget


class PrivacyBudgetTracker(ABC):
    """Base class of privacy budget tracker.
    """
    def __init__(self, total_privacy_budget: PrivacyBudget):
        """
        :param total_privacy_budget: The total privacy budget that can be consumed by the private table. 
            When is there is no privacy budget left, stop answering queries.
        """
        self.total_privacy_budget = total_privacy_budget
        self.consumed_privacy_budget = PrivacyBudget(0., 0.)


class SimplePrivacyBudgetTracker(PrivacyBudgetTracker):
    """Privacy budget tracker that use simple composition theorem to update consumed privacy budget.
    """
    def update_privacy_loss(self, privacy_budget: PrivacyBudget):
        """Update the consumed privacy budget using a simple privacy composition theorem. 
        Also check if the remain privacy budget is enough for the current query.

        :param privacy_budget: A :math:`(\epsilon,\delta)`-privacy budget to be updated
        """
        e = self.consumed_privacy_budget + privacy_budget
        assert e <= self.total_privacy_budget, "there is not enough privacy budget."

        self.consumed_privacy_budget = self.consumed_privacy_budget + privacy_budget


class AdvancedPrivacyBudgetTracker(PrivacyBudgetTracker):
    """Privacy budget tracker that use advance composition theorem to update consumed privacy budget.
    """
    def update_privacy_loss(self, privacy_budget: PrivacyBudget, delta_prime: float, k: int = 1):
        assert(delta_prime > 0)

        kfold_privacy_budget = PrivacyBudget(np.sqrt(2*k*np.log(1/delta_prime))*privacy_budget.epsilon
                                             + k*privacy_budget.epsilon*(np.exp(privacy_budget.epsilon)-1),
                                             k*privacy_budget.delta + delta_prime)
        e = self.consumed_privacy_budget + kfold_privacy_budget
        assert e <= self.total_privacy_budget, "there is not enough privacy budget."

        self.consumed_privacy_budget = self.consumed_privacy_budget + kfold_privacy_budget

# consumed = (0, 0)
# User 1st time query with epislon = 5, kfold_privacy_budget(k=1) = (1, 0.5)
# dict = {5:1, 7:2, 3:1}
# consumed += (1, 0.5)
# 2nd time, kfold_privacy_budget(k=2) = (1.5,0.6)
# consumed += (1.5,0.6)-(1, 0.5)
# Context manager # Use max value of epsilon
# 2 query with epsilon = 5

# Ask user ---- Advance --- delta_prime
