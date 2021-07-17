"""
PrivacyBudgetTracker classes.
"""

from abc import ABC
from typing import List, Union

import numpy as np

from privacy_budget import PrivacyBudget

from calculate_moment import *

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

        self.consumed_privacy_budget = e


class AdvancedPrivacyBudgetTracker(PrivacyBudgetTracker):
    """Privacy budget tracker that use advance composition theorem to update consumed privacy budget.
    """
    def update_privacy_loss(self, privacy_budget: PrivacyBudget, delta_prime: float, k: int = 1):
        """Calculate and update privacy loss of multiple query with same privacy_budget.
        :param privacy_budget: Privacy budget of query
        :param delta_prime: Value of :math:`\epsilon'`
        :param k: Number of query, defaults to 1
        """

        assert delta_prime > 0, "Value of delta should be positive"

        kfold_privacy_budget = PrivacyBudget(np.sqrt(2*k*np.log(1/delta_prime))*privacy_budget.epsilon
                                             + k*privacy_budget.epsilon*(np.exp(privacy_budget.epsilon)-1),
                                             k*privacy_budget.delta + delta_prime)
        
        e = self.consumed_privacy_budget + kfold_privacy_budget
        assert e <= self.total_privacy_budget, "there is not enough privacy budget."

        self.consumed_privacy_budget = e

class MomentPrivacyBudgetTracker(PrivacyBudgetTracker):
    """Privacy budget tracker that use moment accountant (https://arxiv.org/pdf/1607.00133.pdf) to update consumed privacy budget.
    """
    def update_privacy_loss(self, sampling_ratio: float, sigma: float, steps: int, moment_order: int = 32, 
                            target_eps: Union[float,None] = None, target_delta: Union[float,None] = None):
        """Calculate and update privacy loss. Must specify exactly either one of `target_eps` or `target_delta`.

        :param sampling_ratio: Ratio of data used to total data in one step
        :param sigma: Noise scale
        :param steps: Number of update performed
        :param moment_order: Maximum order of moment to calculate privacy budget, defaults to 32
        :param target_eps: Target value of :math:`\epsilon`, defaults to None
        :param target_delta: Target value of :math:`\delta`, defaults to None
        """
        assert (target_eps is None) or (target_eps > 0), "Value of epsilon should be positive"
        assert (target_delta is None) or (target_delta > 0), "Value of delta should be positive"

        log_moments = [(i, compute_log_moment(sampling_ratio, sigma, steps, i)) for i in range(1, moment_order + 1)]
        privacy = get_privacy_spent(log_moments, target_eps, target_delta)
        privacy_budget = PrivacyBudget(privacy[0], privacy[1])

        e = self.consumed_privacy_budget + privacy_budget
        assert e <= self.total_privacy_budget, "there is not enough privacy budget."

        self.consumed_privacy_budget = e