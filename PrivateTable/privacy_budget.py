"""
PrivacyBudget classes.
"""

from typing import List


class PrivacyBudget:
    """A :math:`(\epsilon,\delta)`-privacy budget class with useful operators:

    - `less than or equal to` operator.
    - `add` operator.
    - `equal` operator.
    """

    def __init__(self, epsilon: float, delta: float = 0.):
        """
        :param epsilon: Value of epsilon :math:`\epsilon`
        :type epsilon: float
        :param delta: Value of delta :math:`\delta`, defaults to 0
        :type delta: float, optional
        """
        assert epsilon >= 0, "expecting a non-negative value."
        assert delta >= 0, "expecting a non-negative value."
        self.epsilon = epsilon
        self.delta = delta

    def __iter__(self):
        return iter([self.epsilon, self.delta])

    def __le__(self, other) -> bool:
        """return if one privacy budget is less than or equal to the other privacy budget."""
        return (self.epsilon <= other.epsilon) and (self.delta <= other.delta)

    def __add__(self, other):
        """add two privacy budgets."""
        return combine_privacy_losses([self, other])

    def __eq__(self, other):
        return (self.epsilon == other.epsilon) and (self.delta == other.delta)

    def __repr__(self):
        return f'({self.epsilon}, {self.delta})-DP'


def combine_privacy_losses(losses: List[PrivacyBudget]) -> PrivacyBudget:
    """Use Theorem 3.16 in The Algorithmic Foundations of Differential Privacy to compute the total privacy loss.

    :param losses: List of privacy losses
    :return: The total privacy loss
    """
    e = [(x+y) for (x, y) in zip(*losses)]  # type: ignore
    return PrivacyBudget(*e)
