:mod:`privacy_budget`
=====================

.. py:module:: privacy_budget

.. autoapi-nested-parse::

   PrivacyBudget classes.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   privacy_budget.PrivacyBudget



Functions
~~~~~~~~~

.. autoapisummary::

   privacy_budget.combine_privacy_losses


.. py:class:: PrivacyBudget(epsilon: float, delta: float = 0.0)

   A (epsilon, delta) privacy budget class with useful operators:

   - `less than or equal to` operator.
   - `add` operator.
   - `equal` operator.

   .. method:: __iter__(self)


   .. method:: __le__(self, other) -> bool

      return if one privacy budget is less than or equal to the other privacy budget.


   .. method:: __add__(self, other)

      add two privacy budgets.


   .. method:: __eq__(self, other)

      Return self==value.


   .. method:: __repr__(self)

      Return repr(self).



.. function:: combine_privacy_losses(losses: List[PrivacyBudget]) -> PrivacyBudget

   Use Theorem 3.16 in The Algorithmic Foundations of Differential Privacy to compute the total privacy loss.

   :param losses: list of privacy losses.

   outputs:
       - the total privacy loss.


