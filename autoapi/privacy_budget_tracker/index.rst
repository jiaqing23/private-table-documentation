:mod:`privacy_budget_tracker`
=============================

.. py:module:: privacy_budget_tracker

.. autoapi-nested-parse::

   PrivacyBudgetTracker classes.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   privacy_budget_tracker.PrivacyBudgetTracker
   privacy_budget_tracker.SimplePrivacyBudgetTracker
   privacy_budget_tracker.AdvancedPrivacyBudgetTracker



.. py:class:: PrivacyBudgetTracker(total_privacy_budget: PrivacyBudget)

   Bases: :class:`abc.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: SimplePrivacyBudgetTracker(total_privacy_budget: PrivacyBudget)

   Bases: :class:`privacy_budget_tracker.PrivacyBudgetTracker`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. method:: update_privacy_loss(self, privacy_budget: PrivacyBudget)

      Update the consumed privacy budget using a simple privacy composition theorem.
      Also check if the remain privacy budget is enough for the current query.

      :param - privacy_budget: a (epsilon, delta) budget.



.. py:class:: AdvancedPrivacyBudgetTracker(total_privacy_budget: PrivacyBudget)

   Bases: :class:`privacy_budget_tracker.PrivacyBudgetTracker`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. method:: update_privacy_loss(self, privacy_budget: PrivacyBudget, delta_prime: float, k: int = 1)



