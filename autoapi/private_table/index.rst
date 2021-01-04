:mod:`private_table`
====================

.. py:module:: private_table


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   private_table.PrivateTable



.. py:class:: PrivateTable(dataframe: DataFrame, data_domains: Dict[str, DataDomain], total_privacy_budget: PrivacyBudget, delta_prime=0.5)

   Private Table class which uses pandas dataframe as the backend.

   Differentially private mechanisms for a data table.

   Supported statistical functions:
       - mean: the private mean.

   .. method:: __repr__(self)

      Return repr(self).


   .. method:: check_data_domains(self, domains: Dict[str, DataDomain])

      Check if data in private columns are belong to the data domain `domains`.

      :param - domains: data domain specifications.
      :type - domains: dict[str, DataDomain]

      outputs:
          - whether values in the private table belonging to the `domains`.


   .. method:: mean(self, column: str, privacy_budget: PrivacyBudget) -> float

      Return a private mean using Laplace mechanism.

      :param - columne: name of the selected column.
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.


   .. method:: gaussian_mean(self, column: str, privacy_budget: PrivacyBudget) -> float

      Return a private mean using Gaussian mechanism.

      :param - columne: name of the selected column.
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.


   .. method:: std(self, column: str, privacy_budget: PrivacyBudget) -> float

      Compute the standard deviation.

      :param - columne: name of the selected column.
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.


   .. method:: var(self, column: str, privacy_budget: PrivacyBudget) -> float

      Compute the variance.

      :param - columne: name of the selected column.
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.


   .. method:: min(self, column: str, privacy_budget: PrivacyBudget) -> float

      Compute the minimum.

      :param - columne: name of the selected column.
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.


   .. method:: max(self, column: str, privacy_budget: PrivacyBudget) -> float

      Compute the maximum.

      :param - columne: name of the selected column.
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.


   .. method:: median(self, column: str, privacy_budget: PrivacyBudget) -> float

      Compute the median.

      :param - columne: name of the selected column.
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.


   .. method:: mode(self, column: str, privacy_budget: PrivacyBudget) -> int

      Compute the mode.

      :param - columne: name of the selected column.
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.


   .. method:: cat_hist(self, column: str, privacy_budget: PrivacyBudget) -> ndarray

      Compute the histogram for a categorical column.

      :param - columne: name of the selected column.
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.


   .. method:: num_hist(self, column: str, bins: Union[ndarray, List[float]], privacy_budget: PrivacyBudget) -> ndarray

      Compute the histogram for a categorical column.

      :param - columne: name of the selected column.
      :param - bins: bins of histogram
      :param - privacy: privacy budget.

      outputs:
          - the private mean of a column.



