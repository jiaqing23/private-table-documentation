:mod:`private_mechanisms`
=========================

.. py:module:: private_mechanisms


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   private_mechanisms.laplace_mechanism
   private_mechanisms.gaussian_mechanism
   private_mechanisms.histogram_mechanism
   private_mechanisms.exponential_mechanism


.. function:: laplace_mechanism(x: Union[int, float, ndarray], sensitivity: float, privacy_budget: PrivacyBudget) -> Union[float, ndarray]

   Differentially private Laplace mechanism. Add Laplacian noise to the value:
           x + Laplace(loc=0, scale=sensitivity/privacy_buget)

   The result guarantees `privacy_budget`-differential privacy.

   :param - x: a sensitive data (float/vector/array).
   :type - x: ndarray
   :param - sensitivity: the global sensitivity of `x`.
   :param - privacy_budget: the privacy privacy used for the outputs.

   outputs:
       - the noisy data.


.. function:: gaussian_mechanism(x: Union[int, float, ndarray], sensitivity: float, privacy_budget: PrivacyBudget) -> Union[float, ndarray]

   Differentially private Gaussian mechanism. Add Gaussian noise to the value:
           x + Normal(loc=0, scale=np.sqrt(2*np.log(1.25/delta)*sensitivity**2/epsilon**2)

   The result guarantees `privacy_budget`-differential privacy.

   :param - x: a sensitive data (float/vector/array).
   :type - x: ndarray
   :param - sensitivity: the global sensitivity of `x`.
   :param - privacy_budget: the privacy privacy used for the outputs.

   outputs:
       - the noisy data.


.. function:: histogram_mechanism(x: ndarray, privacy_budget: PrivacyBudget) -> ndarray

   Differentially private histogram mechanism. Add Laplacian noise to the value:
           x + Laplace(loc=0, scale=sensitivity/privacy_buget)

   The result guarantees `privacy_budget`-differential privacy.

   :param - x: a sensitive data (float/vector/array).
   :type - x: ndarray
   :param - sensitivity: the global sensitivity of `x`.
   :param - privacy_budget: the privacy privacy used for the outputs.

   outputs:
       - the noisy data.


.. function:: exponential_mechanism(x: ndarray, score_function: Callable[[ndarray], ndarray], sensitivity: float, privacy_budget: PrivacyBudget) -> Any

   Differentially private exponantial mechanism. Each keys sampling by probability proportional to:
           np.exp(epsilon*score/(2*sensitivity))

   The result guarantees `privacy_budget`-differential privacy.

   :param - x: a sensitive data (float/vector/array).
   :type - x: ndarray
   :param - score_function: a function to receive `x` and return a dictionary with items {`element`: `score`}
   :param - sensitivity: the global sensitivity of `x`.
   :param - privacy_budget: the privacy privacy used for the outputs.

   outputs:
       - the sampled element


