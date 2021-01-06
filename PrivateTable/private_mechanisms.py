from typing import Any, Callable, Dict, NamedTuple, Union

import numpy as np
from numpy import ndarray
from numpy.random import choice, laplace, normal

from privacy_budget import PrivacyBudget
from privacy_budget_tracker import SimplePrivacyBudgetTracker
from utils import check_positive


def laplace_mechanism(x: Union[int, float, ndarray], sensitivity: float, privacy_budget: PrivacyBudget) -> Union[float, ndarray]:
    """Differentially private Laplace mechanism. Add Laplacian noise to the value:

    .. math::
            x + Laplace\left(\mu=0, \sigma=\\frac{\Delta f}{\epsilon}\\right)

    The result guarantees :math:`(\epsilon,0)`-differential privacy.

    :param x: A sensitive data
    :param sensitivity: The global sensitivity :math:`\Delta f` of `x`
    :param privacy_budget: The privacy budget :math:`(\epsilon,0)` used for the outputs
    :return: The noisy data
    """
    check_positive(privacy_budget.epsilon)
    check_positive(sensitivity)

    shape = (1, ) if isinstance(x, (int, float)) else x.shape
    noise = laplace(loc=0., scale=sensitivity / privacy_budget.epsilon, size=shape)
    return x + noise


def gaussian_mechanism(x: Union[int, float, ndarray], sensitivity: float, privacy_budget: PrivacyBudget) -> Union[float, ndarray]:
    """Differentially private Gaussian mechanism. Add Gaussian noise to the value:

    .. math::
            x + Normal\left(\mu=0, \sigma=\\frac{\sqrt{2\log(1.25/\delta)}\Delta f}{\epsilon} \\right)

    The result guarantees :math:`(\epsilon,\delta)`-differential privacy.

    :param x: A sensitive data
    :param sensitivity: The global sensitivity :math:`\Delta f` of `x`
    :param privacy_budget: The privacy budget :math:`(\epsilon,\delta)` used for the outputs
    :return: The noisy data
    """
    check_positive(privacy_budget.epsilon)
    check_positive(privacy_budget.delta)
    check_positive(sensitivity)

    shape = (1, ) if isinstance(x, (int, float)) else x.shape
    noise = normal(loc=0.,
                   scale=np.sqrt(2 * np.log(1.25/privacy_budget.delta)) * sensitivity / privacy_budget.epsilon,
                   size=shape)
    return x + noise


def histogram_mechanism(x: ndarray, privacy_budget: PrivacyBudget) -> ndarray:
    """Differentially private histogram mechanism. Add Laplacian noise to the value:
    
    .. math::
            x + Laplace\left(\mu=0, \sigma=\\frac{\Delta f}{\epsilon}\\right)

    The result guarantees :math:`(\epsilon,0)`-differential privacy.

    :param x: A sensitive data
    :param sensitivity: The global sensitivity :math:`\Delta f` of `x`
    :param privacy_budget: The privacy budget :math:`(\epsilon,0)` used for the outputs
    :return: The noisy data
    """
    return laplace_mechanism(x=x, sensitivity=2, privacy_budget=privacy_budget)


def exponential_mechanism(x: ndarray, score_function: Callable[[ndarray], ndarray], sensitivity: float, privacy_budget: PrivacyBudget) -> Any:
    """Differentially private exponantial mechanism. Each keys sampling by probability proportional to:
            np.exp(epsilon*score/(2*sensitivity))

    The result guarantees `privacy_budget`-differential privacy.

    args:
        - x (ndarray): a sensitive data (float/vector/array).
        - score_function: a function to receive `x` and return a dictionary with items {`element`: `score`}
        - sensitivity: the global sensitivity of `x`.
        - privacy_budget: the privacy budget used for the outputs.

    outputs:
        - the sampled element
    """
    check_positive(privacy_budget.epsilon)
    check_positive(sensitivity)

    R, s = score_function(x)
    s -= np.max(s)
    probability = [np.exp(privacy_budget.epsilon*score/(2*sensitivity)) for score in s]
    probability /= np.sum(probability)
    return choice(R, p=probability)
