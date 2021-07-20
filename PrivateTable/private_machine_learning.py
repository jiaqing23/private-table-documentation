import random
from datetime import datetime
from typing import Any, Callable, Union

import numpy as np
from numpy import ndarray
from numpy.random import normal

from privacy_budget import PrivacyBudget
from privacy_budget_tracker import MomentPrivacyBudgetTracker


def private_SGD(gradient_function: Callable[[Any], Union[int, float, list, ndarray]],
                get_weights_function: Callable[[], Union[int, float, list, ndarray]],
                update_weights_function: Callable[[Any], None],
                learning_rate_function: Callable[[int], float],
                train_data: Union[list, ndarray],
                group_size: int,
                gradient_norm_bound: Union[int, float],
                number_of_steps: int,
                sigma: Union[int, float],
                moment_privacy_budget_tracker: MomentPrivacyBudgetTracker,
                test_interval: int = None,
                test_function: Callable[[], None] = None
                ):
    """This Differencial Privacy(DP) SGD proposed in https://arxiv.org/pdf/1607.00133.pdf. 
    This privacy budget is calculated using :func:`MomentPrivacyBudgetTracker <privacy_budget_tracker.MomentPrivacyBudgetTracker>`. 

    :param gradient_function: Function that receives data batch as input and return the gradient of the training model.
    :param get_weights_function: Function that return the weight of the training model.
    :param update_weights_function: Function that receives new weights as input and update the weight of the model.
    :param learning_rate_function: Fucntion that receives step size as input and return the learning rate.
    :param train_data: Array of traning data. Each element of the array is a training sample.
    :param group_size: Number of data used for training in one step.
    :param gradient_norm_bound: L2-norm bound of the gradient.
    :param number_of_steps: Number of training steps.
    :param sigma: Value of sigma. Greater sigma correspond to higher noise and lower privacy budget.
    :param moment_privacy_budget_tracker: Instance of MomentPrivacyBudgetTracker.
    :param test_interval: test_function will be triggred every test_interval steps if test_function is specify , defaults to None
    :param test_function: Fucntion to test the performance of model, defaults to None
    """

    def gaussian_noise(x, standard_deviation):
        shape = None if isinstance(x, (int, float)) else x.shape
        noise = normal(loc=0.,
                       scale=standard_deviation,
                       size=shape)
        return x + noise

    random.seed(datetime.now())
    idx = list(range(len(train_data)))
    random.shuffle(idx)
    train_data = np.array(train_data)[idx]
    number_of_group = len(train_data)//group_size

    for step in range(number_of_steps):
        group_id = int(random.random()*number_of_group)
        train_data_group = train_data[group_size*group_id: group_size*(group_id+1)]
        total_grad = np.array([])
        total_loss = 0

        for i in range(len(train_data_group)):
            grad = gradient_function(train_data_group[i])
            if isinstance(grad, int) or isinstance(grad, float):
                grad /= max(1, grad**2/gradient_norm_bound)
            elif isinstance(grad, list) or isinstance(grad, ndarray):  # Either list/array or list/array of list/array
                grad = np.array(grad, dtype=object)
                grad /= max(1, np.linalg.norm(np.hstack([np.array(i).flatten() for i in grad]))/gradient_norm_bound)
            else:
                raise(TypeError("Data type returned by gradient_function should be either int, float, list or numpy ndarray"))
            total_grad = (total_grad + grad) if i > 0 else grad

        if isinstance(total_grad, int) or isinstance(total_grad, float):
            total_grad = gaussian_noise(grad, sigma*gradient_norm_bound)
        elif isinstance(total_grad, ndarray):
            total_grad = np.array([gaussian_noise(i, sigma*gradient_norm_bound) for i in total_grad], dtype=object)
        else:
            raise(TypeError)
        total_grad /= len(train_data_group)

        weights = get_weights_function()
        lr = learning_rate_function(step+1)
        if isinstance(total_grad, ndarray):
            weights = np.array(weights)
        weights = weights - lr * total_grad
        update_weights_function(weights)

        if test_function and test_interval and (step+1) % test_interval == 0:
            test_function()

    moment_privacy_budget_tracker.update_privacy_loss(sampling_ratio=group_size/len(train_data),
                                                      sigma=sigma,
                                                      steps=number_of_steps,
                                                      moment_order=32,
                                                      target_delta=0.5/len(train_data))
