from typing import Callable, Iterable, Union, List, Any

import tensorflow as tf

from numpy import array, ndarray
import numpy as np
import random
import itertools
import abc
import threading
from numpy.random import normal
from tensorflow.keras import losses, Model

class FedAvgClient:
    """Client side of FederatedAveraging (FedAvg) algorithm (https://arxiv.org/pdf/1602.05629.pdf).
    The implementation is based on Tensorflow.
    """

    def __init__(self, model_fn: Callable[[], Model], loss_fn: Callable[[Any, array], array], data: Iterable):
        """
        :param model_fn: Function used to create instance of model, should be the same as server.
        :param loss_fn: Function that receives model and data batch, and return the loss value caluclated by tf.keras.losses.
        :param data: Client's traning data. Each element represent one training sample.
        """
        self.data = data
        self.model = model_fn()
        self.loss_fn = loss_fn
        self.optimizer = tf.keras.optimizers.SGD()
        
    def split_data(self, minibatch_size: int) -> list:
        """Function used to split data into minibatches with size of minibatch_size.

        :param minibatch_size: Size of minibatch.
        :return: List of batches data. Each element in the list is list of data of one minibatch.
        """
        args = [iter(self.data)] * minibatch_size
        return list([e for e in t if e != None] for t in itertools.zip_longest(*args))
    
    def train_step(self, minibatch: Union[array, list]):
        """Function for training the model using one minibatch.

        :param minibatch: List of data of minibacth.
        """
        with tf.GradientTape() as tape:
            loss = loss_fn(self.model, minibatch)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def train(self, global_weights: array, minibatch_size: int, epoch: int):
        """Function for training the model.

        :param global_weights: Global weights received from server.
        :param minibatch_size: Size of a single minibatch.
        :param epoch: Number of epoch.
        :return: return model weights and number of data used for training.
        """
        self.model.set_weights(global_weights)
        
        for _ in range(epoch):
            minibatch_data = self.split_data(minibatch_size)
            for batch in minibatch_data:
                self.train_step(batch)
        return self.model.get_weights(), len(self.data)
    
class FedAvgServer:
    """Server side of FederatedAveraging (FedAvg) algorithm (https://arxiv.org/pdf/1602.05629.pdf).
    The implementation is based on Tensorflow.
    """

    def __init__(self, model_fn: Callable[[], Model], clients: List[Any]):
        """
        :param model_fn: Function used to create instance of model, should be the same as clients.
        :param clients: List of client objects used to identify each client.
        """
        self.clients = clients
        self.model = model_fn()
        
    @abc.abstractmethod
    def send_train_request(self, client: Any, minibatch_size: int, epoch: int):
        """Abstract method for sending training request to client.

        :param client: Target client used for training.
        :param minibatch_size: Size of minibatch used by client during training.
        :param epoch: Number of epoch for training.
        """
        pass
    
    def train_client(self, client: Any, minibatch_size: int, epoch: int, res: List[Any], idx: int):
        """Intermediate function to call :func:`send_train_request`

        :param client: Target client used for training.
        :param minibatch_size: Size of minibatch used by client during training.
        :param epoch: Number of epoch for training.
        :param res: Reference to array to record the model weight returned by client.
        :param idx: Index of array res to record the model weight returned by client.
        """
        res[idx] = self.send_train_request(client, minibatch_size, epoch)
    
    def train(self, clients: List[Any], minibatch_size: int, epoch: int):
        """Function to train the model by calling clients using multithreading.

        :param clients: List of client objects used to identify each client.
        :param minibatch_size: Size of minibatch used by client during training.
        :param epoch: Number of epoch used by client for training.
        """
        threads = list()
        res = [[] for i in range(len(clients))]
        for i in range(len(clients)):
            x = threading.Thread(target=self.train_client, args=(clients[i], minibatch_size, epoch, res, i))
            threads.append(x)
            x.start()
        for thread in threads:
            thread.join()
            
        weights = [i[0] for i in res]
        number = [i[1] for i in res]
        number = np.array(number) / sum(number)
        new_weights = sum([np.array(weights[i], dtype=object)*number[i] for i in range(len(clients))])
        self.model.set_weights(new_weights)
        
    def select_clients(self, number: int) -> List[Any]:
        """Select clients from self.clients randomly.

        :param number: Number of client to select.
        :return: List of clients object in self.clients.
        """
        return np.random.choice(self.clients, size = min(number, len(self.clients)), replace = False)

def flat_clip(gradient: array, gradient_norm_bound: float) -> array:
    """Helper function used to clip gradient with L2-norm bound of gradient_norm_bound.

    :param gradient: The gradient to be clipepd.
    :param gradient_norm_bound: L2-norm bound of gradient.
    :return: The clipped gradient.
    """
    grad = np.array(gradient, dtype=object)
    grad /= max(1, np.linalg.norm(np.hstack([np.array(i).flatten() for i in grad]))/gradient_norm_bound)
    return grad

class DPFedAvgClient:
    """Client side of DP-FedAvg algorithm (https://arxiv.org/pdf/1710.06963.pdf).
    The implementation is based on Tensorflow.
    """

    def __init__(self, model_fn: Callable[[], Model], loss_fn: Callable[[Any, array], array], data: Iterable):
        """
        :param model_fn: Function used to create instance of model, should be the same as server.
        :param loss_fn: Function that receives model and data batch, and return the loss value caluclated by tf.keras.losses.
        :param data: Client's traning data. Each element represent one training sample.
        """
        self.data = data
        self.model = model_fn()
        self.loss_fn = loss_fn
        self.optimizer = tf.keras.optimizers.SGD()

    def split_data(self, minibatch_size: int) -> list:
        """Function used to split data into minibatches with size of minibatch_size.

        :param minibatch_size: Size of minibatch.
        :return: List of batches data. Each element in the list is list of data of one minibatch.
        """
        args = [iter(self.data)] * minibatch_size
        return list([e for e in t if e != None] for t in itertools.zip_longest(*args))
    

    def train_step(self, minibatch: Union[array, list], initial_weights: array, 
                    gradient_norm_bound: float):
        """Function for training the model using one minibatch.

        :param minibatch: List of data of minibacth.
        :param initial_weights: Initial global weight get from server.
        :param gradient_norm_bound: L2-norm bound of gradient.
        """
        with tf.GradientTape() as tape:
            loss = loss_fn(self.model, minibatch)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            updated_weights = np.array(self.model.get_weights(), dtype = object)
            updated_weights = initial_weights + flat_clip(updated_weights - initial_weights, gradient_norm_bound)
            self.model.set_weights(updated_weights)


    def train(self, global_weights: array, minibatch_size: int, epoch: int, gradient_norm_bound: float):
        """Function for training the model.

        :param global_weights: Global weights received from server.
        :param minibatch_size: Size of a single minibatch.
        :param epoch: Number of epoch.
        :param gradient_norm_bound: L2-norm bound of gradient.
        :return: return model gradient and number of data used for training.
        """
        self.model.set_weights(global_weights)
        initial_weights = np.array(global_weights, dtype = object)
        
        for _ in range(epoch):
            minibatch_data = self.split_data(minibatch_size)
            for batch in minibatch_data:
                self.train_step(batch, initial_weights, gradient_norm_bound)
                
        grad = np.array(self.model.get_weights(), dtype = object) - initial_weights 
        return grad, len(self.data)
    
class DPFedAvgServer:
    """Server side of DP-FedAvg algorithm (https://arxiv.org/pdf/1710.06963.pdf).
    The implementation is based on Tensorflow.
    """

    def __init__(self, model_fn: Callable[[], Model], clients: List[Any], total_data: int):
        """
        :param model_fn: Function used to create instance of model, should be the same as clients.
        :param clients: List of client objects used to identify each client.
        :param total_data: Number of total data.
        """
        self.clients = clients
        self.model = model_fn()
        
        #since we know len of data of client, and let w_hat = sum(n_clients), W = 1
        self.W = 1
        self.total_data = total_data
        
    @abc.abstractmethod
    def send_train_request(self, client: Any, minibatch_size: int, epoch: int, gradient_norm_bound: float):
        """Abstract method for sending training request to client.

        :param client: Target client used for training.
        :param minibatch_size: Size of minibatch used by client during training.
        :param epoch: Number of epoch for training.
        :param gradient_norm_bound: L2-norm bound of gradient.
        """
        pass

    def train_client(self, client: Any, minibatch_size: int, epoch: int, gradient_norm_bound: float, res: List[Any], idx: int):
        """Intermediate function to call :func:`send_train_request`

        :param client: Target client used for training.
        :param minibatch_size: Size of minibatch used by client during training.
        :param epoch: Number of epoch for training.
        :param gradient_norm_bound: L2-norm bound of gradient.
        :param res: Reference to array to record the model weight returned by client.
        :param idx: Index of array res to record the model weight returned by client.
        """
        res[idx] = self.send_train_request(client, minibatch_size, epoch, gradient_norm_bound)
    

    def train(self, clients: List[Any], minibatch_size: int, epoch: int, gradient_norm_bound: float, noise_scale: float):
        """Function to train the model by calling clients using multithreading.

        :param clients: List of client objects used to identify each client.
        :param minibatch_size: Size of minibatch used by client during training.
        :param epoch: Number of epoch used by client for training.
        :param gradient_norm_bound: L2-norm bound of gradient.
        :param noise_scale: Value of noise scale. Higher noise scale correspond to higher noise and lower privacy budget.
        """
        threads = list()
        res = [[] for i in range(len(clients))]
        for i in range(len(clients)):
            x = threading.Thread(target=self.train_client, 
                                 args=(clients[i], minibatch_size, epoch, gradient_norm_bound, res, i))
            threads.append(x)
            x.start()
        for thread in threads:
            thread.join()
            
        qW = len(clients)/len(self.clients) * self.W 
        grad = [i[0] for i in res]
        number = [i[1] for i in res]
        number = np.array(number) / self.total_data
        total_grad = sum([np.array(grad[i], dtype=object)*number[i] for i in range(len(clients))])
        total_grad /= qW
        
        sigma = noise_scale*gradient_norm_bound/qW
        total_grad = np.array([self.gaussian_noise(i, sigma) for i in total_grad], dtype=object)
        new_weights = np.array(self.model.get_weights(), dtype = object) + total_grad
        
        self.model.set_weights(new_weights)
        
    def select_clients(self, number: int) -> List[Any]:
        """Select clients from self.clients randomly.

        :param number: Number of client to select.
        :return: List of clients object in self.clients.
        """
        return np.random.choice(self.clients, size = min(number, len(self.clients)), replace = False)
    
    def gaussian_noise(self, x: Union[int, float, ndarray], standard_deviation: float):
        """Helper function for adding Gaussian noise.

        :param x: Input data
        :param standard_deviation: Standard deviation of Gaussian noise.
        :return: Input data with noise
        """
        shape = (1, ) if isinstance(x, (int, float)) else x.shape
        noise = normal(loc=0.,
                    scale=standard_deviation,
                    size=shape)
        return x + noise