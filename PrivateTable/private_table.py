from typing import Dict, List, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from data_domain import CategoricalDataDomain, DataDomain, RealDataDomain
from privacy_budget import PrivacyBudget
from privacy_budget_tracker import SimplePrivacyBudgetTracker
from private_mechanisms import (exponential_mechanism, gaussian_mechanism,
                                histogram_mechanism, laplace_mechanism)
from utils import check_positive


class PrivateTable:
    """Private Table class which uses pandas dataframe as the backend.

    Differentially private mechanisms for a data table.

    Supported statistical functions:
        - mean: the private mean.

    """

    def __init__(self, dataframe: DataFrame, data_domains: Dict[str, DataDomain], total_privacy_budget: PrivacyBudget, delta_prime=0.5):
        """Create a private data table.

        args:
            - dataframe (DataFrame): the data source (expected a pandas dataframe).
            - data_domains (Dict[str, DataDomain]): specify the set of all posible value for each data column. It is a map from `column_name` to a `data_domain`.
            - total_privacy_budget (float): the total privacy privacy that can be consumed by the private table. When is there is no privacy budget left, stop answering queries.
        """
        super().__init__()
        self._dataframe = dataframe
        self._data_domains = data_domains
        self._columns = set(dataframe.columns.values.tolist())
        self.privacy_budget_tracker = SimplePrivacyBudgetTracker(total_privacy_budget)
        assert self.check_data_domains(data_domains) == True

    def __repr__(self):
        return (f'PrivateTable (\n'
                f'  Columns        : {self._columns},\n'
                f'  Size           : {len(self._dataframe)},\n'
                f'  Privacy budget : {self.privacy_budget_tracker.total_privacy_budget},\n'
                f'  Privacy loss   : {self.privacy_budget_tracker.consumed_privacy_budget}\n)')

    def check_data_domains(self, domains: Dict[str, DataDomain]):
        """Check if data in private columns are belong to the data domain `domains`.

        args:
            - domains (dict[str, DataDomain]): data domain specifications.

        outputs:
            - whether values in the private table belonging to the `domains`.
        """
        for col in self._columns:
            if not np.all(self._dataframe[col].apply(domains[col].contains)):
                return False
        return True

    def mean(self, column: str, privacy_budget: PrivacyBudget) -> float:
        """Return a private mean using Laplace mechanism.

        args:
            - columne: name of the selected column.
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        assert(privacy_budget.delta == 0)
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, RealDataDomain)

        mean = np.mean(self._dataframe[column])
        sensitivity = domain.length()/len(self._dataframe[column])
        noisy_mean = laplace_mechanism(mean, sensitivity, privacy_budget)

        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_mean

    def gaussian_mean(self, column: str, privacy_budget: PrivacyBudget) -> float:
        """Return a private mean using Gaussian mechanism.

        args:
            - columne: name of the selected column.
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, RealDataDomain)

        mean = np.mean(self._dataframe[column])
        sensitivity = domain.length()/len(self._dataframe[column])
        noisy_mean = gaussian_mechanism(mean, sensitivity, privacy_budget)

        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_mean

    def std(self, column: str, privacy_budget: PrivacyBudget) -> float:
        """Compute the standard deviation.

        args:
            - columne: name of the selected column.
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, RealDataDomain)

        std = np.std(self._dataframe[column])
        sensitivity = domain.length()/np.sqrt(len(self._dataframe[column]))
        noisy_std = laplace_mechanism(std, sensitivity, privacy_budget)

        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_std

    def var(self, column: str, privacy_budget: PrivacyBudget) -> float:
        """Compute the variance.

        args:
            - columne: name of the selected column.
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        # TODO: Your code here
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, RealDataDomain)

        var = np.var(self._dataframe[column])
        sensitivity = domain.length()**2/len(self._dataframe[column])
        noisy_var = laplace_mechanism(var, sensitivity, privacy_budget)

        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_var

    def min(self, column: str, privacy_budget: PrivacyBudget) -> float:
        """Compute the minimum.

        args:
            - columne: name of the selected column.
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        # TODO: Your code here
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, RealDataDomain)

        minn = np.min(self._dataframe[column])
        sensitivity = domain.length()
        noisy_min = laplace_mechanism(minn, sensitivity, privacy_budget)

        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_min

    def max(self, column: str, privacy_budget: PrivacyBudget) -> float:
        """Compute the maximum.

        args:
            - columne: name of the selected column.
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        # TODO: Your code here
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, RealDataDomain)

        maxx = np.max(self._dataframe[column])
        sensitivity = domain.length()
        noisy_max = laplace_mechanism(maxx, sensitivity, privacy_budget)

        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_max

    def median(self, column: str, privacy_budget: PrivacyBudget) -> float:
        """Compute the median.

        args:
            - columne: name of the selected column.
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        # TODO: Your code here
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, RealDataDomain)

        median = np.median(self._dataframe[column])
        sensitivity = domain.length()/2
        noisy_median = laplace_mechanism(median, sensitivity, privacy_budget)

        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_median

    def mode(self, column: str, privacy_budget: PrivacyBudget) -> int:
        """Compute the mode.

        args:
            - columne: name of the selected column.
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, CategoricalDataDomain)

        def score_function(x):
            R = np.unique(x)
            s = [np.sum(x == label) for label in R]
            return R, s

        sensitivity = 2
        noisy_mode = exponential_mechanism(self._dataframe[column], score_function, sensitivity, privacy_budget)

        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_mode

    def cat_hist(self, column: str, privacy_budget: PrivacyBudget) -> ndarray:
        """Compute the histogram for a categorical column.

        args:
            - columne: name of the selected column.
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, CategoricalDataDomain)

        key, hist = np.unique(self._dataframe[column], return_counts=True)
        noisy_hist = histogram_mechanism(hist, privacy_budget)
        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_hist

    def num_hist(self, column: str, bins: Union[ndarray, List[float]], privacy_budget: PrivacyBudget) -> ndarray:
        """Compute the histogram for a categorical column.

        args:
            - columne: name of the selected column.
            - bins: bins of histogram
            - privacy: privacy budget.

        outputs:
            - the private mean of a column.
        """
        assert column in self._columns, f'Column `{column}`is not exists.'
        assert column in self._data_domains
        domain = self._data_domains[column]
        assert isinstance(domain, RealDataDomain)

        hist, binedge = np.histogram(self._dataframe[column], bins=bins)
        noisy_hist = histogram_mechanism(hist, privacy_budget)

        self.privacy_budget_tracker.update_privacy_loss(privacy_budget)

        return noisy_hist
