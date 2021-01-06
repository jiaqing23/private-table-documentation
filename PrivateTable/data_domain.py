import abc
from abc import ABC
from typing import Any, Iterable, Union


class DataDomain(ABC):
    """Representing the set of all possible values for a column in the private table.

    There are two sub-types:
        - RealDataDomain: a range of real value [left, right]
        - CategoricalDataDomain: a list of discrete values [a, b, c, d]
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def contains(self, value: Union[float, Any]):
        pass


class RealDataDomain(DataDomain):
    """A range of real values: [left, right].
    """

    def __init__(self, lower_bound: float, upper_bound: float):
        super().__init__()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        assert lower_bound <= upper_bound

    def contains(self, value: Union[float, Any]) -> bool:
        return self._lower_bound <= value <= self._upper_bound

    def length(self) -> float:
        return self._upper_bound - self._lower_bound

    def __repr__(self):
        return 'real interval [{self._lower_bound}, {self._upper_bound}]'


class CategoricalDataDomain(DataDomain):
    """A list of discrete values.
    """

    def __init__(self, values: Iterable[Any]):
        super().__init__()
        self._values = set(values)

    def contains(self, value: Union[float, Any]) -> bool:
        return value in self._values

    def __repr__(self):
        return f'a set of {self._values}'
