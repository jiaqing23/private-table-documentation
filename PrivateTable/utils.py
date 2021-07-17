"""
A collection of useful functions.
"""
import math
from typing import Iterable, List

import pandas as pd

import private_mechanisms

def check_positive(x: float):
    assert x > 0, "expected a positive value."


def check_absolute_error(x: float, y: float, error: float):
    assert math.fabs(x-y) <= error
