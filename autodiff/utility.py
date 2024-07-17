import numpy as np
import os
import sys


def data_to_numpy(data, dtype):
    # scalar values
    if isinstance(data, int):
        return np.array(object=data)
    elif isinstance(data, float):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    # list or nested list are given
    else:
        return np.array(object=data, dtype=dtype)
