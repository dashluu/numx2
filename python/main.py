import numpy as np
import nx.nn as nn
import torch
from nx.core import arange, from_numpy, i32
from typing_extensions import Sequence
from utils import np_assert_array


def run_basic(shape1: Sequence, shape2: Sequence):
    np1 = np.random.randn(*shape1).astype(np.float32)
    np2 = np.random.randn(*shape2).astype(np.float32)
    nx1 = from_numpy(np1)
    nx2 = from_numpy(np2)
    nx3 = nx1 + nx2
    np3 = np1 + np2
    np_assert_array(nx3, np3)


if __name__ == "__main__":
    pass
