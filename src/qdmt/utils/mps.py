from ncon import ncon
import numpy as np
from typing import Self

def trotter_step(mps, U: np.ndarray, start: int = 0, stop: int = None):

        L = mps.ndim - 2
        N = mps.ndim

        if stop is not None and stop >= L:
            raise ValueError("Cannot stop after the end of chain")
        
        stop = stop if stop else L - 1
        stop = stop + (stop - start) % 2
        
        if start < 0 or start > stop or start >= L:
            raise ValueError("Invalid start index")

        tensors = [
                mps,
                *[U for _ in range(start, stop, 2)]
            ]
        
        indices = [
            [-1] + [-(i+2) for i in range(0, start)] + [i+1 for i in range(start, stop)] + [-(i+2) for i in range(stop, L)] + [-N],
            *[[i+1, i+2, -(i+2), -(i+3)] for i in range(start, stop, 2)]
        ]

        return ncon(tensors, indices)