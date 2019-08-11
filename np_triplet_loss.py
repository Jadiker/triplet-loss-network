import numpy as np
from model_types import Encoding

def triplet_loss(v1: Encoding, v2: Encoding) -> float:
    '''Computes the triplet loss component between two encoding vectors'''
    return np.mean(np.square(v1 - v2))
