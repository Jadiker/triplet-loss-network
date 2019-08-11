# pylint: disable = wildcard-import, unused-wildcard-import
import numpy as np
from np_triplet_loss import *

def test_accuracy():
    # UPGRADE make more test cases
    vector1 = np.asarray([0, 1, 0])
    vector2 = np.asarray([0, 4, 0])
    assert triplet_loss(vector1, vector2) == 3
