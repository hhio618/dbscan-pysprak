import os
import numpy as np


def load_patches():
    return np.genfromtxt("data%spatches.txt" % os.sep, delimiter=',').T