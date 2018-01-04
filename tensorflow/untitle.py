import numpy as np
import matplotlib.pyplot as plt
import types

a = np.arange(12).reshape(6,2)
b = None
if type(b) == types.NoneType:
    print 'ok'