import numpy as np
import matplotlib.pyplot as plt

a = np.array([[48, 24, 0.87679857], [64, 48, 0.81496948], [92, 52, 0.84029979], [96, 44, 0.75338501], [108, 44, 0.83071619]])
print a
print a[:,2]
print a[:,2].argsort()[-1::-1]
print a[a[:,2].argsort()[-1::-1]]