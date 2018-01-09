#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import types
import cv2
import javenlib_tf


x_axis = np.array([2,3,4,5,6])
#quality_test测得的数据
bikes_quality = np.array([[0.476,0.484],[0.372,0.448],[0.219,0.296],[0.139,0.22],[0.06,0.045]])
bark_quality = np.array([[0.008,0.012],[0,0],[0,0],[0.008,0.004],[0,0]])
graf_quality = np.array([[0.016,0.012],[0.012,0.004],[0,0],[0.02,0.0],[0,0]])
leuven_quality = np.array([[0,0],[0,0.004],[0,0],[0,0],[0,0.004]])
trees_quality = np.array([[0.231,0.2],[0.112,0.076],[0.096,0.104],[0.06,0.116],[0.024,0.04]])
ubc_quality = np.array([[0.788,0.888],[0.68,0.776],[0.652,0.76],[0.524,0.58],[0.44,0.524]])
wall_quality = np.array([[0.096,0.036],[0.016,0.028],[0.004,0.008],[0,0.008],[0,0]])
#quantity_test测得的数据
bikes_quantity = np.array([[0.508,0.62],[0.448,0.564],[0.287,0.426],[0.216,0.322],[0.136,0.133]])
bark_quantity = np.array([[0.284,0.306],[0.046,0.094],[0.186,0.158],[0.19,0.124],[0.024,0.054]])
graf_quantity = np.array([[0.084,0.088],[0.09,0.068],[0.044,0.038],[0.036,0.04],[0.038,0.038]])
leuven_quantity = np.array([[0.03,0.078],[0.11,0.074],[0.11,0.066],[0,0],[0.036,0.062]])
trees_quantity = np.array([[0.281,0.308],[0.16,0.208],[0.184,0.2],[0.184,0.178],[0.066,0.106]])
ubc_quantity = np.array([[0.852,0.912],[0.748,0.84],[0.708,0.832],[0.638,0.722],[0.542,0.654]])
wall_quantity = np.array([[0.114,0.088],[0.03,0.042],[0.024,0.05],[0.026,0.042],[0.016,0.016]])
plt.plot(x_axis,wall_quantity[:,0])
plt.plot(x_axis,wall_quantity[:,1])
plt.show()


