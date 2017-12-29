import numpy as np

a = np.arange(20).reshape(10,2)
for i in range(len(a)):
    print i
    a = np.delete(a,0,axis=0)
    print 'a.shape',a.shape