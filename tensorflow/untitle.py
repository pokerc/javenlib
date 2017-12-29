import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('/home/javen/javenlib/images/amos17603_201704/img2.jpg')
new_img = np.copy(img)
new_img[100-32:100+32,100-32:100+32,0] = 255
new_img[200-16:200+16,200-16:200+16,0] = 255
new_img[300-8:300+8,300-8:300+8,0] = 255
new_img[217-16:217+16,628-16:628+16,0] = 255
new_img[630-8:630+8,729-8:729+8,0] = 255
plt.figure()
plt.imshow(new_img)
plt.show()