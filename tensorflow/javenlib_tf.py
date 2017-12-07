#encoding=utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def image_resize(image,rows,columns,toGray=True):
    new_image = tf.image.resize_images(image,[rows,columns],method=1)
    new_image = tf.to_float(tf.image.rgb_to_grayscale(new_image))
    return new_image