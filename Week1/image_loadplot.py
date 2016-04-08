"""
 Basic python stuffs 
 Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""

import numpy as np # Super-important numpy! 
import os # This is for acquiring directory paths
from scipy import misc # This is for reading images
import matplotlib.pyplot as plt # This is for plotting images
import cv2

""" 
 0. Get current directory path 
 pwd: ('run' mode and 'eval with python' will show different paths!)
 cwd: it will show the same result 
"""
pwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
print pwd 
print cwd 
""" We should use cwd rather than pwd """


""" 1. Load an image """
cat = misc.imread(cwd + "/cat.jpg")
print "Type of cat is ", type(cat)
# Type of cat is  <type 'numpy.ndarray'>
print "size of the image is %s" % (cat.shape,) 
""" DONT FORGET ","!! """
# size of the image is (1026, 1368, 3)


""" 2. Show the loaded image """
plt.figure(2)
plt.imshow(cat)
plt.title("Original Image")
plt.draw()


"""
 This will not plot an image!  
 Why? 
"""


"""
 The answer is simple. 
 We need following command! 
"""
plt.show()

print "Good Job"