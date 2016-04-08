"""
 Python 101
"""
import numpy as np
import tensorflow as tf
import os 
from scipy import misc
import matplotlib.pyplot as plt 
import cv2

# Get current folder 
cwd = os.getcwd()
print "Current Folder is %s" %(cwd) 

# Load Image 
cat = misc.imread(cwd + "/cat.jpg")
print "Type of cat is %s" % type(cat)
print "Shape of cat is %s" % (cat.shape,)

cat2 = cv2.imread(cwd + "/cat.jpg")
print "Shape of cat2 is %s" % (cat2.shape,)
cv2.imwrite("cat2.png", cat2)

# Extract single channel
cat_R = cat[:, :, 2]
print "Type of cat is %s" % type(cat_R)
print "Shape of cat is %s" % (cat_R.shape,)

# Plot!!
plt.figure(0)
plt.imshow(cat)
plt.title("Original Image")


# plt.figure(1)
# 1.
# plt.imshow(cat_R)

# 2.
a = plt.matshow(cat_R, fignum=1, cmap=plt.get_cmap("gray"))
plt.colorbar(a)

# 3.
# plt.colorbar(plt.matshow(cat_R, fignum=1, cmap=plt.get_cmap("gray")))

plt.title("Red Image")



# Resize
catsmall = misc.imresize(cat, [100, 100, 1])
print "\nsize of catsmall is %s" % (catsmall.shape,)
print "type of catsmall is", type(catsmall)

def rgb2gray(rgb):
    return np.dot(rgb[... , :3], [0.299, 0.587, 0.114])
catsmallgray = rgb2gray(catsmall)

print "\nsize of catsmallgray is %s" % (catsmallgray.shape,)
print "type of catsmallgray is", type(catsmallgray)

# Convert to Vector // 
catrowvec = np.reshape(catsmallgray, (1, -1))
print "\nsize of catrowvec is %s" % (catrowvec.shape,)
print "type of catrowvec is", type(catrowvec)

# Convert to Matrix
catmatrix = np.reshape(catrowvec, (100, -1));
print "\nsize of catmatrix is %s" % (catmatrix.shape,)
print "type of catmatrix is", type(catmatrix)





# Plot 
print "\n\n"
plt.show()
print "AAAA"
