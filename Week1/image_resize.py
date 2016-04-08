"""
 Resize Images
 Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""

import numpy as np 
import os 
from scipy import misc 
import matplotlib.pyplot as plt 

"""
 1. Load the Cat Image
"""
cwd = os.getcwd()
cat = misc.imread(cwd + "/cat.jpg")
print "\nsize of cat is %s" % (cat.shape,)
print "type of cat is", type(cat)

"""
 2. Resize the Image
"""
catsmall = misc.imresize(cat, [100, 100, 3])
print "\nsize of catsmall is %s" % (catsmall.shape,)
print "type of catsmall is", type(catsmall)

"""
 3. Convert to Grayscale
"""
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
catsmallgray = rgb2gray(catsmall)
print "\nsize of catsmallgray is %s" % (catsmallgray.shape,)
print "type of catsmallgray is", type(catsmallgray)


"""
 4. Convert to Vector
"""
catrowvec = np.reshape(catsmallgray, (1, -1));
print "\nsize of catrowvec is %s" % (catrowvec.shape,)
print "type of catrowvec is", type(catrowvec)

"""
 5. Convert to Matrix
"""
catmatrix = np.reshape(catrowvec, (100, 100));
print "\nsize of catmatrix is %s" % (catmatrix.shape,)
print "type of catmatrix is", type(catmatrix)



"""
 6. Now, Plot things
"""
plt.figure(1)
plt.imshow(cat)
plt.title("[imshow] Original Image")
plt.draw()

plt.figure(2)
plt.imshow(catsmall)
plt.title("[imshow] Smaller Image")
plt.draw()

plt.figure(3)
plt.imshow(catsmallgray, cmap=plt.get_cmap("gray"))
plt.title("[imshow] Gray Image")
plt.colorbar()
plt.draw()

plt.figure(4)
plt.matshow(catrowvec, fignum=4)
plt.title("[matshow] Row Vector")
plt.colorbar()
plt.draw()

plt.figure(5)
plt.matshow(catmatrix, fignum=5, cmap=plt.get_cmap("gray"))
plt.title("[matshow] Row Vector")
plt.colorbar()
plt.draw()

plt.show()
