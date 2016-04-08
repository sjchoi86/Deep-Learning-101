"""
 Load Images from Folder
 Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""

import numpy as np 
import os 
import matplotlib.pyplot as plt 
from scipy import misc 


"""
 Load Images from Folder to "imgs" List  
"""
cwd  = os.getcwd()
imgs = []
names = []
path = cwd + "/images"
valid_exts = [".jpg",".gif",".png",".tga"]

print os.listdir(path)



for f in os.listdir(path):
    print "f: ", f
    # For all files 
    ext = os.path.splitext(f)[1]
    print "ext: ", ext
    # Check types 
    if ext.lower() not in valid_exts:
        continue
    fullpath = os.path.join(path,f)
    imgs.append(misc.imread(fullpath))
    names.append(os.path.splitext(f)[0]+os.path.splitext(f)[1])
 
# Check
print "Type of 'imgs': ", type(imgs)
print "Length of 'imgs': ", len(imgs)
# for i in range(0, len(imgs)):
#     curr_img = imgs[i]
i = 0
for curr_img in imgs:
    i = i + 1
    print "" # Just for black line
    print i, "Type of 'curr_img': ", type(curr_img)
    print i, "Size of 'curr_img': %s" % (curr_img.shape,)

"""    
# Plot
for i in range(0, len(imgs)):
    curr_img = imgs[i]
    plt.figure(i)
    plt.imshow(curr_img)
    plt.title("Img" + str(i) + " " + names[i])
        

plt.draw()
plt.show()

"""