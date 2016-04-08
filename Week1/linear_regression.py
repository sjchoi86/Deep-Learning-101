"""
 Linear Regression with TensorFlow
 Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def f(x, a, b):
    n    = train_X.size
    vals = np.zeros((1, n))
    for i in range(0, n):
        ax  = np.multiply(a, x.item(i))
        val = np.add(ax, b)
        vals[0, i] = val
    return vals

"""
 Generate Training Data 
"""
Wref = 0.7
bref = -1.
n    = 20
noise_var = 0.001
train_X   = np.random.random((1, n))
ref_Y     = f(train_X, Wref, bref)
train_Y   = ref_Y + np.sqrt(noise_var)*np.random.randn(1, n)
n_samples = train_X.size # <= Just for using size operator 
print ""
print " Type of 'train_X' is ", type(train_X)
print " Shape of 'train_X' is %s" % (train_X.shape,)
print " Type of 'train_Y' is ", type(train_Y)
print " Shape of 'train_Y' is %s" % (train_Y.shape,)
# print " 'train_X' is ", train_X
# print " 'ref_Y' is ", train_Y
# print " 'train_Y' is ", train_Y

"""
 Plot 'ref_Y' and 'train_Y'
"""
plt.figure(1)
# plt.plot(train_X, ref_Y, 'ro', label='Original data')
# plt.plot(train_X, train_Y, 'bo', label='Training data')
plt.plot(train_X[0, :], ref_Y[0, :], 'ro', label='Original data')
plt.plot(train_X[0, :], train_Y[0, :], 'bo', label='Training data')
plt.axis('equal')
plt.legend(loc='lower right')

"""
 Linear Regression
"""
# Parameters 
training_epochs = 2000
display_step    = 50


# Set TensorFlow Graph
X = tf.placeholder(tf.float32, name="input")
Y = tf.placeholder(tf.float32, name="output")
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Construct a Model
activation = tf.add(tf.mul(X, W), b)

# Define Error Measure and Optimizer
learning_rate   = 0.01
cost = tf.reduce_mean(tf.pow(activation-Y, 2))

# learning_rate   = 0.001
# cost = tf.sqrt(tf.reduce_sum(tf.pow(activation-Y, 2)))


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

"""
 tf.reduce_sum()
 tf.reduce_mean()
 _____

 tf.pow(Yhat, Y, 2)
 tf.nn.softmax_cross_entropy_with_logits(Yhat, Y)
 _____

 tf.train.GradientDescentOptimizer(0.05).minimize(cost)
 tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
"""

# Initializer
init = tf.initialize_all_variables()

# Run! 
sess = tf.Session()
# Initialize
sess.run(init)    
for epoch in range(training_epochs):
    for (x, y) in zip(train_X[0, :], train_Y[0, :]):
        # print "x: ", x, " y: ", y
        sess.run(optimizer, feed_dict={X:x, Y:y})
    
    # Check cost
    if epoch % display_step == 0:
        costval = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        print "Epoch:", "%04d"%(epoch+1), "cost=", "{:.5f}".format(costval)
        Wtemp = sess.run(W)
        btemp = sess.run(b)
        print " Wtemp is", "{:.4f}".format(Wtemp), "btemp is", "{:.4f}".format(btemp)
        print " Wref is", "{:.4f}".format(Wref), "bref is", "{:.4f}".format(bref)
        
# Final W and b
Wopt = sess.run(W)
bopt = sess.run(b)
fopt = f(train_X, Wopt, bopt)

# Plot
plt.figure(2)
plt.plot(train_X[0, :], ref_Y[0, :], 'ro', label='Original data')
plt.plot(train_X[0, :], train_Y[0, :], 'bo', label='Training data')
plt.plot(train_X[0, :], fopt[0, :], 'k-', label='Fitted Line')
plt.axis('equal')
plt.legend(loc='lower right')


"""
 This should come at the end of the code 
"""
plt.show()
