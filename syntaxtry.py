import tensorflow as tf
import numpy as np

x = np.random.rand(100).astype(np.float32)
y = x*0.1+0.3

w = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))

Y = w*x + b

loss = tf.reduce_mean(tf.square(Y-y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

for i in range(201):
    sess.run(train)
    if i%20 == 0:
        print(i,":",sess.run(w),sess.run(b))
