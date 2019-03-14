import tensorflow as tf

#funciton to transform gradients
def T(g, decay=1.0):
    #return decayed gradient
    return decay*g

# x variable
x = tf.Variable(10.0,name='x')
# b placeholder (simualtes the "data" part of the training)
b = tf.placeholder(tf.float32)
# make model (1/2)(x-b)^2
xx_b = 0.5*tf.pow(x-b,2)
y=xx_b

learning_rate = 1.0
opt = tf.train.GradientDescentOptimizer(learning_rate)
# gradient variable list = [ (gradient,variable) ]
gv = opt.compute_gradients(y,[x])
# transformed gradient variable list = [ (T(gradient),variable) ]
decay = 0.1 # decay the gradient for the sake of the example
tgv = [(T(g,decay=decay),v) for (g,v) in gv] #list [(grad,var)]
# apply transformed gradients (this case no transform)
apply_transform_op = opt.apply_gradients(tgv)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    epochs = 10
    for i in range(epochs):
        b_val = 1.0 #fake data (in SGD it would be different on every epoch)
        print( '----')
        x_before_update = x.eval()
        print('before update',x_before_update)

        # compute gradients
        grad_vals = sess.run([g for (g,v) in gv], feed_dict={b: b_val})
        grad_vals_2 = sess.run([g for (g, v) in gv], feed_dict={b: b_val})
        print ('grad_vals: ',grad_vals)
        # applies the gradients
        result = sess.run(apply_transform_op, feed_dict={b: b_val})

        print ('value of x should be: ', x_before_update - T(grad_vals[0], decay=decay))
        x_after_update = x.eval()
        print ('after update', x_after_update)