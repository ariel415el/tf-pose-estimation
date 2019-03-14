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

V_BATCH =  3
LR = 0.1
TARGET_VAL = b_val = 1.0 #fake data (in SGD it would be different on every epoch)

opt = tf.train.GradientDescentOptimizer(LR)

## Retrieve all trainable variables you defined in your graph
tvs = tf.trainable_variables()
## Creation of a list of variables with the same shape as the trainable ones
# initialized with 0s
accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]# gradient variable list = [ (gradient,variable) ]

gvs = opt.compute_gradients(y,tvs)

accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

train_step = opt.apply_gradients([(accum_vars[i] / V_BATCH, gv[1]) for i, gv in enumerate(gvs)])


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    epochs = 10
    for i in range(epochs):
        sess.run(zero_ops)

        print( '----')
        x_before_update = x.eval()
        print('before update',x_before_update)
        for i in range(V_BATCH):
            accumm_g_vals= sess.run(accum_ops,  feed_dict={b: TARGET_VAL})
            print ('grad_vals: ',accumm_g_vals)

        sess.run(train_step)
        # print('result: ', result)
        #result = sess.run(apply_transform_op, feed_dict={b: b_val})

        x_after_update = x.eval()
        print ('after update', x_after_update)