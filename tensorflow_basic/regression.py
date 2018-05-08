import tensorflow as tf

# Linear model y = Wx + b
W = tf.get_variable("W", shape=[1], initializer=tf.random_uniform_initializer(-1, 1))
b = tf.get_variable("b", shape=[1], initializer=tf.random_uniform_initializer(-1, 1))

# Placeholder for input and prediction
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

y_pred = W * x + b

loss = tf.reduce_sum(tf.square(y_pred - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [1.5, 3.5, 5.5, 7.5]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train, {x:x_train, y:y_train})
        if i%100==0:
            l_cost = sess.run(loss, {x:x_train, y:y_train})
            print(f"i: {i} cost: {l_cost}")

            save_path = saver.save(sess, "tmp/model.ckpt")

    l_W, l_b, l_cost  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print(f"W: {l_W} b: {l_b} cost: {l_cost}")
    # W: [ 1.99999797] b: [-0.49999401] cost: 2.2751578399038408e-11

