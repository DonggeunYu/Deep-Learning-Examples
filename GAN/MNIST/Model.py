import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')

train_x = mnist.train.images
train_y = mnist.train.labels

print('Check: ', train_x.shape, train_y.shape)

total_epochs = 100
batch_size = 100
learning_rate = 2e-4


def generator(z, reuse=False):

    if reuse==False:
        with tf.variable_scope(name_or_scope='Gen') as scope:
            gw1 = tf.Variable(tf.truncated_normal(shape=[128, 256], stddev=0.1))
            gb1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
            gw2 = tf.Variable(tf.truncated_normal(shape=[256, 784], stddev=0.1))
            gb2 = tf.Variable(tf.truncated_normal(shape=[784], stddev=0.1))
    else:
        with tf.variable_scope(name_or_scope='Gen', reuse = True) as scope :
            gw1 = tf.Variable(tf.truncated_normal(shape=[128, 256], stddev=0.1))
            gb1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
            gw2 = tf.Variable(tf.truncated_normal(shape=[256, 784], stddev=0.1))
            gb2 = tf.Variable(tf.truncated_normal(shape=[784], stddev=0.1))

    hidden = tf.nn.relu(tf.matmul(z, gw1) + gb1)
    output = tf.nn.sigmoid(tf.matmul(hidden, gw2) + gb2)

    return output


def discriminator(x, reuse=False):

    if reuse==False:
        with tf.variable_scope(name_or_scope='Dis') as scope :
            dw1 = tf.Variable(tf.truncated_normal(shape=[784, 256], stddev=0.1))
            db1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
            dw2 = tf.Variable(tf.truncated_normal(shape=[256, 1], stddev=0.1))
            db2 = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1))
    else:
        with tf.variable_scope(name_or_scope='Dis', reuse=True) as scope:
            dw1 = tf.Variable(tf.truncated_normal(shape=[784, 256], stddev=0.1))
            db1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
            dw2 = tf.Variable(tf.truncated_normal(shape=[256, 1], stddev=0.1))
            db2 = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1))

    hidden = tf.nn.relu(tf.matmul(x, dw1) + db1)
    output = tf.nn.sigmoid(tf.matmul(hidden, dw2) + db2)
    return output


def random_noise(batch_size):
    return np.random.normal(size=[batch_size, 128])


g = tf.Graph()

with g.as_default():

    X = tf.placeholder(tf.float32, [None, 784])
    Z = tf.placeholder(tf.float32, [None, 128])

    fake_x = generator(Z)

    result_of_fake = discriminator(fake_x)
    result_of_real = discriminator(X, True)

    g_loss = tf.reduce_mean(tf.log(result_of_fake))
    d_loss = tf.reduce_mean(tf.log(result_of_real) + tf.log(1 - result_of_fake))

    t_vars = tf.trainable_variables()

    g_vars = [var for var in t_vars if 'Gen' in var.name]
    d_vars = [var for var in t_vars if 'Dis' in var.name]

    optimizer = tf.train.AdamOptimizer(learning_rate)

    g_train = optimizer.minimize(-g_loss, var_list=g_vars)
    d_train = optimizer.minimize(-d_loss, var_list=d_vars)


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    total_batchs = int(train_x.shape[0] / batch_size)

    for epoch in range(total_epochs):

        for batch in range(total_batchs):
            batch_x = train_x[batch * batch_size: (batch+1) * batch_size]  # [batch_size , 784]
            batch_y = train_y[batch * batch_size: (batch+1) * batch_size]  # [batch_size,]
            noise = random_noise(batch_size)  # [batch_size, 128]

            sess.run(g_train, feed_dict={Z: noise})
            sess.run(d_train, feed_dict={X: batch_x, Z: noise})

            gl, dl = sess.run([g_loss, d_loss], feed_dict={X: batch_x, Z: noise})

        if (epoch+1) % 1 == 0 or epoch == 1:
            print('=======Epoch: ', epoch, '=======================================')
            print('Generator: ', gl)
            print('Discrimination: ', dl)

        if epoch == 0 or (epoch + 1) % 1 == 0:
            sample_noise = random_noise(10)

            generated = sess.run(fake_x , feed_dict = { Z : sample_noise})

            fig, ax = plt.subplots(1, 10, figsize=(10, 1))
            for i in range(10) :
                ax[i].set_axis_off()
                ax[i].imshow( np.reshape( generated[i], (28, 28)) )

            plt.savefig('goblin-gan-generated/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

    print('End')
