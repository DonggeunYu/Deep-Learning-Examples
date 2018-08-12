import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')


total_epochs = 100
batch_size = 100
learning_rate = 2e-4


def Generator(z):
    gw1 = tf.Variable(tf.truncated_normal(shape=[128, 256], stddev=0.1))
    gb1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
    gw2 = tf.Variable(tf.truncated_normal(shape=[256, 784], stddev=0.1))
    gb2 = tf.Variable(tf.truncated_normal(shape=[784], stddev=0.1))

    hidden = tf.nn.relu(tf.matmul(z, gw1) + gb1)
    output = tf.nn.sigmoid(tf.matmul(hidden, gw2) + gb2)

    return output


def Discrimination(x):
    gw1 = tf.Variable(tf.truncated_normal(shape=[784, 256], stddev=0.1))
    gb1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
    gw2 = tf.Variable(tf.truncated_normal(shape=[256, 1], stddev=0.1))
    gb2 = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1))

    hidden = tf.nn.relu(tf.matmul(x, gw1) + gb1)
    output = tf.nn.sigmoid(tf.matmul(hidden, gw2) + gb2)

    return output


def Random_noise(batch_size):
    return np.random.normal(size=[batch_size, 128])


X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 256])

fake_x = Generator(Z)

result_of_fake = Discrimination(fake_x)
result_of_real = Discrimination(X)

g_loss = tf.reduce_mean(tf.log(1 - result_of_fake))
d_loss = tf.reduce_mean(tf.loq(result_of_fake) + tf.log(1- result_of_real))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, d_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(total_epochs):
        images, _ = mnist.train.next_batch(batch_size)
        gl, dl = sess.run([g_loss, d_loss], feed_dict={X: images, Z: Random_noise(batch_size)})

        print('=======Epoch: ', epoch, '=======================================')
        print('Generator: ', gl)
        print('Discrimination: ', dl)

        generated = sess.run(fake_x, feed_dict={Z: Random_noise(batch_size)})
        fig, ax = plt.subplots(1, 10, figsize=(10, 1))
        for i in range(10):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(generated[i], (28, 28)))

        plt.savefig('goblin-gan-generated/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

    print('End')
