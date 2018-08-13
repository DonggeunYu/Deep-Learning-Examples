import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')

train_x = mnist.train.images

total_epochs = 1000
batch_size = 100
learning_rate = 2e-4


gw1 = tf.Variable(tf.truncated_normal(shape=[128, 256], stddev=0.1))
gb1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
gw2 = tf.Variable(tf.truncated_normal(shape=[256, 784], stddev=0.1))
gb2 = tf.Variable(tf.truncated_normal(shape=[784], stddev=0.1))


def Generator(z):
    hidden = tf.nn.relu(tf.matmul(z, gw1) + gb1)
    output = tf.nn.sigmoid(tf.matmul(hidden, gw2) + gb2)

    return output


dw1 = tf.Variable(tf.truncated_normal(shape=[784, 256], stddev=0.1))
db1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
dw2 = tf.Variable(tf.truncated_normal(shape=[256, 1], stddev=0.1))
db2 = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1))



def Discrimination(x):
    hidden = tf.nn.relu(tf.matmul(x, dw1) + db1)
    output = tf.nn.sigmoid(tf.matmul(hidden, dw2) + db2)

    return output


def Random_noise(batch_size):
    return np.random.normal(size=[batch_size, 128])


X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 128])

fake_x = Generator(Z)

result_of_fake = Discrimination(fake_x)
result_of_real = Discrimination(X)

g_loss = -tf.reduce_mean(tf.log(result_of_fake))
d_loss = -tf.reduce_mean(tf.log(result_of_real) + tf.log(1- result_of_fake))

g_train = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=[gw1, gb1, gw2, gb2])
d_train = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=[dw1, db1, dw2, db2])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batchs = int(train_x.shape[0] / batch_size)

    for epoch in range(total_epochs):

        for batch in range(total_batchs):
            batch_x = train_x[batch * batch_size: (batch + 1) * batch_size]
            noise = Random_noise(batch_size)

            sess.run(g_train, feed_dict={Z: noise})
            sess.run(d_train, feed_dict={X: batch_x, Z: noise})

            gl, dl = sess.run([g_loss, d_loss], feed_dict={X: batch_x, Z: noise})

        print('=======Epoch: ', epoch, '=======================================')
        print('Generator: ', gl)
        print('Discrimination: ', dl)

        if epoch % 10 == 0:
            sample_noise = Random_noise(10)
            generated = sess.run(fake_x, feed_dict={Z: sample_noise})
            fig, ax = plt.subplots(1, 10, figsize=(10, 1))
            for i in range(10):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(generated[i], (28, 28)))

            plt.savefig('goblin-gan-generated/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

    print('End')
