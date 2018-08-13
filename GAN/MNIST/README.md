# MNIST_GAN

## Summary

GAN을 이용하여 이미지를 생성해내는 것이 목적이다. GAN은 경찰과 지폐위조범은 개념으로 생각하면 쉽다. 경찰은 위조된 지폐를 구별하기 위해 노력하면서 구별하는 실력이 향상되고 지폐위조범은 경찰이 구별하기 힘든 지폐를 만들기 위해 노력하면서 가짜 지폐를 만들어 내는 실력이 향상될 것이다. 이 처럼 두 모델의 경쟁을 통해 서로의 모델이 학습을 하게된다.

참고 : [GAN(Generative Adversarial Nets)논문 정리](https://github.com/Yudonggeun/Paper-Summary-ENG/blob/master/Summaries/GAN(Generative-Adversarial-Nets).md)





## Model.py

### Library

tensorflow, numpy, matplotlit.pyplot 라이브러리를 사용하고 텐서플로우의 예제에 있는 MNIST를 사용한다.

Epoch: 100

Batch size: 100

learning rate: 0.0002

~~~python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')


total_epochs = 100
batch_size = 100
learning_rate = 2e-4
~~~





### Generator & Discrimination & Noise

Generator: 가짜 이미지를 생성하는 모델이다. 128 사이즈의 Noise가 들어오면 1개의 Hidden Layer를 거쳐 Output Layer에서 784 사이즈로 출력해준다.

Discrimination: 이미지가 진짜인지 가짜인지를 구별해 내는 모델이다. 784 사이즈로 입력 받아 1개의 Hidden Layer를 거쳐 Output Layer에서 1개로 출력해준다.

Noise: 128 사이즈의 노이즈를 생성 모델에 주는 함수이다.

~~~python
def Generator(z):
    gw1 = tf.Variable(tf.truncated_normal(shape=[128, 256], stddev=0.1))
    gb1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
    gw2 = tf.Variable(tf.truncated_normal(shape=[256, 784], stddev=0.1))
    gb2 = tf.Variable(tf.truncated_normal(shape=[784], stddev=0.1))

    hidden = tf.nn.relu(tf.matmul(z, gw1) + gb1)
    output = tf.nn.sigmoid(tf.matmul(hidden, gw2) + gb2)

    return output


def Discrimination(x):
    dw1 = tf.Variable(tf.truncated_normal(shape=[784, 256], stddev=0.1))
    db1 = tf.Variable(tf.truncated_normal(shape=[256], stddev=0.1))
    dw2 = tf.Variable(tf.truncated_normal(shape=[256, 1], stddev=0.1))
    db2 = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1))

    hidden = tf.nn.relu(tf.matmul(x, dw1) + db1)
    output = tf.nn.sigmoid(tf.matmul(hidden, dw2) + db2)

    return output

def Random_noise(batch_size):
    return np.random.normal(size=[batch_size, 128])
~~~

