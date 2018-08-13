# MNIST GAN

## Summary

GAN을 이용하여 이미지를 생성해내는 것이 목적이다. GAN은 경찰과 지폐위조범은 개념으로 생각하면 쉽다. 경찰은 위조된 지폐를 구별하기 위해 노력하면서 구별하는 실력이 향상되고 지폐위조범은 경찰이 구별하기 힘든 지폐를 만들기 위해 노력하면서 가짜 지폐를 만들어 내는 실력이 향상될 것이다. 이 처럼 두 모델의 경쟁을 통해 서로의 모델이 학습을 하게된다.

참고 : [GAN (Generative Adversarial Nets) 논문 정리](https://github.com/Yudonggeun/Paper-Summary-ENG/blob/master/Summaries/GAN(Generative-Adversarial-Nets).md)



<br/>



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
~~~





### Input and Loss Optimizer

Discrimination()에 가짜 이미지를 넣으면 0이 출력되고 진짜 이미지를 넣으면 1이 출력되야함. g_loss와 d_loss가 0에 가까워지는 과정이 학습이다. 그리고 가장 중요한 것은Generator()와 Discrimination()이 학습을 할 때는 서로를 건드리지 않고 별개로 Train된다.

~~~python
X = tf.placeholder(tf.float32, shape=[None, 784]) # 진짜 이미지가 들어감.
Z = tf.placeholder(tf.float32, shape=[None, 128]) # Noise가 들어감.

fake_x = Generator(Z) # Generator가 만든 가짜 이미지.

result_of_fake = Discrimination(fake_x) # 가짜 이미지를 판별기에 넣음.
result_of_real = Discrimination(X) # 진짜 이미지를 판별기에 넣음.

g_loss = -tf.reduce_mean(tf.log(result_of_fake)) # 가짜 이미지는 0을 출력해야함.
d_loss = -tf.reduce_mean(tf.log(result_of_real) + tf.log(1- result_of_fake)) # 진짜 이미지와 1 - 가짜 이미지의 결과값의 합이 0이 되야함.

g_train = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=[gw1, gb1, gw2, gb2]) # g_loss를 0에 가깝게 Minimize한다.
d_train = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=[dw1, db1, dw2, db2]) # d_loss를 0에 가깝게 Minimize한다.
~~~





### Session

이 Session에서는 값을 설정한 Epoch 수만큼 설정한 batch_size를 넣어 학습시킨다. 그리고 Epoch이 10의 배수 일 때마다 학습 정도를 확인 하기 위해서 Generator에  Noise를 넣어 만들어진 가짜 이미지 10개를 1개의 이미지로 합하여 저장한다.

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batchs = int(train_x.shape[0] / batch_size)

    for epoch in range(total_epochs): # 설정한 Epoch 수만큼 반복.

        for batch in range(total_batchs):
            batch_x = train_x[batch * batch_size: (batch + 1) * batch_size]
            noise = Random_noise(batch_size) # 학습을 시킬때 마다 랜덤으로 Noise가 달라짐.

            sess.run(g_train, feed_dict={Z: noise}) # Noise로 Gernerator 학습.
            sess.run(d_train, feed_dict={X: batch_x, Z: noise}) # 진짜 이미지와 가짜 이미지로 Discrimination 학습

            gl, dl = sess.run([g_loss, d_loss], feed_dict={X: batch_x, Z: noise}) # g_loss, d_loss를 구함.

        print('=======Epoch: ', epoch, '=======================================')
        print('Generator: ', gl)
        print('Discrimination: ', dl)

        if epoch % 10 == 0: 3 Epoch 10 마다 이미지로 저장.
            sample_noise = Random_noise(10)
            generated = sess.run(fake_x, feed_dict={Z: sample_noise})
            fig, ax = plt.subplots(1, 10, figsize=(10, 1))
            for i in range(10):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(generated[i], (28, 28))) # 이미지를 28x28로 reshape한다.

            plt.savefig('goblin-gan-generated/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

    print('End')
```



<br/>



## Full Sources

```python
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
```



<br/>



## Result

텐서플로우를 접하고 딥러닝에 대해 배우면서 처음으로 nan을 출력하면서 overshooting이 발생하는 것을 경험하였다.



### Log

~~~
=======Epoch:  0 =======================================
Generator:  3.0925958
Discrimination:  0.1050963
=======Epoch:  1 =======================================
Generator:  3.6625414
Discrimination:  0.053676713
=======Epoch:  2 =======================================
Generator:  4.4238315
Discrimination:  0.050269403
=======Epoch:  3 =======================================
Generator:  5.044856
Discrimination:  0.01532165
=======Epoch:  4 =======================================
Generator:  4.01807
Discrimination:  0.051850792
=======Epoch:  5 =======================================
Generator:  3.6015613
Discrimination:  0.088804156
=======Epoch:  6 =======================================
Generator:  4.2444305
Discrimination:  0.07612028
=======Epoch:  7 =======================================
Generator:  4.479038
Discrimination:  0.04592362
=======Epoch:  8 =======================================
Generator:  4.307972
Discrimination:  0.08963592
=======Epoch:  9 =======================================
Generator:  3.8229687
Discrimination:  0.06827531
=======Epoch:  10 =======================================
Generator:  3.1119785
Discrimination:  0.16969046
=======Epoch:  11 =======================================
Generator:  3.642295
Discrimination:  0.18712962
=======Epoch:  12 =======================================
Generator:  3.9087908
Discrimination:  0.22462493
=======Epoch:  13 =======================================
Generator:  3.7366276
Discrimination:  0.23505901
=======Epoch:  14 =======================================
Generator:  3.2886856
Discrimination:  0.22315234
=======Epoch:  15 =======================================
Generator:  3.1806686
Discrimination:  0.29730093
=======Epoch:  16 =======================================
Generator:  2.90413
Discrimination:  0.39224696
=======Epoch:  17 =======================================
Generator:  3.3989782
Discrimination:  0.31891015
=======Epoch:  18 =======================================
Generator:  3.4603133
Discrimination:  0.3561284
=======Epoch:  19 =======================================
Generator:  3.2232695
Discrimination:  0.47254577
=======Epoch:  20 =======================================
Generator:  3.620379
Discrimination:  0.3764592
=======Epoch:  21 =======================================
Generator:  3.4605215
Discrimination:  0.31752387
=======Epoch:  22 =======================================
Generator:  2.8774686
Discrimination:  0.3179536
=======Epoch:  23 =======================================
Generator:  2.9325361
Discrimination:  0.2803762
=======Epoch:  24 =======================================
Generator:  3.059424
Discrimination:  0.29181257
=======Epoch:  25 =======================================
Generator:  2.6317751
Discrimination:  0.38484073
=======Epoch:  26 =======================================
Generator:  3.1722507
Discrimination:  0.3668856
=======Epoch:  27 =======================================
Generator:  3.1378555
Discrimination:  0.27861735
=======Epoch:  28 =======================================
Generator:  3.0006487
Discrimination:  0.37954536
=======Epoch:  29 =======================================
Generator:  2.8202057
Discrimination:  0.34367067
=======Epoch:  30 =======================================
Generator:  3.1866696
Discrimination:  0.2482017
=======Epoch:  31 =======================================
Generator:  2.8977385
Discrimination:  0.36496058
=======Epoch:  32 =======================================
Generator:  2.894899
Discrimination:  0.2910026
=======Epoch:  33 =======================================
Generator:  2.704748
Discrimination:  0.3355739
=======Epoch:  34 =======================================
Generator:  2.636716
Discrimination:  0.35199115
=======Epoch:  35 =======================================
Generator:  2.7484865
Discrimination:  0.40564334
=======Epoch:  36 =======================================
Generator:  2.8518484
Discrimination:  0.27292228
=======Epoch:  37 =======================================
Generator:  2.647268
Discrimination:  0.39419624
=======Epoch:  38 =======================================
Generator:  3.0659692
Discrimination:  0.24029393
=======Epoch:  39 =======================================
Generator:  3.1273642
Discrimination:  0.2592874
=======Epoch:  40 =======================================
Generator:  2.4939263
Discrimination:  0.38065526
=======Epoch:  41 =======================================
Generator:  2.8060935
Discrimination:  0.30365655
=======Epoch:  42 =======================================
Generator:  2.728335
Discrimination:  0.39338648
=======Epoch:  43 =======================================
Generator:  2.4866095
Discrimination:  0.30964306
=======Epoch:  44 =======================================
Generator:  2.8642864
Discrimination:  0.3252285
=======Epoch:  45 =======================================
Generator:  2.843691
Discrimination:  0.36758342
=======Epoch:  46 =======================================
Generator:  2.3694363
Discrimination:  0.37503365
=======Epoch:  47 =======================================
Generator:  2.7595747
Discrimination:  0.29100373
=======Epoch:  48 =======================================
Generator:  2.8050408
Discrimination:  0.38433638
=======Epoch:  49 =======================================
Generator:  2.527235
Discrimination:  0.29517838
=======Epoch:  50 =======================================
Generator:  2.291405
Discrimination:  0.4204319
=======Epoch:  51 =======================================
Generator:  2.3062928
Discrimination:  0.46655792
=======Epoch:  52 =======================================
Generator:  2.3559678
Discrimination:  0.36762157
=======Epoch:  53 =======================================
Generator:  2.4650552
Discrimination:  0.36139405
=======Epoch:  54 =======================================
Generator:  2.4466891
Discrimination:  0.42787617
=======Epoch:  55 =======================================
Generator:  2.3942199
Discrimination:  0.44224817
=======Epoch:  56 =======================================
Generator:  2.1619282
Discrimination:  0.419592
=======Epoch:  57 =======================================
Generator:  2.3826258
Discrimination:  0.40382725
=======Epoch:  58 =======================================
Generator:  2.22319
Discrimination:  0.4371454
=======Epoch:  59 =======================================
Generator:  2.1551576
Discrimination:  0.47373638
=======Epoch:  60 =======================================
Generator:  2.2968078
Discrimination:  0.40650985
=======Epoch:  61 =======================================
Generator:  2.1384006
Discrimination:  0.4918656
=======Epoch:  62 =======================================
Generator:  2.0744262
Discrimination:  0.40453354
=======Epoch:  63 =======================================
Generator:  2.0631433
Discrimination:  0.46594226
=======Epoch:  64 =======================================
Generator:  2.2963178
Discrimination:  0.41596398
=======Epoch:  65 =======================================
Generator:  2.247974
Discrimination:  0.3783644
=======Epoch:  66 =======================================
Generator:  2.0648394
Discrimination:  0.5165674
=======Epoch:  67 =======================================
Generator:  2.1401014
Discrimination:  0.41112083
=======Epoch:  68 =======================================
Generator:  2.0866308
Discrimination:  0.47367024
=======Epoch:  69 =======================================
Generator:  2.1179366
Discrimination:  0.44822773
=======Epoch:  70 =======================================
Generator:  2.1258092
Discrimination:  0.4458595
=======Epoch:  71 =======================================
Generator:  2.1029596
Discrimination:  0.4680241
=======Epoch:  72 =======================================
Generator:  2.121829
Discrimination:  0.48157722
=======Epoch:  73 =======================================
Generator:  2.146594
Discrimination:  0.44197506
=======Epoch:  74 =======================================
Generator:  2.2030702
Discrimination:  0.51029634
=======Epoch:  75 =======================================
Generator:  2.17894
Discrimination:  0.36927018
=======Epoch:  76 =======================================
Generator:  1.8641808
Discrimination:  0.5088781
=======Epoch:  77 =======================================
Generator:  2.1434886
Discrimination:  0.4273092
=======Epoch:  78 =======================================
Generator:  2.1873968
Discrimination:  0.43277457
=======Epoch:  79 =======================================
Generator:  2.2243586
Discrimination:  0.4703676
=======Epoch:  80 =======================================
Generator:  2.4009738
Discrimination:  0.3888484
=======Epoch:  81 =======================================
Generator:  1.9651268
Discrimination:  0.47994152
=======Epoch:  82 =======================================
Generator:  2.1641757
Discrimination:  0.45268852
=======Epoch:  83 =======================================
Generator:  2.172931
Discrimination:  0.45162123
=======Epoch:  84 =======================================
Generator:  2.2285898
Discrimination:  0.35669038
=======Epoch:  85 =======================================
Generator:  2.037091
Discrimination:  0.38922527
=======Epoch:  86 =======================================
Generator:  1.9083421
Discrimination:  0.49164864
=======Epoch:  87 =======================================
Generator:  2.250847
Discrimination:  0.39609635
=======Epoch:  88 =======================================
Generator:  2.0481327
Discrimination:  0.47458243
=======Epoch:  89 =======================================
Generator:  2.2851644
Discrimination:  0.39447322
=======Epoch:  90 =======================================
Generator:  2.2613447
Discrimination:  0.413838
=======Epoch:  91 =======================================
Generator:  2.326611
Discrimination:  0.39614686
=======Epoch:  92 =======================================
Generator:  1.8975428
Discrimination:  0.50665045
=======Epoch:  93 =======================================
Generator:  2.3789105
Discrimination:  0.3676082
=======Epoch:  94 =======================================
Generator:  2.2206838
Discrimination:  0.4015464
=======Epoch:  95 =======================================
Generator:  2.4073336
Discrimination:  0.3831463
=======Epoch:  96 =======================================
Generator:  2.241024
Discrimination:  0.41167778
=======Epoch:  97 =======================================
Generator:  2.4687376
Discrimination:  0.3754435
=======Epoch:  98 =======================================
Generator:  2.265462
Discrimination:  0.34332407
=======Epoch:  99 =======================================
Generator:  2.4980216
Discrimination:  0.38974658
=======Epoch:  100 =======================================
Generator:  2.13206
Discrimination:  0.51424176
=======Epoch:  101 =======================================
Generator:  2.1126409
Discrimination:  0.41336286
=======Epoch:  102 =======================================
Generator:  2.135554
Discrimination:  0.43298066
=======Epoch:  103 =======================================
Generator:  2.022551
Discrimination:  0.52980554
=======Epoch:  104 =======================================
Generator:  2.1288147
Discrimination:  0.39613238
=======Epoch:  105 =======================================
Generator:  2.2320995
Discrimination:  0.4061155
=======Epoch:  106 =======================================
Generator:  2.213645
Discrimination:  0.43103734
=======Epoch:  107 =======================================
Generator:  2.197738
Discrimination:  0.42626354
=======Epoch:  108 =======================================
Generator:  2.343012
Discrimination:  0.4300716
=======Epoch:  109 =======================================
Generator:  2.4275787
Discrimination:  0.30320892
=======Epoch:  110 =======================================
Generator:  2.5240185
Discrimination:  0.34501296
=======Epoch:  111 =======================================
Generator:  2.1535196
Discrimination:  0.41874433
=======Epoch:  112 =======================================
Generator:  2.3047485
Discrimination:  0.39607337
=======Epoch:  113 =======================================
Generator:  2.0555756
Discrimination:  0.44649753
=======Epoch:  114 =======================================
Generator:  2.4397812
Discrimination:  0.3336724
=======Epoch:  115 =======================================
Generator:  2.1439738
Discrimination:  0.43037423
=======Epoch:  116 =======================================
Generator:  2.1781511
Discrimination:  0.38974518
=======Epoch:  117 =======================================
Generator:  2.3524888
Discrimination:  0.45379552
=======Epoch:  118 =======================================
Generator:  1.9886721
Discrimination:  0.47378474
=======Epoch:  119 =======================================
Generator:  2.44724
Discrimination:  0.3366022
=======Epoch:  120 =======================================
Generator:  2.5549805
Discrimination:  0.38405824
=======Epoch:  121 =======================================
Generator:  2.321571
Discrimination:  0.4159838
=======Epoch:  122 =======================================
Generator:  2.3765824
Discrimination:  0.41593227
=======Epoch:  123 =======================================
Generator:  2.4741518
Discrimination:  0.3656885
=======Epoch:  124 =======================================
Generator:  2.334412
Discrimination:  0.38853145
=======Epoch:  125 =======================================
Generator:  2.1127338
Discrimination:  0.44116306
=======Epoch:  126 =======================================
Generator:  2.6718142
Discrimination:  0.37638202
=======Epoch:  127 =======================================
Generator:  2.5163796
Discrimination:  0.33527106
=======Epoch:  128 =======================================
Generator:  2.1863248
Discrimination:  0.38444397
=======Epoch:  129 =======================================
Generator:  2.2747843
Discrimination:  0.29815197
=======Epoch:  130 =======================================
Generator:  2.28432
Discrimination:  0.4066985
=======Epoch:  131 =======================================
Generator:  2.3289595
Discrimination:  0.3422924
=======Epoch:  132 =======================================
Generator:  2.5548124
Discrimination:  0.37221637
=======Epoch:  133 =======================================
Generator:  2.2254388
Discrimination:  0.39078885
=======Epoch:  134 =======================================
Generator:  2.2387369
Discrimination:  0.39773384
=======Epoch:  135 =======================================
Generator:  2.412875
Discrimination:  0.32951018
=======Epoch:  136 =======================================
Generator:  2.572439
Discrimination:  0.34272656
=======Epoch:  137 =======================================
Generator:  2.4249117
Discrimination:  0.37517196
=======Epoch:  138 =======================================
Generator:  2.5203261
Discrimination:  0.32005855
=======Epoch:  139 =======================================
Generator:  2.3604226
Discrimination:  0.36953166
=======Epoch:  140 =======================================
Generator:  2.4330199
Discrimination:  0.38688496
=======Epoch:  141 =======================================
Generator:  2.2645216
Discrimination:  0.43226188
=======Epoch:  142 =======================================
Generator:  2.434627
Discrimination:  0.30889466
=======Epoch:  143 =======================================
Generator:  2.3583066
Discrimination:  0.3470979
=======Epoch:  144 =======================================
Generator:  2.383791
Discrimination:  0.42594978
=======Epoch:  145 =======================================
Generator:  2.568884
Discrimination:  0.27567872
=======Epoch:  146 =======================================
Generator:  2.264855
Discrimination:  0.40707862
=======Epoch:  147 =======================================
Generator:  2.4900007
Discrimination:  0.3724792
=======Epoch:  148 =======================================
Generator:  2.44426
Discrimination:  0.33292207
=======Epoch:  149 =======================================
Generator:  2.605241
Discrimination:  0.3136946
=======Epoch:  150 =======================================
Generator:  2.3961265
Discrimination:  0.3381552
=======Epoch:  151 =======================================
Generator:  2.1596813
Discrimination:  0.40510038
=======Epoch:  152 =======================================
Generator:  2.3696976
Discrimination:  0.39109558
=======Epoch:  153 =======================================
Generator:  2.4915373
Discrimination:  0.39817488
=======Epoch:  154 =======================================
Generator:  2.2942045
Discrimination:  0.31553984
=======Epoch:  155 =======================================
Generator:  2.528287
Discrimination:  0.33341193
=======Epoch:  156 =======================================
Generator:  2.2909105
Discrimination:  0.3288333
=======Epoch:  157 =======================================
Generator:  2.298271
Discrimination:  0.42839175
=======Epoch:  158 =======================================
Generator:  2.5110319
Discrimination:  0.32267883
=======Epoch:  159 =======================================
Generator:  2.1285517
Discrimination:  0.40650514
=======Epoch:  160 =======================================
Generator:  2.7600553
Discrimination:  0.35397997
=======Epoch:  161 =======================================
Generator:  2.6346247
Discrimination:  0.34170204
=======Epoch:  162 =======================================
Generator:  2.4196386
Discrimination:  0.3125661
=======Epoch:  163 =======================================
Generator:  2.7932136
Discrimination:  0.39539662
=======Epoch:  164 =======================================
Generator:  2.2437835
Discrimination:  0.44217572
=======Epoch:  165 =======================================
Generator:  2.3742576
Discrimination:  0.43778905
=======Epoch:  166 =======================================
Generator:  2.3432236
Discrimination:  0.33637848
=======Epoch:  167 =======================================
Generator:  2.7562933
Discrimination:  0.21748774
=======Epoch:  168 =======================================
Generator:  2.432252
Discrimination:  0.3713739
=======Epoch:  169 =======================================
Generator:  2.7630668
Discrimination:  0.31534854
=======Epoch:  170 =======================================
Generator:  2.6263359
Discrimination:  0.38502192
=======Epoch:  171 =======================================
Generator:  2.3422463
Discrimination:  0.39934373
=======Epoch:  172 =======================================
Generator:  2.5371926
Discrimination:  0.36540544
=======Epoch:  173 =======================================
Generator:  2.563651
Discrimination:  0.29330802
=======Epoch:  174 =======================================
Generator:  2.67241
Discrimination:  0.3100065
=======Epoch:  175 =======================================
Generator:  2.5789306
Discrimination:  0.3101134
=======Epoch:  176 =======================================
Generator:  2.4114535
Discrimination:  0.3783127
=======Epoch:  177 =======================================
Generator:  2.3562188
Discrimination:  0.36295903
=======Epoch:  178 =======================================
Generator:  2.4928954
Discrimination:  0.32291806
=======Epoch:  179 =======================================
Generator:  2.2611997
Discrimination:  0.3966045
=======Epoch:  180 =======================================
Generator:  2.5661063
Discrimination:  0.3327717
=======Epoch:  181 =======================================
Generator:  2.6642272
Discrimination:  0.28507307
=======Epoch:  182 =======================================
Generator:  2.6574655
Discrimination:  0.2861865
=======Epoch:  183 =======================================
Generator:  2.4136443
Discrimination:  0.4042819
=======Epoch:  184 =======================================
Generator:  2.8260179
Discrimination:  0.28278074
=======Epoch:  185 =======================================
Generator:  2.5107763
Discrimination:  0.3142423
=======Epoch:  186 =======================================
Generator:  2.453369
Discrimination:  0.3066168
=======Epoch:  187 =======================================
Generator:  2.1322396
Discrimination:  0.47132266
=======Epoch:  188 =======================================
Generator:  2.361123
Discrimination:  0.40100583
=======Epoch:  189 =======================================
Generator:  2.649889
Discrimination:  0.29530627
=======Epoch:  190 =======================================
Generator:  2.6277878
Discrimination:  0.3589235
=======Epoch:  191 =======================================
Generator:  2.5252388
Discrimination:  0.4863189
=======Epoch:  192 =======================================
Generator:  2.49534
Discrimination:  0.34440425
=======Epoch:  193 =======================================
Generator:  2.5472224
Discrimination:  0.28877208
=======Epoch:  194 =======================================
Generator:  2.4682808
Discrimination:  0.35827857
=======Epoch:  195 =======================================
Generator:  2.3296497
Discrimination:  0.34576556
=======Epoch:  196 =======================================
Generator:  2.6268847
Discrimination:  0.43061557
=======Epoch:  197 =======================================
Generator:  2.35575
Discrimination:  0.37526476
=======Epoch:  198 =======================================
Generator:  2.5076172
Discrimination:  0.32035422
=======Epoch:  199 =======================================
Generator:  2.4209213
Discrimination:  0.35692245
=======Epoch:  200 =======================================
Generator:  2.6552308
Discrimination:  0.30624154
=======Epoch:  201 =======================================
Generator:  2.5546763
Discrimination:  0.32854897
=======Epoch:  202 =======================================
Generator:  2.3448184
Discrimination:  0.35452583
=======Epoch:  203 =======================================
Generator:  2.364304
Discrimination:  0.38807136
=======Epoch:  204 =======================================
Generator:  2.4764502
Discrimination:  0.31242743
=======Epoch:  205 =======================================
Generator:  2.580538
Discrimination:  0.33858544
=======Epoch:  206 =======================================
Generator:  2.6476285
Discrimination:  0.37321472
=======Epoch:  207 =======================================
Generator:  2.642636
Discrimination:  0.36831322
=======Epoch:  208 =======================================
Generator:  2.6533766
Discrimination:  0.33412308
=======Epoch:  209 =======================================
Generator:  2.9812996
Discrimination:  0.29355937
=======Epoch:  210 =======================================
Generator:  2.8068478
Discrimination:  0.3188842
=======Epoch:  211 =======================================
Generator:  2.7919464
Discrimination:  0.3015813
=======Epoch:  212 =======================================
Generator:  2.8181849
Discrimination:  0.2779876
=======Epoch:  213 =======================================
Generator:  2.687631
Discrimination:  0.2963673
=======Epoch:  214 =======================================
Generator:  2.5401144
Discrimination:  0.2916065
=======Epoch:  215 =======================================
Generator:  2.4311152
Discrimination:  0.3078477
=======Epoch:  216 =======================================
Generator:  2.4220815
Discrimination:  0.28410146
=======Epoch:  217 =======================================
Generator:  2.929306
Discrimination:  0.26920566
=======Epoch:  218 =======================================
Generator:  2.6831617
Discrimination:  0.30941623
=======Epoch:  219 =======================================
Generator:  2.6619816
Discrimination:  0.37796324
=======Epoch:  220 =======================================
Generator:  2.5334663
Discrimination:  0.30370486
=======Epoch:  221 =======================================
Generator:  2.6231441
Discrimination:  0.344699
=======Epoch:  222 =======================================
Generator:  2.7680914
Discrimination:  0.2787267
=======Epoch:  223 =======================================
Generator:  2.546308
Discrimination:  0.33373028
=======Epoch:  224 =======================================
Generator:  2.4399314
Discrimination:  0.31327355
=======Epoch:  225 =======================================
Generator:  2.94629
Discrimination:  0.25763258
=======Epoch:  226 =======================================
Generator:  2.519133
Discrimination:  0.2862872
=======Epoch:  227 =======================================
Generator:  2.4169047
Discrimination:  0.32756603
=======Epoch:  228 =======================================
Generator:  2.5591362
Discrimination:  0.30961627
=======Epoch:  229 =======================================
Generator:  2.7177017
Discrimination:  0.29452205
=======Epoch:  230 =======================================
Generator:  2.7101278
Discrimination:  0.28765708
=======Epoch:  231 =======================================
Generator:  2.4438198
Discrimination:  0.34397775
=======Epoch:  232 =======================================
Generator:  2.631759
Discrimination:  0.38064852
=======Epoch:  233 =======================================
Generator:  2.8147535
Discrimination:  0.2875987
=======Epoch:  234 =======================================
Generator:  2.5263643
Discrimination:  0.3287905
=======Epoch:  235 =======================================
Generator:  2.4116263
Discrimination:  0.3691255
=======Epoch:  236 =======================================
Generator:  2.5485876
Discrimination:  0.3319469
=======Epoch:  237 =======================================
Generator:  2.8194227
Discrimination:  0.2203684
=======Epoch:  238 =======================================
Generator:  2.605083
Discrimination:  0.29914016
=======Epoch:  239 =======================================
Generator:  2.4260273
Discrimination:  0.29823756
=======Epoch:  240 =======================================
Generator:  2.6868474
Discrimination:  0.33586147
=======Epoch:  241 =======================================
Generator:  2.6346087
Discrimination:  0.3292735
=======Epoch:  242 =======================================
Generator:  2.9760993
Discrimination:  0.28349683
=======Epoch:  243 =======================================
Generator:  2.5854514
Discrimination:  0.36600402
=======Epoch:  244 =======================================
Generator:  2.5148041
Discrimination:  0.3449806
=======Epoch:  245 =======================================
Generator:  2.9033995
Discrimination:  0.266802
=======Epoch:  246 =======================================
Generator:  2.8348413
Discrimination:  0.30950898
=======Epoch:  247 =======================================
Generator:  2.6788764
Discrimination:  0.29273206
=======Epoch:  248 =======================================
Generator:  2.688797
Discrimination:  0.2836097
=======Epoch:  249 =======================================
Generator:  2.8236542
Discrimination:  0.32218432
=======Epoch:  250 =======================================
Generator:  2.7232425
Discrimination:  0.28197205
=======Epoch:  251 =======================================
Generator:  2.6193054
Discrimination:  0.3352263
=======Epoch:  252 =======================================
Generator:  2.6632676
Discrimination:  0.3388756
=======Epoch:  253 =======================================
Generator:  2.8907952
Discrimination:  0.24128586
=======Epoch:  254 =======================================
Generator:  2.5839393
Discrimination:  0.34169137
=======Epoch:  255 =======================================
Generator:  2.7442167
Discrimination:  0.3022849
=======Epoch:  256 =======================================
Generator:  2.696444
Discrimination:  0.2756491
=======Epoch:  257 =======================================
Generator:  3.0397906
Discrimination:  0.31959265
=======Epoch:  258 =======================================
Generator:  2.835224
Discrimination:  0.26500562
=======Epoch:  259 =======================================
Generator:  2.6028357
Discrimination:  0.30419707
=======Epoch:  260 =======================================
Generator:  2.8017642
Discrimination:  0.25802755
=======Epoch:  261 =======================================
Generator:  2.7526188
Discrimination:  0.3042086
=======Epoch:  262 =======================================
Generator:  2.513451
Discrimination:  0.29171246
=======Epoch:  263 =======================================
Generator:  2.7141519
Discrimination:  0.34746093
=======Epoch:  264 =======================================
Generator:  2.4836016
Discrimination:  0.32197642
=======Epoch:  265 =======================================
Generator:  2.5094633
Discrimination:  0.33812663
=======Epoch:  266 =======================================
Generator:  2.7555094
Discrimination:  0.26767656
=======Epoch:  267 =======================================
Generator:  3.0023293
Discrimination:  0.25023967
=======Epoch:  268 =======================================
Generator:  3.0306625
Discrimination:  0.27780977
=======Epoch:  269 =======================================
Generator:  2.5909646
Discrimination:  0.2995811
=======Epoch:  270 =======================================
Generator:  2.899701
Discrimination:  0.32927755
=======Epoch:  271 =======================================
Generator:  2.8314304
Discrimination:  0.3058872
=======Epoch:  272 =======================================
Generator:  2.5302107
Discrimination:  0.3657488
=======Epoch:  273 =======================================
Generator:  2.9361215
Discrimination:  0.26095265
=======Epoch:  274 =======================================
Generator:  2.7210867
Discrimination:  0.27639306
=======Epoch:  275 =======================================
Generator:  3.0596704
Discrimination:  0.282941
=======Epoch:  276 =======================================
Generator:  2.8102312
Discrimination:  0.32120186
=======Epoch:  277 =======================================
Generator:  3.3136806
Discrimination:  0.21916242
=======Epoch:  278 =======================================
Generator:  2.7385912
Discrimination:  0.36365378
=======Epoch:  279 =======================================
Generator:  2.4687285
Discrimination:  0.3680612
=======Epoch:  280 =======================================
Generator:  2.7635272
Discrimination:  0.3149047
=======Epoch:  281 =======================================
Generator:  2.7535107
Discrimination:  0.2914606
=======Epoch:  282 =======================================
Generator:  2.8251688
Discrimination:  0.2509626
=======Epoch:  283 =======================================
Generator:  2.824462
Discrimination:  0.27187616
=======Epoch:  284 =======================================
Generator:  2.6157706
Discrimination:  0.3245717
=======Epoch:  285 =======================================
Generator:  2.535302
Discrimination:  0.33781773
=======Epoch:  286 =======================================
Generator:  nan
Discrimination:  nan
~~~





### Images



#### Epoch 0

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/000.png)

#### Epoch 10

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/010.png)



#### Epoch 20

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/020.png)



#### Epoch 30

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/030.png)



#### Epoch 40

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/040.png)



#### Epoch 50

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/050.png)



#### Epoch 60

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/060.png)



#### Epoch 70

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/070.png)



#### Epoch 80

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/080.png)



#### Epoch 90

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/090.png)



#### Epoch 100

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/100.png)



#### Epoch 110

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/110.png)



#### Epoch 120

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/120.png)



#### Epoch 130

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/130.png)



#### Epoch 140

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/140.png)



#### Epoch 150

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/150.png)



#### Epoch 160

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/160.png)



#### Epoch 170

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/170.png)



#### Epoch 180

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/180.png)



#### Epoch 190

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/190.png)



#### Epoch 200

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/200.png)



#### Epoch 210

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/210.png)



#### Epoch 220

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/220.png)



#### Epoch 230

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/230.png)



#### Epoch 240

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/240.png)



#### Epoch 250

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/250.png)



#### Epoch 260

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/260.png)



#### Epoch 270

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/270.png)



#### Epoch 280

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/280.png)



#### Epoch 290

![](https://raw.githubusercontent.com/Yudonggeun/Deep-Learning-Projects/master/GAN/MNIST/goblin-gan-generated/290.png)

