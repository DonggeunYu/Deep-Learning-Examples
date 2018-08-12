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

