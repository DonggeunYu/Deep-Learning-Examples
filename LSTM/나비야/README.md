# LSTM 나비야

## Summary

김태영 저자님의 '블록과 함께하는 파이썬 딥러닝 케라스'에 나오는 LSTM(순환 신경망 모델) 예제이다. '나비야'라는 노래를 학습시킬 것이며 자연어 처리에 강한 LSTM을 사용할 것이다.

LSTM의 장점은 순차적인 자료에서 규칙적인 패턴을 인식하거나 그 의미를 추론 할 수 있다. 순차적인 특성 때문에 간단한 레이어로도 다양한 형태의 모델을 구성할 수 있다.



## Package

keras, Numpy, matplotlib 사용한다.

~~~python
import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
~~~



## Input

