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

음정과 음길이로 나누어서 2개의 속성으로 입력된다. 'c4'는 '(c, 4)'로 나누어서 입력된다.

~~~python
# 랜덤시드 고정시키기
np.random.seed(5)

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# 데이터넷 생성 함수
def seq2dataset(seq, window_size):
    dataset_X = []
    dataset_Y = []
    for i in range(len(seq) - window_size):
        subset = seq[i:(i+window_size + 1)]

        for si in range(len(subset) - 1):
            features = code2features(subset[si])
            dataset_X.append(features)

        dataset_Y.append([code2idx[subset[window_size]]])

    return np.array(dataset_X), np.array(dataset_Y)

# 속성 변환 함수
def code2features(code):
    features = []
    features.append(code2scale[code[0]] / float(max_scale_value))
    features.append(code2length[code[1]])
    return features

code2scale = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}
code2length = {'4': 0, '8': 1}

code2idx = {'c4': 0, 'd4': 1, 'e4': 2, 'f4': 3, 'g4': 4, 'a4': 5, 'b4': 6,
            'c8': 7, 'd8': 8, 'e8': 9, 'f8': 10, 'g8': 11, 'a8': 12, 'b8': 13}

idx2code = {0: 'c4', 1: 'd4', 2: 'e4', 3: 'f4', 4: 'g4', 5: 'a4', 6: 'b4',
            7: 'c8', 8: 'd8', 9: 'e8', 10: 'f8', 11: 'g8', 12: 'a8', 13: 'b8'}

max_scale_value = 6.0

seq = ['g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'd8', 'e8', 'f8', 'g8', 'g8', 'g4',
       'g8', 'e8', 'e8', 'e8', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4',
       'd8', 'd8', 'd8', 'd8', 'd8', 'e8', 'f4', 'e8', 'e8', 'e8', 'e8', 'e8', 'f8', 'g4',
       'g8', 'e8', 'e4', 'f8', 'd8', 'd4', 'c8', 'e8', 'g8', 'g8', 'e8', 'e8', 'e4']



x_train, y_train = seq2dataset(seq, window_size=4)

# 입력 (샘플 수, 타임스텝, 특성 수)로 형태 변환
x_train = np.reshape(x_train, (50, 4, 2))

# 라벨값에 대한 one-hot encoding 수행
y_train = np_utils.to_categorical(y_train)

one_hot_vec_size = y_train.shape[1]

print('one hot encoding vector size is ', one_hot_vec_size)
~~~



## Model



~~~~python
# 모델 구성
model = Sequential()
model.add(LSTM(128, batch_input_shape=(1, 4, 2), stateful=True))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 모델 학습과정 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = LossHistory()
history.init()

num_epochs = 2000

for epoch_idx in range(num_epochs):
    print('epochs: ', str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, shuffle=False, callbacks=[history])
    model.reset_states()
~~~~

