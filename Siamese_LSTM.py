# 代码2020-8-29版本:实现了siamese LSTM,后续还需要针对论文进行改进
# 目标:复现ATC2020论文:HDDse
# 环境设置为:Pycharm 2020.2
#          tensorflow 2.1.0
#          keras 2.4.2
# 代码还未进行整理
from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import data_process as dp
import parameter
import numpy as np
from keras.callbacks import Callback
from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional
from keras.optimizers import Adam
from Eludist_loss import EluDist, contrastive_loss
from keras.callbacks import LearningRateScheduler
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import minmax_scale

import os  # in order to use code in MacOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 计算故障样本的召回率
def recall_failure(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# 创建训练网络
def create_base_network(input_shape_sl):
    # Base network to be shared (eq. to feature extraction).
    # four hidden layer
    # 根据论文中给出的,Dropout使用的比例为0.5,只在LSTM层中使用了dropout = 0.5
    _input = Input(shape=input_shape_sl)
    # add l2 regularization
    x = Bidirectional(  # not use recurrent_dropout = 0.5
        LSTM(units=16, dropout=0.5, return_sequences=True, kernel_regularizer=regularizers.l2(1e-3)),
        input_shape=input_shape_sl)(_input)
    x = Bidirectional(
        LSTM(units=32, dropout=0.5, return_sequences=True, kernel_regularizer=regularizers.l2(1e-3)))(x)
    x = Bidirectional(
        LSTM(units=64, dropout=0.5, return_sequences=True, kernel_regularizer=regularizers.l2(1e-3)))(x)
    x = Bidirectional(
        LSTM(units=128, dropout=0.5, return_sequences=False, kernel_regularizer=regularizers.l2(1e-3)))(x)
    x = Dense(128)(x)
    x = Dense(256)(x)
    return Model(_input, x)


# learning rate decay
def step_decay(epoch):
    if epoch == 0:
        K.set_value(model.optimizer.lr, 0.05)  # 论文中使用的学习率为0.1,这里使用0.05
    if epoch % 5 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)


# read data and change data shape
health_data_set = dp.data_reshape("health_disks.csv")
failure_data_set = dp.data_reshape("failure_disks.csv")

# construct training data
x_train, y_train = dp.create_pairs(health_data_set, failure_data_set)
x_train = x_train.astype('float32')  # x_train的shape是(-1,14,9),也就是有多个输入,每个输入为14*9
# print(x_train)
input_shape = x_train.shape[2:]  # 形状为14 * 9,只保留数据的输入格式

base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# processed_a和processed_a是孪生LSTM网络处理之后的结果
processed_a = base_network(input_a)  # shape为(None，256)
processed_b = base_network(input_b)

distance = EluDist()([processed_a, processed_b])

learn_rate = LearningRateScheduler(step_decay)

model = Model([input_a, input_b], distance)
adam = Adam()  # 定义优化器

model.compile(loss=contrastive_loss, optimizer=adam)  # 激活model

# 这里可以使用两个for循环替代grid search
history = model.fit([x_train[:, 0], x_train[:, 1]], y_train,  # 前面两个是训练的数据,后面tr_y是标签
                    batch_size=parameter.BATCH_SIZE,
                    epochs=parameter.N_EPOCH,
                    callbacks=[learn_rate],  # 用来调整学习率
                    verbose=1,
                    validation_split=0.3)

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend('Train', loc='upper left')
plt.show()

save_path = "数据/"
model.save(save_path + "model_train.h5")
