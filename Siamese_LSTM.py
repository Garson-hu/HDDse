# 代码2020-8-29版本:实现了siamese LSTM,后续还需要针对论文进行改进
# 目标:复现ATC2020论文:HDDse
# 环境设置为:Pycharm 2020.2
#          tensorflow 2.1.0
#          keras 2.4.2
# 代码还未进行整理
from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import data_process as dp
import tensorflow as tf
# from sklearn.metrics import classification_report
from data_process import EluDist
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# gpus = tf.compat.v1.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.compat.v1.config.experimental.set_memory_growth(gpu, True)


# 计算损失函数
def contrastive_loss(y_true, y_pred):
    margin = 1.25
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# 创建训练网络
def create_base_network(input_shape_sl):
    # Base network to be shared (eq. to feature extraction).
    # 四个隐藏层,即四层BiLSTM和一个全连接层
    # 根据论文中给出的,Dropout使用的比例为0.5,只在LSTM层中使用了dropout = 0.5
    input = Input(shape=input_shape_sl)
    # 在这一步加入了L2正则化项,但是目前不知道这种做法是否正确
    x = Bidirectional(
        LSTM(units=16, dropout=0.5, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        input_shape=input_shape_sl)(input)
    x = Bidirectional(
        LSTM(units=32, dropout=0.5, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-3)))(x)
    x = Bidirectional(
        LSTM(units=64, dropout=0.5, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-3)))(x)
    x = Bidirectional(
        LSTM(units=128, dropout=0.5, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-3)))(x)
    x = Dense(128)(x)
    x = Dense(256)(x)
    return Model(input, x)


# 学习率衰减
def step_decay(epoch):
    if epoch == 0:
        K.set_value(model.optimizer.lr, 0.05)  # 论文中使用的学习率为0.1,这里使用0.05
    if epoch % 5 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)


def compute_accuracy(y_true, y_pred):  # 已弃用本函数
    # Compute classification accuracy with a fixed threshold on distances.
    # ravel()函数主要是将多维数组转换为一维数组,然后筛选出预测误差比较接近的
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


# 读取数据并改变shape
health_data_set = dp.data_reshape("health_disks.csv")
failure_data_set = dp.data_reshape("failure_disks.csv")

# 创建训练数据
x_train, y_train = dp.create_pairs(health_data_set, failure_data_set)
x_train = x_train.astype('float32')  # x_train的shape是(-1,14,9),也就是有多个输入,每个输入为14*9
# print(x_train)
input_shape = x_train.shape[2:]  # 形状为14 * 9,只保留数据的输入格式

n_epochs = 100

base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# processed_a和processed_a是孪生LSTM网络处理之后的结果
processed_a = base_network(input_a)  # shape为(None.256)
processed_b = base_network(input_b)

distance = EluDist()([processed_a, processed_b])

learn_rate = LearningRateScheduler(step_decay)

model = Model([input_a, input_b], distance)
adam = Adam()  # 定义优化器

model.compile(loss=contrastive_loss, optimizer=adam)  # 激活model

history = model.fit([x_train[:, 0], x_train[:, 1]], y_train,  # 前面两个是训练的数据,后面tr_y是标签
                    batch_size=32,
                    epochs=n_epochs,
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

save_path = "数据集/"
model.save(save_path + "model_train.h5")
