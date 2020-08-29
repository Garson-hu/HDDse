#!/opt/anaconda3/envs/tensor/bin/python
# -*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
import random
import math
import data_process as dp
from data_process import EluDist
# import tensorflow as tf
import keras
import pickle
# from keras.models import Model
# from keras.layers import Input, Dense, Dropout, LSTM, Layer, Bidirectional
# from keras.optimizers import Adam
from keras import backend as K
# from tensorflow.keras.callbacks import LearningRateScheduler

# data_path = "C:/Users/guang/Desktop/强化学习/HUAWEI/数据集/"

# def data_reshape(filename):
#     data_set = np.loadtxt(data_path + filename,delimiter=",",max_rows = 11200,skiprows=0)
#     data_set = data_set.astype('float32')
#     data_set = data_set.reshape(-1,14,9)
#     return data_set #shape为(-1,14,9)


# health_data_set = data_reshape("health_disk.csv")

# pairs = []
# labels = []

# for i in range(len(health_data_set) - 1):
#         j = i + 1
#         for j in range(len(health_data_set)):
#             pairs += [[health_data_set[i],health_data_set[j]]]
#             labels += [1]

# print(pairs)

# #print(data)
# #data.to_csv("C:/Users/guang/Desktop/强化学习/HUAWEI/数据集/failure_disk_1.csv",sep = ',',header = None)
##########################################################################
#这段主要是确定如何取pair
# data_1 = [[[1,2,3,4],[1,3,4,5]],[[1,1,1,1],[1,7,6,6]]]
# data_2 = [[[2,1,3,4],[2,3,21,2]]]
# print(len(data_1),len(data_2))
# length = len(data_1) if len(data_1) >= len(data_2) else len(data_2)
# pairs = []
# for i in range(length):
#     pairs += [[data_1[i],data_2]]
# pairs = np.array(pairs)
# print(pairs)
# print('###############')
# print(pairs[:,0])
# print('###############')
# print(pairs[:,1])
# 0: [[[1, 2, 3, 4], [1, 3, 4, 5]], [[[2, 1, 3, 4], [2, 3, 21, 2]]]]  
# 1: [[[1, 1, 1, 1], [1, 7, 6, 6]], [[[2, 1, 3, 4], [2, 3, 21, 2]]]]

###########################################################################
#这段主要是用来确定如何取数据
# dates = pd.date_range('20200712',periods = 6)
# df = pd.DataFrame(np.arange(24).reshape((6,4)), index = dates,columns = ['A','B','C','D'])
# print(df)
# #df = df.iloc[:,1:]#不读第一列数据
# print(df[df.iloc[:,0] == 4])#得到第一列数据中值为4的所有数据

######################################################

# data_path_dm = "HDDse_Siamese_LSTM/数据集/带模型数据/WDC/process/"

# #首先从文件中读取数据跟模型号
# #读取测试数据的模型号
# test_data_model = pd.read_csv(data_path_dm + "test_data_only_model.csv",header = None)
# test_data_model = np.array(test_data_model).reshape(-1,14)
# #读取模型号对应的数据
# test_data_only = np.loadtxt(data_path_dm + "test_data_only_data.csv",delimiter = ",")
# test_data_only = test_data_only.reshape(-1,14,9)

# #读取health的训练数据带模型与不带模型
# train_data_health = pd.read_csv(data_path_dm + "health_disks_with_model.csv")
# train_data_health = pd.DataFrame(train_data_health)
# train_data_health_not = np.loadtxt(data_path_dm + "health_disks.csv",delimiter = ",")
# #读取failure的训练数据
# train_data_failure = pd.read_csv(data_path_dm + "failure_disks_with_model.csv")

# for i in range(len(test_data_only)):
#     model_name = test_data_model[i][0]
#     #train_data_health = train_data_health[train_data_health["model"].isin(model_name)]
#     index = train_data_health[train_data_health.model == model_name].index.tolist()  #得到带模型数据的所在列
#     train_data_health_tmp =  np.array(train_data_health_not[index]).reshape(-1,14,9)#得到不带模型号的训练数据
#     print(len(train_data_health_tmp),len(index))
#     msvcrt.getch()
#print(train_data_health)

#############################################
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# def single_np(arr, target,flag):
#     arr = np.array(arr)
#     if flag:
#         mask = (arr < target)
#         arr_new = arr[mask]
#     else:
#         mask = (arr > target)
#         arr_new = arr[mask]
#     return arr_new.size

# save_path = "./HDDse_Siamese_LSTM/数据集/model_train.h5"
# model = keras.models.load_model(save_path,custom_objects = {"EluDist":EluDist,"contrastive_loss":contrastive_loss})

# data_path_dm = "HDDse_Siamese_LSTM/数据集/带模型数据/WDC/process/"
# test_data_model = pd.read_csv(data_path_dm + "test_data_only_model.csv",header = None)
# test_data_model = np.array(test_data_model).reshape(-1,14)

# #读取模型号对应的数据
# test_data_only = np.loadtxt(data_path_dm + "test_data_only_data.csv",delimiter = ",")
# test_data_only = test_data_only.reshape(-1,14,9)

# #读取failure的训练数据带模型与不带模型
# train_data_failure = pd.read_csv(data_path_dm + "failure_disks_with_model.csv")
# train_data_failure = pd.DataFrame(train_data_failure)
# train_data_failure_not = np.loadtxt(data_path_dm + "failure_disks.csv",delimiter = ",")
# print(len(train_data_failure_not))
# model_name = test_data_model[1][0]

# np.random.shuffle(train_data_failure_not)
# row_rand = train_data_failure_not[:int(len(train_data_failure_not) / 10),:]
# print(row_rand)
# print(len(row_rand))
# index = train_data_failure[train_data_failure.model == model_name].index.tolist() 
# #得到不带模型号的数据
# train_data_failure_tmp =  np.array(train_data_failure_not[index]).reshape(-1,14,9)
# pairs_1 = create_test_pairs(train_data_failure_tmp,test_data_only[1])

# prediction_temp = model.predict([pairs_1[:,0], pairs_1[:,1]])
# # print(prediction_temp)
# # print(single_np(prediction_temp, 0.5,True))
#####################################
#测试了pickle
# import pickle
# import os
# if os.path.getsize("predict_label.pickle") > 0: 
#     file_ = open("predict_label.pickle",'rb')
#     arr = pickle.load(file_)
#     file_.close()
#     print(arr.size)
####################################

# save_path = "./HDDse_Siamese_LSTM/数据集/model_train_m1.h5"
# model = keras.models.load_model(save_path,custom_objects = {"EluDist":EluDist,"contrastive_loss":contrastive_loss})
# #model.summary()

# label_test = np.append(dp.create_predict_label("test_health_disks.csv"),dp.create_predict_label("test_failure_disks.csv"))

# #根据decision maker生成预测值

# file_ = open("predict_label_m1.pickle",'rb')
# prediction_label = pickle.load(file_)
# prediction_label = np.array(prediction_label)
# file_.close()
# #根据预测值与真实值求召回率
# recall = classification_report(label_test,prediction_label)
# print(recall)


