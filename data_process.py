# 这里是生成训练集的所有相关代码
import numpy as np
import pandas as pd
from keras.layers import Layer
from keras import backend as K

data_path = "数据集/test_1/"

class EluDist(Layer):  # 封装成keras层的欧式距离计算

    # 初始化EluDist层，此时不需要任何参数输入
    def __init__(self, **kwargs):
        self.result = None
        super(EluDist, self).__init__(**kwargs)

    # 建立EluDist层
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(256, 1),
                                      initializer='uniform',
                                      trainable=True)
        super(EluDist, self).build(input_shape)

    # 计算欧式距离
    def call(self, vects, **kwargs):
        # 是否需要for循环
        x, y = vects
        self.result = K.abs(x - y)
        self.result = K.dot(self.result, self.kernel)  # β乘上欧式距离
        self.result = K.sigmoid(self.result)
        return self.result

    # 返回结果
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


# 对于csv文件中的输入的形状进行更改,并返回数据
# (-1,9)->(-1,14,9)
def data_reshape(filename):
    data_set = np.loadtxt(data_path + filename, delimiter=",", skiprows=0)  # 去掉了max_rows
    data_set = data_set.astype('float32')
    data_set = data_set.reshape(-1, 14, 9)
    return data_set  # shape为(-1,14,9)


# 生成训练数据pair对
# 主要是生成三类pair数据,<health,health>,<health,failure>,<failure,failure>
# 这些pair数据的标签分别为(Y = 1,Y = 0,Y = 1)
def create_pairs(health_data_set, failure_data_set):
    pairs = []
    labels = []
    for i in range(len(health_data_set)):
        for j in range(len(failure_data_set)):
            pairs += [[health_data_set[i], failure_data_set[j]]]
            labels += [0]
    for i in range(len(health_data_set) - 1):
        j = i + 1
        for j in range(len(failure_data_set)):  # 在这里进行了更改:health_data_set ---->failure_data_set
            pairs += [[health_data_set[i], health_data_set[j]]]
            labels += [1]
    for i in range(len(failure_data_set) - 1):
        j = i + 1
        for j in range(len(failure_data_set)):
            pairs += [[failure_data_set[i], failure_data_set[j]]]
            labels += [1]
    return np.array(pairs), np.array(labels)


# 测试数据的label需要求出来,然后与prediction的数据去比较
def create_predict_label(filename):
    labels = []
    data_set = data_reshape(filename)
    if "health" in filename:
        for i in range(len(data_set)):
            labels += [0]
    else:
        for i in range(len(data_set)):
            labels += [1]
    return np.array(labels)


# 主要是对测试数据创建pairs,后面一个数据是待检测数据
def create_test_pairs(data_1, data_2):
    pairs = []
    for i in range(len(data_1)):
        pairs += [[data_1[i], data_2]]
    return np.array(pairs)


# 统计数量,健康时为True,不健康时为False
def get_number(arr, flag):
    arr = np.array(arr)
    if flag:
        mask = (arr < 0.5)
        # print(mask)
        arr_new = arr[mask]
    else:
        mask = (arr > 0.5)
        arr_new = arr[mask]
    return arr_new.size


# 下面的代码实现四步decision maker
def decision_maker(model):
    data_path = "数据集/带模型数据/WDC/process/"
    test_data_model = pd.read_csv(data_path + "test_data_only_model.csv", header=None)
    test_data_model = np.array(test_data_model).reshape(-1, 14)

    # 读取模型号对应的数据
    test_data_only = np.loadtxt(data_path + "test_data_only_data.csv", delimiter=",")
    test_data_only = test_data_only.reshape(-1, 14, 9)

    # 读取health的训练数据带模型与不带模型
    train_data_health = pd.read_csv(data_path + "health_disks_with_model.csv")
    train_data_health = pd.DataFrame(train_data_health)
    train_data_health_not = np.loadtxt(data_path + "health_disks.csv", delimiter=",")

    # 读取failure的训练数据带模型与不带模型
    train_data_failure = pd.read_csv(data_path + "failure_disks_with_model.csv")
    train_data_failure = pd.DataFrame(train_data_failure)
    train_data_failure_not = np.loadtxt(data_path + "failure_disks.csv", delimiter=",")
    label_predict = []
    cnt = 0
    for i in range(len(test_data_model)):
        if i % 100 == 0 and i != 0:
            print("滴滴滴:", i)

        # 获得该测试数据的模型号
        model_name = test_data_model[i][0]

        # 获得故障样本中的数据
        same_model_data = train_data_failure.loc[train_data_failure['model'] == model_name]

        # 故障样本是否存在相同模型的数据
        if same_model_data.empty == False:
            same_model_data = same_model_data.iloc[:, 0:9].values
            same_model_data = same_model_data.reshape(-1, 14, 9)

            # 然后将训练数据与测试数据组成pair
            pairs_1 = create_test_pairs(same_model_data, test_data_only[i])

            # 接着根据待检测的数据来进行预测
            prediction_temp = model.predict([pairs_1[:, 0], pairs_1[:, 1]])

            if (prediction_temp > 0.5).any():
                label_predict += [1]
                cnt += 1
                continue

        # 获得健康样本中的数据
        same_model_data = train_data_health.loc[train_data_health['model'] == model_name]

        if same_model_data.empty == False:
            same_model_data = same_model_data.iloc[:, 0:9].values
            same_model_data = same_model_data.reshape(-1, 14, 9)

            # 随机选择10%
            np.random.shuffle(same_model_data)
            same_model_data = same_model_data[:int(len(same_model_data) / 10), :]

            # 然后将训练数据与测试数据组成pair
            pairs_2 = create_test_pairs(same_model_data, test_data_only[i])

            # 接着根据待检测的数据来进行预测
            prediction_temp = model.predict([pairs_2[:, 0], pairs_2[:, 1]])

            count_2 = get_number(prediction_temp, True)

            # 根据预测的结果得到一个label
            if count_2 == len(prediction_temp):
                label_predict += [0]
                cnt += 1
                continue

        # 获得同一厂商的故障数据
        same_manu_data = train_data_failure_not.reshape(-1, 14, 9)

        # 然后将训练数据与测试数据组成pair
        pairs_3 = create_test_pairs(same_manu_data, test_data_only[i])

        # 接着根据待检测的数据来进行预测
        prediction_temp = model.predict([pairs_3[:, 0], pairs_3[:, 1]])
        count_3 = get_number(prediction_temp, False)
        if count_3 > (len(prediction_temp) / 2):
            label_predict += [0]
            cnt += 1
            continue

        # 获得同一厂商的健康数据
        same_manu_data = train_data_health_not.reshape(-1, 14, 9)
        np.random.shuffle(same_model_data)
        same_model_data = same_model_data[:int(len(same_model_data) / 10), :]

        # 然后将训练数据与测试数据组成pair
        pairs_4 = create_test_pairs(same_manu_data, test_data_only[i])

        # 接着根据待检测的数据来进行预测
        prediction_temp = model.predict([pairs_4[:, 0], pairs_4[:, 1]])
        count_4 = get_number(prediction_temp, True)

        if count_4 == len(prediction_temp):
            label_predict += [1]
            cnt += 1
        else:
            label_predict += [0]
            cnt += 1
    print(cnt)
    return np.array(label_predict)  # 返回预测结果得到的label
