# 这里是生成训练集的所有相关代码
import numpy as np
import pandas as pd

data_path = "./Data/"


# 对于csv文件中的输入的形状进行更改,并返回数据
# (-1,9)->(-1,14,9)
def data_reshape(filename):
    data_set = pd.read_csv(data_path + filename, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], nrows=140, header=None)
    data_set = np.array(data_set)
    data_set = data_set.astype('float32')
    data_set = data_set.reshape(-1, 14, 9)
    return data_set  # shape为(-1,14,9)


# 生成训练数据pair对
# 主要是生成三类pair数据,<health,health>,<health,failure>,<failure,failure>
# 这些pair数据的标签：相同为1，不同为0
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

    # 加载样本池中健康的已标样本及其模型
    health_labeled = pd.read_csv('health.csv',  header=None)
    print("健康样本数据池已加载")

    # 加载样本池中故障的已标记样本及其模型
    failure_labeled = pd.read_csv('failure.csv',  header=None)
    print("故障样本数据池已加载")
    # 加载测试集
    test_set = pd.read_csv('test.csv', header=None)
    print("测试样本已加载")

    # 取出测试集数据
    test_set_data = test_set.iloc[:, 0:9].values
    test_set_data = test_set_data.reshape(-1, 14, 9)

    # 取出测试集合模型号
    test_set_model = test_set.iloc[:,9].values
    test_set_model = test_set_model.reshape(-1,14)

    label_predict = []
    cnt = 0
    for i in range(len(test_set_model)):
        if i % 100 == 0:
            print("滴滴滴:", i)

        # 获得该测试数据的模型号
        model_name = test_set_model[i][0]

        # 获得故障样本中的数据
        same_model_data = failure_labeled.loc[failure_labeled[9] == model_name]

        # 故障样本是否存在相同模型的数据
        if not same_model_data.empty:
            same_model_data = same_model_data.iloc[:, 0:9].values
            same_model_data = same_model_data.reshape(-1, 14, 9)

            # 然后将训练数据与测试数据组成pair
            pairs_1 = create_test_pairs(same_model_data, test_set_data[i])

            # 接着根据待检测的数据来进行预测
            prediction_temp = model.predict([pairs_1[:, 0], pairs_1[:, 1]])

            if (prediction_temp > 0.5).any():
                label_predict += [1]
                continue

        # 获得健康样本中的数据
        same_model_data = health_labeled.loc[health_labeled[9] == model_name]

        if not same_model_data.empty:
            same_model_data = same_model_data.iloc[:, 0:9].values
            same_model_data = same_model_data.reshape(-1, 14, 9)

            # 随机选择10%
            np.random.shuffle(same_model_data)
            same_model_data = same_model_data[:int(len(same_model_data) / 10), :]

            # 然后将训练数据与测试数据组成pair
            pairs_2 = create_test_pairs(same_model_data, test_set_data[i])

            # 接着根据待检测的数据来进行预测
            prediction_temp = model.predict([pairs_2[:, 0], pairs_2[:, 1]])

            count_2 = get_number(prediction_temp, True)

            # 根据预测的结果得到一个label
            if count_2 == len(prediction_temp):
                label_predict += [0]
                continue

        # 获得同一厂商的故障数据
        same_manu_data = failure_labeled.iloc[:, 0:9].values
        same_manu_data = same_manu_data.reshape(-1, 14, 9)

        # 然后将训练数据与测试数据组成pair
        pairs_3 = create_test_pairs(same_manu_data, test_set_data[i])

        # 接着根据待检测的数据来进行预测
        prediction_temp = model.predict([pairs_3[:, 0], pairs_3[:, 1]])
        count_3 = get_number(prediction_temp, False)
        if count_3 > (len(prediction_temp) / 2):
            label_predict += [0]
            continue

        # 获得同一厂商的健康数据
        same_manu_data = health_labeled.iloc[:, 0:9].values
        same_manu_data = same_manu_data.reshape(-1, 14, 9)
        np.random.shuffle(same_manu_data)
        same_manu_data = same_manu_data[:int(len(same_manu_data) / 10), :]

        # 然后将训练数据与测试数据组成pair
        pairs_4 = create_test_pairs(same_manu_data, test_set_data[i])

        # 接着根据待检测的数据来进行预测
        prediction_temp = model.predict([pairs_4[:, 0], pairs_4[:, 1]])
        count_4 = get_number(prediction_temp, True)

        if count_4 == len(prediction_temp):
            label_predict += [1]
        else:
            label_predict += [0]
    return np.array(label_predict)  # 返回预测结果得到的label
