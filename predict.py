# 本代码是根据已有的模型去预测新的样本

import numpy as np
import keras.models
import data_process as dp
# import keras_metrics as km
import pickle
from data_process import EluDist
from keras import backend as K
# from keras.layers import Lambda,Layer
# from keras.utils import plot_model
from sklearn.metrics import classification_report


# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def contrastive_loss(y_true, y_pred):
    margin = 1.25
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


# 加载模型
save_path = "./HDDse_Siamese_LSTM/数据集/model_train_m125.h5"
model = keras.models.load_model(save_path, custom_objects={"EluDist": EluDist, "contrastive_loss": contrastive_loss})
model.summary()

# plot_model(model, to_file='model_architecture.png',show_shapes=True)

# 生成测试样本的标签,用来后续与预测值进行对比
label_test = np.append(dp.create_predict_label("test_health_disks.csv"),
                       dp.create_predict_label("test_failure_disks.csv"))

# 根据decision maker生成预测值
prediction_label = dp.decision_maker(model)
file = open("predict_label.pickle", 'wb')
pickle.dump(prediction_label, file)
file.close()
# 根据预测值与真实值求召回率
recall = classification_report(label_test, prediction_label)
print(recall)
