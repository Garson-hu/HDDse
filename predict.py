# 本代码是根据已有的模型去预测新的样本

import pickle
import numpy as np
import data_process as dp
import keras.models
from Eludist_loss import EluDist, contrastive_loss
from sklearn.metrics import classification_report

# 加载模型
save_path = "数据/model_train_m1.h5"
model = keras.models.load_model(save_path, custom_objects={"EluDist": EluDist, "contrastive_loss": contrastive_loss})  # , "contrastive_loss": contrastive_loss
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
