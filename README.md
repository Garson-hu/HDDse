# 之前的代码都是在VSCode上写的，现在将代码进行一个简单的迁移，在Pycharm上也创建一个仓库
# 主要是实现了ATC2020的文章HDDse
# 论文地址为：https://www.usenix.org/system/files/atc20-zhang-ji.pdf
# 每个文件的含义为：
# parameter.py 中包含了一些超参数
# Siamese_LSTM.py 是整个网络的结构
# Eludist_loss.py 包含自定义用来求欧式距离的层以及损失函数
# predict.py 用来加载训练后保存的网络模型，然后进行预测等任务