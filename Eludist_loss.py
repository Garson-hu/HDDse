from keras.layers import Layer
import keras.backend as K
import parameter


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
        x, y = vects
        self.result = K.abs(x - y)
        self.result = K.dot(self.result, self.kernel)  # β乘上欧式距离
        self.result = K.sigmoid(self.result)
        return self.result

    # 返回结果
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


# 计算损失函数
def contrastive_loss(y_true, y_pred):
    margin = parameter.MARGIN
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (margin - y_true) * margin_square)
