import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

# https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
# https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.orthogonal_
# orthogonal_initializer = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
from tensorflow.keras.initializers import Orthogonal, Zeros


# https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo.py
class ValueNet(tf.keras.Model):
    """
    价值网络
    """

    def __init__(self):
        super().__init__()
        self.model = Sequential(
            [
                # Dense(units=hidden_dim, activation=tf.nn.relu),
                Dense(units=64, activation=tf.nn.tanh, kernel_initializer=Orthogonal(gain=np.sqrt(2)),
                      bias_initializer=Zeros()),
                Dense(units=64, activation=tf.nn.tanh, kernel_initializer=Orthogonal(gain=np.sqrt(2)),
                      bias_initializer=Zeros()),
                Dense(units=1, kernel_initializer=Orthogonal(gain=1.0), bias_initializer=Zeros())
            ]
        )
        pass

    def call(self, inputs):
        return self.model(inputs)
        pass
