from keras.engine.base_layer import InputSpec
from keras.utils import conv_utils
from keras import backend as K
import numpy as np

from convLSTM import ConvLSTM2DCell
from convLSTM import ConvRNN2D

cell = ConvLSTM2DCell(10, (2, 2))
convrnn = ConvRNN2D(cell, 20)

_input = np.random.random([32, 64, 64, 3])
input_shape = _input.shape
convrnn.build(input_shape)

inputs = K.random_uniform_variable(shape=(32, 64, 64, 3), low=0, high=1)
output = convrnn.call(inputs)