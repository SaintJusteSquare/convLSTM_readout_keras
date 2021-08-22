from keras.engine.base_layer import InputSpec
from keras.utils import conv_utils
from keras import backend as K
import numpy as np

from convLSTM import ConvLSTM2DCell
from convLSTM import ConvRNN2D

cell = ConvLSTM2DCell(10, (2, 2))
convrnn = ConvRNN2D(cell, 20)

_input = np.random.random([32, 10, 64, 64, 3])
input_shape = _input.shape
convrnn.build(input_shape)

inputs = K.random_uniform_variable(shape=(32, 10, 64, 64, 3), low=0, high=1)

# (samples, timesteps, rows, cols, filters)
initial_state = K.zeros_like(inputs)
# (samples, rows, cols, filters)
initial_state = K.sum(initial_state, axis=1)
shape = list(cell.kernel_shape)
shape[-1] = cell.filters

print('shape: ', shape)

import tensorflow as tf
kernel = tf.zeros(tuple(shape))
print('kernel shape: ', kernel.shape)

initial_state = cell.input_conv(initial_state, kernel, padding=cell.padding)
print("initiall state: ", initial_state.shape)

keras_shape = list(K.int_shape(inputs))
keras_shape.pop(1)
print('keras shape: ', keras_shape)

print('K.image_data_format(): ', K.image_data_format())

if K.image_data_format() == 'channels_first':
    indices = 2, 3
else:
    indices = 1, 2
for i, j in enumerate(indices):
    print('i: ', i)
    print('j: ', j)
    print('before kernel shape[j]: ', keras_shape[j])
    print('before shape i: ', shape[i])
    keras_shape[j] = conv_utils.conv_output_length(
        keras_shape[j],
        shape[i],
        padding=cell.padding,
        stride=cell.strides[i],
        dilation=cell.dilation_rate[i])
    print('after kernel shape[j]: ', keras_shape[j])
    print('after shape i: ', shape[i])

quit()
initial_state._keras_shape = keras_shape

"""if hasattr(self.cell.state_size, '__len__'):
    return [initial_state for _ in self.cell.state_size]
else:
    return [initial_state]"""

print('\n ====== Test changes ======')