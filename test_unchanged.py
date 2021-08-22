from keras.engine.base_layer import InputSpec
from keras import backend as K
import numpy as np

from unchanged import ConvLSTM2DCell
from unchanged import ConvRNN2D

cell = ConvLSTM2DCell(10, (2, 2))
convrnn = ConvRNN2D(cell, return_sequences=True)

_input = np.random.random([32, 10, 64, 64, 3])
input_shape = _input.shape
convrnn.build(input_shape)

inputs = K.random_uniform_variable(shape=(32, 10, 64, 64, 3), low=0, high=1)
initial_state = convrnn.get_initial_state(inputs)
output = convrnn.call(inputs)

def _get_dynamic_axis_num(x):
    if hasattr(x, 'dynamic_axes'):
        return len(x.dynamic_axes)
    else:
        return 0

output = convrnn.call(inputs)
print('output shape: ', output.shape)

print('\n ====== Test changes ======')
from convLSTM import ConvLSTM2DCell
from convLSTM import ConvRNN2D

cell = ConvLSTM2DCell(10, (2, 2))
convrnn = ConvRNN2D(cell, timestep_out=10, return_sequences=True)

_input = np.random.random([32, 10, 64, 64, 3])
input_shape = _input.shape
convrnn.build(input_shape)

inputs = K.random_uniform_variable(shape=(32, 64, 64, 3), low=0, high=1)
initial_state = convrnn.get_initial_state(inputs)
output = convrnn.call(inputs)