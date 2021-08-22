from keras.engine.base_layer import InputSpec
import numpy as np

from convLSTM import ConvLSTM2DCell
from convLSTM import ConvRNN2D

convlstmcell = ConvLSTM2DCell(10, (2,2))
convrnn = ConvRNN2D(convlstmcell, 20)

_input = np.random.random([32, 10, 64, 64, 3])
input_shape = _input.shape
input_spec = [InputSpec(ndim=5)]
batch_size = None
input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])
print('input_spec[0]: ', input_spec[0])
step_input_shape = (input_shape[0],) + input_shape[2:]
print('step_inut_shape: ', step_input_shape)
convlstmcell.build(step_input_shape)
state_size = list(convlstmcell.state_size)
print('state size: ', state_size)

print('\n ====== Test changes ======')

_input = np.random.random([32, 64, 64, 3])
input_shape = _input.shape
convrnn.build(input_shape)



