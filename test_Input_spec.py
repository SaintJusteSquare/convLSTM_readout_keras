from keras.utils.generic_utils import transpose_shape
from keras.engine.base_layer import InputSpec
import numpy as np

# TODO : Need to change the input_spec linked things later.

# Premier cas: input tensor de taille 5.
print('====== Premier cas ======')

stateful = False
_input = np.random.random([10, 64, 64, 3])
input_shape = _input.shape
print(input_shape)

input_spec = [InputSpec(ndim=5)]
batch_size = input_shape[0] if stateful else None
input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])
input_shape = input_spec[0].shape
print('(batch_size, None): ', type((batch_size, None)))
print('input_shape[2:5]: ', type(input_shape[2:5]))
print(input_shape)
print('input_shape: ', type(input_shape))

rows = 32
cols = 32
filters = 20
data_format = 'channels_last'

print(input_shape[:2])

output_shape = input_shape[:2] + (rows, cols, filters)
print(output_shape)

output_shape = transpose_shape(output_shape, data_format, spatial_axes=(2, 3))
print(output_shape)
print('output_shape: ', type(output_shape))

print('\n')

# Second cas: input tensor de taille 4.
print('====== Second Cas ======')

stateful = False
_input = np.random.random([64, 64, 3])
input_shape = _input.shape
print(input_shape)

input_spec = [InputSpec(ndim=4)]
batch_size = input_shape[0] if stateful else None
input_spec[0] = InputSpec(shape=tuple([batch_size]) + input_shape[1:4])
input_shape = input_spec[0].shape
print(input_shape)
print('input_shape: ', type(input_shape))

rows = 32
cols = 32
filters = 20
data_format = 'channels_last'

print(input_shape[:1])

output_shape = tuple(input_shape[:1]) + (rows, cols, filters)

print(output_shape)

output_shape = transpose_shape(output_shape, data_format, spatial_axes=(2, 3))
print(output_shape)
print('output_shape: ', type(output_shape))

print('\n')

# Troisème cas: inut tensor taille 4 et output tensor taille 5.
print('====== Troisième Cas ======')

stateful = False
_input = np.random.random([64, 64, 3])
input_shape = _input.shape
print(input_shape)

input_spec = [InputSpec(ndim=4)]
batch_size = input_shape[0] if stateful else None
input_spec[0] = InputSpec(shape=tuple([batch_size]) + input_shape[1:4])
input_shape = input_spec[0].shape
print(input_shape)
print('input_shape: ', type(input_shape))

rows = 32
cols = 32
filters = 20
data_format = 'channels_last'

print(input_shape[:1])

output_shape = input_shape[:1] + (None, rows, cols, filters)

print(output_shape)

output_shape = transpose_shape(output_shape, data_format, spatial_axes=(2, 3))
print(output_shape)
print('output_shape: ', type(output_shape))