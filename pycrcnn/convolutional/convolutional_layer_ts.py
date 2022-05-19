import numpy as np

from pycrcnn.functional.padding import apply_padding
from ..crypto import crypto as c


class ConvolutionalLayer:
    def __init__(self, HE, weights, stride=(1, 1), padding=(0, 0), bias=None):
        self.HE = HE
        self.weights = c.encode_matrix(HE, weights)
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if bias is not None:
            self.bias = c.encode_matrix(HE, bias)

    def __call__(self, t):
        # t = apply_padding(t, self.padding)
        result = np.array([[np.sum([convolute1d(ts_layer, filter_layer, self.stride)
                                    for ts_layer, filter_layer in zip(_ts, _filter)], axis=0)
                             for _filter in self.weights] 
                           for _ts in t])

        if self.bias is not None:
            return np.array([[layer + bias for layer, bias in zip(_ts, self.bias)] for _ts in result])
        else:
            return result


def convolute1d(ts, filter_matrix, stride):
    x_d = len(ts)
    x_f = len(filter_matrix)

    x_stride = stride[0]

    x_o = ((x_d - x_f) // x_stride) + 1

    def get_submatrix(matrix, x):
        index_column = x * x_stride
        return matrix[index_column: index_column + x_f]

    return np.array(
        [np.sum(get_submatrix(ts, x) * filter_matrix) for x in range(0, x_o)])