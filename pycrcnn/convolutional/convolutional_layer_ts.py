import numpy as np

from pycrcnn.functional.padding import apply_padding
from ..crypto import crypto as c


class ConvolutionalLayer:
    """
    A class used to represent a convolutional layer
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to encode weights and bias
    weights : np.array( dtype=PyPtxt )
        Weights of the layer, aka filters in form
        [n_filters, n_layers, y, x]
    stride : (int, int)
        Stride (y, x)
    padding : (int, int)
        Padding (y, x)
    bias : np.array( dtype=PyPtxt ), default=None
        Biases of the layer, 1-D array


    Methods
    -------
    __init__(self, HE, weights, x_stride, y_stride, bias=None)
        Constructor of the layer, bias is set to None if not provided.
    __call__(self, t)
        Execute che convolution operation on a batch of images, t, in the form
            [n_images, n_layers, y, x]
        using weights, biases and strides of the layer.
    """

    def __init__(self, HE, weights, stride=(1, 1), padding=(0, 0), bias=None):
        self.HE = HE
        self.weights = c.encode_matrix(HE, weights)
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if bias is not None:
            self.bias = c.encode_matrix(HE, bias)

    

    def __call__(self, t):
        out = []
        for ts in t:
            for b,filter in enumerate(self.weights):
                i = 0
                while(i <= ts.shape[1] - filter.shape[1]):
                    out.append(np.sum(np.multiply(np.expand_dims(subMatrix(ts,i,filter.shape[1]),0),filter)) + self.bias[b])
                    i += 1
        
        return (np.array(out).reshape(t.shape[0],self.weights.shape[0],t.shape[2]-self.weights.shape[2]+1))

def subMatrix(ts,idx,kernel_length):    
    return ts[0][idx:kernel_length+idx]