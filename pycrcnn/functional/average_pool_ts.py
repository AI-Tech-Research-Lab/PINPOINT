import numpy as np

from pycrcnn.functional.padding import apply_padding


class AveragePoolLayer:
    """
    A class used to represent a layer which performs an average pool operation
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to perform the pool operation
    kernel_size: int
        Size of the square kernel
    stride : int
        Stride of the pool operaiton

    Methods
    -------
    __init__(self, HE, kernel_size, stride)
        Constructor of the layer.
    __call__(self, t)
        Execute che average pooling operation on a batch of images, t, in the form
            [n_images, n_layers, y, x]
        using kernel size and strides of the layer.
    """
    def __init__(self, HE, kernel_size, stride=(1, 1), padding=(0, 0)):
        self.HE = HE
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    

    def __call__(self, t):
        def subMatrix(ts):
            avg_arr = np.array([])    
            for elem in np.array_split(ts,2):
                if(len(elem)==2):
                    avg_arr = np.append(avg_arr,(elem[0]+elem[1])/2)
            return avg_arr    

        inner_dim = np.array([])
        outer_dim = np.array([])
        for dim1 in t:            
            for ts in dim1:                
                inner_dim = np.append(inner_dim,subMatrix(ts))            
            
            outer_dim = np.append(outer_dim,inner_dim)
            inner_dim = np.array([])

        return outer_dim.reshape(t.shape[0],t.shape[1],1)
            

    
