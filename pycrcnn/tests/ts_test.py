import unittest

import numpy as np
from Pyfhel import Pyfhel
import numpy as np
import torch.nn as nn
import torch
from pycrcnn.models__ import Square

from pycrcnn.convolutional import convolutional_layer as conv
from pycrcnn.convolutional.convolutional_layer_ts import ConvolutionalLayer
from pycrcnn.crypto import crypto as cr
from pycrcnn.linear import linear_layer_ts as lin

class MyTestCase(unittest.TestCase):
    
    def setUp(self) -> None:
        self.HE = Pyfhel()
        self.HE.contextGen(65537)
        self.HE.keyGen()
        self.HE.relinKeyGen(20, 100)    

    def test_conv_pool(self):
        input = np.array([[[2],
                            [2],
                            [2],
                            [2],
                            [2]],
                            
                            [[1],
                            [1],
                            [1],
                            [1],
                            [1]]
                            ])                            
                                    
        encrypted_image = cr.encrypt_matrix(self.HE, input)

        weights = np.array([[[ 1],
                                [2],
                                [3],
                                [4],
                                [5]],

                                [[0],
                                [0],
                                [0],
                                [0],
                                [0]]])

        bias = np.array([0,1])            

        conv_layer = ConvolutionalLayer(self.HE, weights, stride=(1, 1), padding=(0, 0), bias=bias)
        result = cr.decrypt_matrix(self.HE, conv_layer(encrypted_image))

        expected_result = np.array([[[30.],[1.]],
                                    [[15.],[1.]]]
        )

        self.assertTrue(np.allclose(result, expected_result))

    def test_conv_pool2(self):

        model = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=1),
            
            Square(),
            
            nn.Flatten(),

            nn.Linear(32, 10), 
            nn.Linear(10, 1)
        )

        param = model.named_parameters()
        weights = next(param)[1]
        bias = next(param)[1]
        input = torch.randn(32,5,1)     
        
        out = model[0](input)
                
                                   
                                            
        encrypted_image = cr.encrypt_matrix(self.HE, input.cpu().detach().numpy()) 

        conv_layer = ConvolutionalLayer(self.HE, weights.cpu().detach().numpy(), stride=(1, 1), padding=(0, 0), bias=bias.cpu().detach().numpy())
        result = cr.decrypt_matrix(self.HE, conv_layer(encrypted_image))

        expected_result = out.cpu().detach().numpy()

        self.assertTrue(np.allclose(result, expected_result))


    def test_linear(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(30, 100)

        model = nn.Sequential(
            

            nn.Linear(32, 10), 
            nn.Linear(10, 1)
        )

        param = model.named_parameters()
        weights = next(param)[1]
        bias = next(param)[1]
        input = torch.rand(32,32)
        out = model[0](input)

        

        linear_layer = lin.LinearLayer(HE, weights.cpu().detach().numpy(), bias.cpu().detach().numpy())

        flattened_input = input.cpu().detach().numpy()
        encrypted_input = cr.encrypt_matrix(HE, flattened_input)

        encrypted_result = linear_layer(encrypted_input)
        result = cr.decrypt_matrix(HE, encrypted_result)

        expected_result = out.cpu().detach().numpy()
        self.assertTrue(np.allclose(result, expected_result))


if __name__ == '__main__':
    unittest.main()