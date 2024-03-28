from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class AvgPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.params = ...
        pass

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        return ...
    
    def initialize(self): 
        "TODO: initialize weights and bias"
        pass

    def __str__(self) -> str:
        return "avg pool 2d"
