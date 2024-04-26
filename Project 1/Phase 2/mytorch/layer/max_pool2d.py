from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class MaxPool2d(Layer):
    def __init__(self, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        return ...
    
    def __str__(self) -> str:
        return "max pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
