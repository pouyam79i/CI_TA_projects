from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class Linear(Layer):
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias

        "TODO: build your linear network using initialize method"
        self.params = []
        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        return ...
    
    def initialize(self, mode="random"):
        "TODO: initialize weights and bias"
        pass

    def zero_grad(self):
        "TODO: implement zero grad"
        pass

    def parameters(self):
        "TODO: return weights and bias"
        return ...
    
    def __str__(self) -> str:
        return "linear - total param: {} - in: {}, out: {}".format(self.inputs * self.outputs, self.inputs, self. outputs)
 