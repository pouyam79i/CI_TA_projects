from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class Linear(Layer):
    def __init__(self, inputs: int, outputs: int) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.params = ...
        pass

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        return ...
    
    def initialize(self):
        "TODO: initialize weights and bias"
        pass
    
    def __str__(self) -> str:
        return "linear"
