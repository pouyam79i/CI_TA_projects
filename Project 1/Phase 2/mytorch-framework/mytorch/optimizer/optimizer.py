from typing import List
from mytorch.layer import Layer


class Optimizer:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        pass

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
