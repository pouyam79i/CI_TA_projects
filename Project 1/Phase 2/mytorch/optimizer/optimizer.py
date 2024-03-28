from typing import List
from mytorch.layer import Layer

class Optimizer:
    def __init__(self, params: List[Layer]):
        self.params = params
        pass
    
    def step(self):
        pass

    def zero_grad(self):
        pass