from typing import Any
from mytorch import Tensor

class Model:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(args[0])

    def forward(self, x: Tensor) -> Tensor:
        pass

    def train(self):
        pass
    
    def eval(self):
        pass
