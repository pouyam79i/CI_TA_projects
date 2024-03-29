from typing import Any
from mytorch import Tensor

class Layer:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(args[0])

    def forward(self, x: Tensor) -> Tensor:
        return None
    
    def __str__(self) -> str:
        return "Layer class is an abstract for other type of layers"
