from mytorch import Tensor
from mytorch.layer import Layer

class AvgPool2d(Layer):
    def __init__(self) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        return ...

    def __str__(self) -> str:
        return "avg pool 2d"