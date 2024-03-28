from mytorch import Tensor
from mytorch.layer import Layer

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, name, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), initialize_method="random") -> None:
        super()
        pass

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        return ...
    
    def __str__(self) -> str:
        return "conv 2d"