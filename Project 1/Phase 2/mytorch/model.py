from typing import Any, List
from mytorch import Tensor
from mytorch.layer import Layer

"This class is an abstraction for your model."
class Model:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(args[0])

    "Override this method when defining your own model."
    def forward(self, x: Tensor) -> Tensor:
        return None

    def train(self):
        "TODO: (optional) prepare model for training."
        pass
    
    def eval(self):
        "TODO: (optional) prepare model for evaluation."
        pass
    
    def parameters(self) -> List[Layer]:
        params = []
        for _, attribValue in self.__dict__.items():
            if issubclass(attribValue, Layer):
                params.append(attribValue)
        return params

    def summary(self):
        for attribName, attribValue in self.__dict__.items():
            if issubclass(attribValue, Layer):
                print(attribName + ': ', attribValue)
