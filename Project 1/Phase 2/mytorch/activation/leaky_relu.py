import numpy as np
from mytorch import Tensor, Dependency

def leaky_relu(x: Tensor) -> Tensor:
    """
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn
    hint: use np.where like Relu method but for LeakyRelu
    """
    # TODO: implement leaky_relu function

    data = ...

    req_grad = ...
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return None

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
