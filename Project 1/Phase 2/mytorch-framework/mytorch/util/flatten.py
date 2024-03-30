import numpy as np
from mytorch import Tensor

def flatten(x: Tensor) -> Tensor:
    data = x.data.flatten()
    req_grad = x.requires_grad
    depends_on = x.depends_on

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
