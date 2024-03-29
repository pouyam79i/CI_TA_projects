import numpy as np
from mytorch import Tensor


def flatten(x: Tensor) -> Tensor:
    data = x.data.flatten()
    req_grad = x.requires_grad
    depends_on = x.depends_on

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


# print(flatten(Tensor(np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]))))
