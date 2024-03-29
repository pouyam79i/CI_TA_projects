import numpy as np
from mytorch import Tensor, Dependency


def step(x: Tensor) -> Tensor:

    data = np.where(x.data < 0, np.zeros_like(x.data), np.ones_like(x.data))

    req_grad = x.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * 0

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)

print(step(Tensor(np.array([[-1, -2, 3],
                            [4, 5, -6],
                            [7, 8, 9]]),
                  requires_grad=True)))
