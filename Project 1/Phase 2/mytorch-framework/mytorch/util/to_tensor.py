from mytorch import Tensor
import numpy as np
from torchvision.transforms import ToTensor as PytorchToTensor, Compose


class ToMyTorchTensor(object):
    def __call__(self, sample):
        return Tensor(sample.numpy)


def custom_collate(batch):
    if isinstance(batch[0][0], Tensor):
        return [item[0] for item in batch], [item[1] for item in batch]
    else:
        return torch.utils.data.dataloader.default_collate(batch)


class MyTransform(object):
    def __call__(self, sample):
        return ToMyTorchTensor()(transforms.ToTensor()(sample))
