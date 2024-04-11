import numpy as np


# TODO: implement xavier_initializer, he_initializer, random_normal_initializer, zero_initializer

def xavier_initializer(shape):
    pass


def he_initializer(shape):
    pass


def random_normal_initializer(shape, mean=0.0, stddev=0.05):
    pass


def zero_initializer(shape):
    pass


def one_initializer(shape):
    return np.ones(shape, dtype=np.float64)


def initializer(shape, mode="xavier"):
    if mode == "xavier":
        return xavier_initializer(shape)
    elif mode == "he":
        return he_initializer(shape)
    elif mode == "random_normal":
        return random_normal_initializer(shape)
    elif mode == "zero":
        return zero_initializer(shape)
    elif mode == "one":
        return one_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
