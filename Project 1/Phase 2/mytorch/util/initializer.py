import numpy as np

def xavier_initializer(shape):
    "TODO: implement xavier_initializer" 
    return ...

def he_initializer(shape):
    "TODO: implement he_initializer" 
    return ...

def random_normal_initializer(shape, mean=0.0, stddev=0.05):
    "TODO: implement random_normal_initializer" 
    return ...

def zero_initializer(shape):
    "TODO: implement zero_initializer" 
    return ...

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
