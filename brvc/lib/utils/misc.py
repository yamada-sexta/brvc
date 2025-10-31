def get_padding(kernel_size: int, dilation: int=1):
    return int((kernel_size * dilation - dilation) / 2)


from torch import nn
def init_weights(m, mean: float = 0.0, std: float = 0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
