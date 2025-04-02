import torch.nn as nn

def loss_func(name, *args):
    if name == 'ce':
        return nn.CrossEntropyLoss()
    elif name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"{name} is not found in supported losses.")