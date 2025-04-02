import albumentations as A

def normalize(config):
    return A.Compose([A.Normalize(mean=config.normalize.mean, std=config.normalize.std)], p=1) 
    

import torch

def normalize_tensor(tensor, a:float=0.0, b:float=1.0):
    """
    Normalizes a tensor to a specified range [a, b].

    Parameters:
    tensor (torch.Tensor): The tensor to be normalized (shape [1, num_features]).
    a (float): The minimum value of the target range.
    b (float): The maximum value of the target range.

    Returns:
    torch.Tensor: The normalized tensor within the range [a, b].
    """
    # Find the min and max values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    
    if min_val == max_val:
        raise ValueError("All values in the tensor are the same. Normalization is not possible.")
    
    # Normalize the tensor to the range [0, 1] and scale to [a, b]
    normalized_tensor = a + ((tensor - min_val) / (max_val - min_val)) * (b - a)
    
    return normalized_tensor