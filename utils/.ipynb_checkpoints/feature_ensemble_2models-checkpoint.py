import sys
sys.path.append("/research/m324371/Project/adnexal/networks/")

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEnsemble2models(nn.Module):
    """ Ensemble feature maps of two models 
    Args:
        - model1: First model
        - model2: Second model
        - trim1: (int) No. of blocks to be trimmed from the end of model1, usually used to trim classification head
        - trim2: (int) No. of blocks to be trimmed from the end of model2, usually used to trim classification head

    Returns:
        - combined_features: A feature map combining both model1 and model2
    """
    
    def __init__(self, model1, model2, trim1:int, trim2:int):
        super(FeatureEnsemble2models, self).__init__()

        self.trimped_model1 = nn.Sequential(*list(model1.children()))[:-trim1] # trim first model
        self.trimped_model2 = nn.Sequential(*list(model2.children()))[:-trim2] # trim second model

    def forward(self, x):

        features1 = self.trimped_model1(x)
        features2 = self.trimped_model2(x)

        # Ensure both feature maps have the same spatial dimensions
        if features1.shape[2:] != features2.shape[2:]:
            # Resize feature2's output to match feature1's spatial dimensions
            features2 = F.interpolate(features2, size=features1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate the feature maps along the channel dimension
        combined_features = torch.cat((features1, features2), dim=1) 
        
        return combined_features

if __name__ == "__main__":
    from res50pscse_512x28x28 import ResNet50Pscse_512x28x28
    from enetb2lpscse_384x28x28 import EfficientNetB2LPscse_384x28x28

    inp=torch.rand(1, 3, 224, 224)
    num_classes=2
    out_channels=[1024, 512, 256]
    pretrain=True
    dropout=0.3
    activation='leakyrelu'
    reduction=16
    
    model1 = ResNet50Pscse_512x28x28(num_classes, out_channels, pretrain, dropout, activation, reduction) 
    model2 = EfficientNetB2LPscse_384x28x28(num_classes, out_channels, pretrain, dropout, activation, reduction)
    
    feature_ensembled_model = FeatureEnsemble2models(model1, model2, trim1=1, trim2=1) # trim classification head
    out = feature_ensembled_model(inp)
    
    print(out.shape) # torch.Size([1, 2048, 7, 7])
