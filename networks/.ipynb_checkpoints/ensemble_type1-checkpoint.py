"""
Author@ Mrinal Kanti Dhar
October 24, 2024
"""

import sys
sys.path.append("/research/m324371/Project/adnexal/utils/")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from copy import deepcopy

from pscse_cab import PscSEWithCAB
from classification_head import ClassificationHead
from feature_ensemble_2models import FeatureEnsemble2models

from res50pscse_512x28x28 import ResNet50Pscse_512x28x28
from enetb2lpscse_384x28x28 import EfficientNetB2LPscse_384x28x28
from base_models_collection import base_models
from base_models_features_collection import base_models_features_only

# ===========================================================================================================
class EnsembleResNet18Ft512_EfficientNetB2SFt1408(nn.Module):
    """ Ensembles ResNet18 with 512 features and EfficientNetB2 with 1408 features """
    def __init__(self, 
                 num_classes: int,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 pretrain: bool = True,
                 dropout: float = 0.3,
                 in_chs: int = None,
                 separate_inputs: int = None):  # separate_inputs defines the number of inputs

        super(EnsembleResNet18Ft512_EfficientNetB2SFt1408, self).__init__()
        
        self.separate_inputs = separate_inputs

        model1 = base_models('resnet18', pretrain=pretrain, num_classes=num_classes, in_chs=in_chs)
        model2 = base_models('efficientnet_v2_s', pretrain=pretrain, num_classes=num_classes, in_chs=in_chs)

        self.ens_model1 = FeatureEnsemble2models(model1, model2, trim1=2, trim2=2)  # clip classification head

        # Create a list of models for separate inputs if separate_inputs is specified
        if self.separate_inputs is not None:
            self.ensemble_models = nn.ModuleList([deepcopy(self.ens_model1) for _ in range(self.separate_inputs)])

        self.classification = ClassificationHead(num_classes=num_classes,
                                                 out_channels=out_channels,
                                                 dropout=dropout)

    def forward(self, x):

        if self.separate_inputs is not None:

            # Ensure input data has self.separate_inputs no. of channels
            if x.shape[1] < self.separate_inputs:
                raise ValueError(f"Can't split. Input data has {x.shape[1]} channels whereas separate_inputs parameter is {self.separate_inputs}. \
Check the separate_inputs parameter in the config file.")
            
            features_list = []

            # Loop over each input channel, process it, and store the features
            for i in range(self.separate_inputs):
                # Separate the i-th input (single channel)
                xi = x[:, i:i + 1, :, :]  # extract ith channel

                # Convert to 3 channels by repeating or concatenating along the channel dimension
                xi_3ch = torch.cat([xi, xi, xi], dim=1)

                # Get features from the i-th ensemble model
                features_i = self.ensemble_models[i](xi_3ch)

                # Collect features
                features_list.append(features_i)

            # Concatenate features along the channel dimension
            features = torch.cat(features_list, dim=1)

        else:
            features = self.ens_model1(x)

        # Pass the features through the classification head
        out = self.classification(features)

        return out

""" Example
if __name__ == "__main__":
    inp=torch.rand(1, 3, 224, 224)
    num_classes=2
    out_channels=[5376, 1024, 512, 256]
    pretrain = True
    dropout=0.3
    separate_inputs = 3
    in_channels = 3
    
    model = EnsembleResNet18Ft512_EfficientNetB2SFt1408(num_classes, out_channels, pretrain, dropout, in_channels, separate_inputs)
    
    out = model(inp)
    
    print(out.shape)
"""    

# ===========================================================================================================
class EnsembleResNet18Ft512_EfficientNetB2SFt1408V2(nn.Module):
    """ Ensembles ResNet18 with 512 features and EfficientNetB2 with 1408 features.
        V2 added a new argument only_feature_extraction. It decides whether the model be a feature extractor
        or a classifier. 
    """
    def __init__(self, 
                 num_classes: int,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 pretrain: bool = True,
                 dropout: float = 0.3,
                 in_chs: int = None,
                 separate_inputs: int = None, # separate_inputs defines the number of inputs
                 only_feature_extraction: bool = False, # If true classification head will be ignored. Will work as a feature extractor
                ):  

        super(EnsembleResNet18Ft512_EfficientNetB2SFt1408V2, self).__init__()
        
        self.separate_inputs = separate_inputs
        self.only_feature_extraction = only_feature_extraction

        model1 = base_models('resnet18', pretrain=pretrain, num_classes=num_classes, in_chs=in_chs)
        model2 = base_models('efficientnet_v2_s', pretrain=pretrain, num_classes=num_classes, in_chs=in_chs)

        self.ens_model1 = FeatureEnsemble2models(model1, model2, trim1=2, trim2=2)  # clip classification head

        # Create a list of models for separate inputs if separate_inputs is specified
        if self.separate_inputs is not None:
            self.ensemble_models = nn.ModuleList([deepcopy(self.ens_model1) for _ in range(self.separate_inputs)])

        if not self.only_feature_extraction:
            self.classification = ClassificationHead(num_classes=num_classes,
                                                     out_channels=out_channels,
                                                     dropout=dropout)

    def forward(self, x):

        if self.separate_inputs is not None:

            # Ensure input data has self.separate_inputs no. of channels
            if x.shape[1] < self.separate_inputs:
                raise ValueError(f"Can't split. Input data has {x.shape[1]} channels whereas separate_inputs parameter is {self.separate_inputs}. \
Check the separate_inputs parameter in the config file.")
            
            features_list = []

            # Loop over each input channel, process it, and store the features
            for i in range(self.separate_inputs):
                # Separate the i-th input (single channel)
                xi = x[:, i:i + 1, :, :]  # extract ith channel

                # Convert to 3 channels by repeating or concatenating along the channel dimension
                xi_3ch = torch.cat([xi, xi, xi], dim=1)

                # Get features from the i-th ensemble model
                features_i = self.ensemble_models[i](xi_3ch)

                # Collect features
                features_list.append(features_i)

            # Concatenate features along the channel dimension
            features = torch.cat(features_list, dim=1)

        else:
            features = self.ens_model1(x)

        # Feature extractor or classifier
        if self.only_feature_extraction:
            return features
        else:
            # Pass the features through the classification head
            out = self.classification(features)
            return out

# ===========================================================================================================

class EnsembleResNet50_512x28PscseEfficientNetB2Pscse384X28(nn.Module):

    """ Ensembles ResNet50Pscse_512x28x28 and EfficientNetB2LPscse_384x28x28 """
    
    def __init__(self, 
                 num_classes: int,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 pretrain: bool = True,
                 dropout: float = 0.3,
                 activation: str = 'leakyrelu',
                 reduction: int = 16,
                 separate_inputs: int = None):  # separate_inputs defines the number of inputs

        super(EnsembleResNet50_512x28PscseEfficientNetB2Pscse384X28, self).__init__()
        
        self.separate_inputs = separate_inputs

        model1 = ResNet50Pscse_512x28x28(num_classes, out_channels, pretrain, dropout, activation, reduction)
        model2 = EfficientNetB2LPscse_384x28x28(num_classes, out_channels, pretrain, dropout, activation, reduction)

        self.ens_model1 = FeatureEnsemble2models(model1, model2, trim1=1, trim2=1)  # clip classification head

        # Create a list of models for separate inputs if separate_inputs is specified
        if self.separate_inputs is not None:
            self.ensemble_models = nn.ModuleList([deepcopy(self.ens_model1) for _ in range(self.separate_inputs)])

        self.classification = ClassificationHead(num_classes=num_classes,
                                                 out_channels=out_channels,
                                                 dropout=dropout)

    def forward(self, x):

        if self.separate_inputs is not None:

            # Ensure input data has self.separate_inputs no. of channels
            if x.shape[1] < self.separate_inputs:
                raise ValueError(f"Can't split. Input data has {x.shape[1]} channels whereas separate_inputs parameter is {self.separate_inputs}. \
Check the separate_inputs parameter in the config file.")
            
            features_list = []

            # Loop over each input channel, process it, and store the features
            for i in range(self.separate_inputs):
                # Separate the i-th input (single channel)
                xi = x[:, i:i + 1, :, :]  # extract ith channel

                # Convert to 3 channels by repeating or concatenating along the channel dimension
                xi_3ch = torch.cat([xi, xi, xi], dim=1)

                # Get features from the i-th ensemble model
                features_i = self.ensemble_models[i](xi_3ch)

                # Collect features
                features_list.append(features_i)

            # Concatenate features along the channel dimension
            features = torch.cat(features_list, dim=1)

        else:
            features = self.ens_model1(x)

        # Pass the features through the classification head
        out = self.classification(features)

        return out

""" Example
if __name__ == "__main__":
    inp=torch.rand(1, 3, 224, 224)
    num_classes=2
    out_channels=[6144, 512, 256]
    pretrain = True
    dropout=0.3
    activation='leakyrelu'
    reduction=16
    separate_inputs = 3
    
    model = EnsembleResNet50_512x28PscseEfficientNetB2Pscse384X28(num_classes, out_channels, pretrain, dropout, activation, reduction, separate_inputs)
    
    out = model(inp)
    
    print(out.shape)
"""
# ===========================================================================================================
class BaseModelSepIn(nn.Module):
    """This model takes a base model and creates multiple copies based on the number of channels in the input image. 
    During training, it first splits all channels and uses each copy to train on each split. Finally, it concatenates 
    all channels and sends them to the classification head."""
    
    def __init__(self, 
                 name, 
                 num_classes:int=None, 
                 out_channels:list=None, 
                 pretrain:bool=True, 
                 dropout: float = 0.3,
                 in_chs:int=None,
                 cls_activation: str = 'leakyrelu',# for instance [1024, 512, 256]. Used in classification head
                 separate_inputs:int=None): # separate_inputs defines the number of inputs
        
        super(BaseModelSepIn, self).__init__()

        self.separate_inputs = separate_inputs

        # Load the base model
        base_model = base_models_features_only(name, pretrain, num_classes, in_chs)
        
        # Create a list of models for separate inputs 
        self.ensemble_models = nn.ModuleList([deepcopy(base_model) for _ in range(self.separate_inputs)])

        self.classification = ClassificationHead(num_classes=num_classes,
                                                 out_channels=out_channels,
                                                 activation=cls_activation,
                                                 dropout=dropout)

    def forward(self, x):
        if self.separate_inputs is not None:

            # Ensure input data has self.separate_inputs no. of channels
            if x.shape[1] < self.separate_inputs:
                raise ValueError(f"Can't split. Input data has {x.shape[1]} channels whereas separate_inputs parameter is {self.separate_inputs}. \
Check the separate_inputs parameter in the config file.")
            
            features_list = []

            # Loop over each input channel, process it, and store the features
            for i in range(self.separate_inputs):
                # Separate the i-th input (single channel)
                xi = x[:, i:i + 1, :, :]  # extract ith channel

                # Convert to 3 channels by repeating or concatenating along the channel dimension
                xi_3ch = torch.cat([xi, xi, xi], dim=1)

                # Get features from the i-th ensemble model
                features_i = self.ensemble_models[i](xi_3ch)

                # Collect featureas
                features_list.append(features_i)

            # Concatenate features along the channel dimension
            features = torch.cat(features_list, dim=1)

        else:
            # features = self.ens_model1(x)
            raise ValueError("Input parameter `separate_inputs` can't be None for this model. Check the config file.")

        # Pass the features through the classification head
        out = self.classification(features)

        return out

# if __name__ == "__main__":
#     name = 'convnext_large'
#     inp=torch.rand(1, 3, 224, 224)
#     num_classes=2
#     in_chs=3
#     out_channels=[4608, 512, 256]
#     pretrain = True
#     dropout=0.3
#     activation='leakyrelu'
#     separate_inputs = 3

#     model = BaseModelSepIn(name, num_classes, out_channels, pretrain, dropout, in_chs, activation, separate_inputs)    
    
#     out = model(inp)
    
#     print(out.shape)

# ===========================================================================================================
