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
from operator import itemgetter

from pscse_cab import PscSEWithCAB
from classification_head import ClassificationHead
from feature_ensemble_2models import FeatureEnsemble2models
from feature_ensemble import FeatureEnsembleNModelsNoTrim

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
    all channels and sends them to the classification head. It does not support separate_inputs as None."""
    
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
class EnsembleModelsV1(nn.Module):
    """ Ensembles models available in base_models_features_collection.py"""
    
    def __init__(self, 
                 names: list, # e.g. ['resnet18', 'efficientnet_v2_s', 'mobilenet_v3_large'] 
                 num_classes: int,
                 out_channels: list = None, # for instance [1024, 512, 256]. Used in classification head
                 pretrain: bool = True,
                 dropout: float = 0.3,
                 in_chs: int = None,
                 cls_activation: str = 'leakyrelu', # activation in classification head 
                 input_seq: list = [[0,0,0], [1,1,1], [2,2,2], [0,1,2]],
                 # input_seq is a nested list. Consider, input_seq = [[0,0,0], [1,1,1], [2,2,2], [0,1,2]]
                 # It means that it expects four copies of the ensemble model. 0,1,2 are channel indices of 
                 # the original input image coming from the dataloader. So, the 1st ensemble model will 
                 # take the input image consisting of 3 copies of the 0th-ch of the original image. Similarly,
                 # 2nd and 3rd ensemble models takes 3 copies of 1st- and 2nd-ch of the original image, respectively.
                 # The 4th ensemble model takes input image that consists of 0-th, 1st-, and 2nd-ch of the 
                 # original image. 
                
                ):  

        super(EnsembleModelsV1, self).__init__()

        self.input_seq = input_seq

        # Load models
        models = [base_models_features_only(name, pretrain, num_classes, in_chs) for name in names] 

        self.ens_model1 = FeatureEnsembleNModelsNoTrim(models)  # ensemble features

        # Create a list of models if length of input_seq is greater than 1
        self.num_copies = len(self.input_seq) # similar to separate_inputs
        if self.num_copies > 1:
            self.ensemble_models = nn.ModuleList([deepcopy(self.ens_model1) for _ in range(self.num_copies)])

        # Classification head
        self.classification = ClassificationHead(num_classes=num_classes,
                                                 out_channels=out_channels,
                                                 activation=cls_activation,
                                                 dropout=dropout)

    def forward(self, x):

        if self.num_copies > 1: # multiple copies of ensemble model 
            
            features_list = []

            # Loop over each input channel, process it, and store the features
            for i in range(self.num_copies):
                # Create input image for the i-th ensemble model based on channel indices indicated by input_seq[i]
                xi = x[:, self.input_seq[i], :, :]

                # Get features from the i-th ensemble model
                features_i = self.ensemble_models[i](xi)

                # Collect features
                features_list.append(features_i)

            # Concatenate features along the channel dimension
            features = torch.cat(features_list, dim=1)

        else:
            features = self.ens_model1(x)

        # Pass the features through the classification head
        out = self.classification(features)

        return out

# if __name__ == "__main__":
#     inp=torch.rand(1, 3, 224, 224)
    
#     names = ['resnet18', 'efficientnet_v2_s', 'mobilenet_v3_large'] 
#     num_classes=2
#     out_channels=[11008, 4608, 512, 256]
#     pretrain = True
#     dropout=0.3
#     in_chs=3
#     cls_activation='leakyrelu'
#     input_seq: list = [[0,0,0], [1,1,1], [2,2,2], [0,1,2]]
    
#     model = EnsembleModelsV1(names, num_classes, out_channels, pretrain, dropout, in_chs, cls_activation, input_seq)
    
#     out = model(inp)
    
#     print(out.shape) # torch.Size([1, 2])


# ===========================================================================================================
class EnsembleModelsV2(nn.Module):
    """ Ensembles models available in base_models_features_collection.py.
        V2 offers more flexibility than V1. In V2, we can create single models, ensemble models, even multiple
        ensemble models. We also can control which image will go to which model. For details, read the description
        for input_seq and preensemble below.
    """
    
    def __init__(self, 
                 names: list, # e.g. ['resnet18', 'efficientnet_v2_s', 'mobilenet_v3_large'] 
                 num_classes: int,
                 out_channels: list = None, # for instance [1024, 512, 256]. Used in classification head
                 pretrain: bool = True, # used in base_models_features_only
                 dropout: float = 0.3, # used in classification head
                 in_chs: int = 3, # used in base_models_features_only
                 cls_activation: str = 'leakyrelu', # activation in classification head 
                 input_seq: list = None, # e.g. [[0,0,0], [1,1,1], [2,2,2], [0,1,2]],
                 preensemble: list = None, # e.g. [[0], [1], [0,1], [2]],
                 
                 # input_seq is a nested list. Consider, input_seq = [[0,0,0], [1,1,1], [2,2,2], [0,1,2]]
                 # It means that it expects four copies of the ensemble model. 0,1,2 are channel indices of 
                 # the original input image coming from the dataloader. So, the 1st ensemble model will 
                 # take the input image consisting of 3 copies of the 0th-ch of the original image. Similarly,
                 # 2nd and 3rd ensemble models takes 3 copies of 1st- and 2nd-ch of the original image, respectively.
                 # The 4th ensemble model takes input image that consists of 0-th, 1st-, and 2nd-ch of the 
                 # original image. 

                 # preensemble is a nested list. Consider, model names are ['resnet18', 'efficientnet_v2_s', 'mobilenet_v3_large'], 
                 # and input_seq is [[0,0,0], [1,1,1], [2,2,2], [0,1,2]]. If the preensemble is [[0], [1], [0,1], [2]],
                 # then for [0], it will call resnet18 and pass the image [0,0,0] to it. For [1], image [1,1,1] will be fed to 
                 # efficientnet_v2_s. For [0,1], it will ensemble resnet18 and efficientnet_v2_s, and then image [2,2,2] will
                 # be fed to the ensembled model. For [2], image [0,1,2] will be fed to mobilenet_v3_large. 
                
                ):  

        super(EnsembleModelsV2, self).__init__()

        self.input_seq = input_seq

        # Load models
        models_ = [base_models_features_only(name, pretrain, num_classes, in_chs) for name in names]

        if preensemble is not None:
            self.models = []
            for ind in preensemble:
                if len(ind) == 1: selected_models = [models_[ind[0]]] # handle single model case explicitly     
                else: selected_models = list(itemgetter(*ind)(models_)) # handle multiple model selection using itemgetter
                    
                # Pass the selected models to FeatureEnsembleNModelsNoTrim
                self.models.append(FeatureEnsembleNModelsNoTrim(selected_models))
        else:
            self.models = models_

        self.models = nn.ModuleList(self.models)
             
        # Classification head
        self.classification = ClassificationHead(num_classes=num_classes,
                                                 out_channels=out_channels,
                                                 activation=cls_activation,
                                                 dropout=dropout)

    def forward(self, x):

        if len(self.input_seq) != len(self.models):
            raise ValueError(f"Length of input_seq ({len(self.input_seq)}) is not the same as the number of models ({len(self.models)}).\
            They should have the same length.")

        features_list = []
        for i in range(len(self.models)):
            # Create input image for the i-th model based on channel indices indicated by input_seq[i]
            xi = x[:, self.input_seq[i], :, :]

            # Get features from the i-th model
            features_i = self.models[i](xi)

            # Collect features
            features_list.append(features_i)

        # Concatenate features along the channel dimension
        features = torch.cat(features_list, dim=1)

        # Pass the features through the classification head
        out = self.classification(features)

        return out


# if __name__ == "__main__":
#     inp=torch.rand(1, 3, 224, 224)
    
#     names = ['resnet18', 'efficientnet_v2_s', 'mobilenet_v3_large'] 
#     num_classes=2
#     out_channels=[3264, 1024, 512, 256]
#     pretrain = True
#     dropout=0.3
#     in_chs=3
#     cls_activation='leakyrelu'
#     input_seq = [[0,0,0], [1,1,1], [0,1,2]]
#     preensemble = [[0], [0,1], [2]]
    
    
#     model = EnsembleModelsV2(names, num_classes, out_channels, pretrain, dropout, in_chs, cls_activation, input_seq, preensemble)
    
#     out = model(inp)
    
#     print(out.shape) # torch.Size([1, 2])



# ===========================================================================================================

class EnsembleResNet18Ft512_MBV3LFt960(nn.Module):
    """ Ensembles ResNet18 with 512 features and MobileNetV3Large with 960 features """
    def __init__(self, 
                 num_classes: int,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 pretrain: bool = True,
                 dropout: float = 0.3,
                 in_chs: int = None,
                 cls_activation: str = 'leakyrelu',
                 separate_inputs: int = None):  # separate_inputs defines the number of inputs

        super(EnsembleResNet18Ft512_MBV3LFt960, self).__init__()

        self.separate_inputs = separate_inputs

        model1 = base_models('resnet18', pretrain=pretrain, num_classes=num_classes, in_chs=in_chs)
        model2 = base_models('mobilenet_v3_large', pretrain=pretrain, num_classes=num_classes, in_chs=in_chs)

        self.ens_model1 = FeatureEnsemble2models(model1, model2, trim1=2, trim2=2)  # clip classification head

        # Create a list of models for separate inputs if separate_inputs is specified
        if self.separate_inputs is not None:
            self.ensemble_models = nn.ModuleList([deepcopy(self.ens_model1) for _ in range(self.separate_inputs)])

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

                # Collect features
                features_list.append(features_i)

            # Concatenate features along the channel dimension
            features = torch.cat(features_list, dim=1)

        else:
            features = self.ens_model1(x)

        # Pass the features through the classification head
        out = self.classification(features)

        return out

# if __name__ == "__main__":
#     inp=torch.rand(1, 3, 224, 224)
#     num_classes=2
#     out_channels=[4416, 1024, 512, 256]
#     pretrain = True
#     dropout=0.3
#     in_chs=3
#     cls_activation='leakyrelu'
#     separate_inputs = 3


#     model = EnsembleResNet18Ft512_MBV3LFt960(num_classes, out_channels, pretrain, dropout, in_chs, cls_activation, separate_inputs)
    
#     out = model(inp)
    
#     print(out.shape) # torch.Size([1, 2])


# ===========================================================================================================

class EnsembleEfficientNetB2SFt1408_MBV3LFt960(nn.Module):
    """ Ensembles EfficientNetB2 with 1408 features and MobileNetV3Large with 960 features """
    def __init__(self, 
                 num_classes: int,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 pretrain: bool = True,
                 dropout: float = 0.3,
                 in_chs: int = None,
                 cls_activation: str = 'leakyrelu',
                 separate_inputs: int = None):  # separate_inputs defines the number of inputs

        super(EnsembleEfficientNetB2SFt1408_MBV3LFt960, self).__init__()

        self.separate_inputs = separate_inputs

        model1 = base_models('efficientnet_v2_s', pretrain=pretrain, num_classes=num_classes, in_chs=in_chs)
        model2 = base_models('mobilenet_v3_large', pretrain=pretrain, num_classes=num_classes, in_chs=in_chs)

        self.ens_model1 = FeatureEnsemble2models(model1, model2, trim1=2, trim2=2)  # clip classification head

        # Create a list of models for separate inputs if separate_inputs is specified
        if self.separate_inputs is not None:
            self.ensemble_models = nn.ModuleList([deepcopy(self.ens_model1) for _ in range(self.separate_inputs)])

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

                # Collect features
                features_list.append(features_i)

            # Concatenate features along the channel dimension
            features = torch.cat(features_list, dim=1)

        else:
            features = self.ens_model1(x)

        # Pass the features through the classification head
        out = self.classification(features)

        return out

# if __name__ == "__main__":
#     inp=torch.rand(1, 3, 224, 224)
#     num_classes=2
#     out_channels=[6720, 1024, 512, 256]
#     pretrain = True
#     dropout=0.3
#     in_chs=3
#     cls_activation='leakyrelu'
#     separate_inputs = 3


#     model = EnsembleEfficientNetB2SFt1408_MBV3LFt960(num_classes, out_channels, pretrain, dropout, in_chs, cls_activation, separate_inputs)
    
#     out = model(inp)
    
#     print(out.shape) # torch.Size([1, 2])

# ===========================================================================================================
class TwoPlusOneEnsemble(nn.Module):
    """ It takes three models as input. The first two models are ensembled and handle each channel 
        of the input image individually. The third model handles all channels together. 
    """
    def __init__(self, 
                 names: list, # e.g. ["resnet18", "efficientnet_v2_s", "mobilenet_v3_large"]
                 num_classes: int,
                 out_channels: list = None,  # for instance [1024, 512, 256]. Used in classification head
                 pretrain: bool = True,
                 dropout: float = 0.3,
                 in_chs: int = None,
                 cls_activation: str = 'leakyrelu',
                 separate_inputs: int = None):  # separate_inputs defines the number of inputs

        super(TwoPlusOneEnsemble, self).__init__()

        self.separate_inputs = separate_inputs

        model1 = base_models_features_only(names[0], pretrain, num_classes, in_chs)
        model2 = base_models_features_only(names[1], pretrain, num_classes, in_chs)
        self.model3 = base_models_features_only(names[2], pretrain, num_classes, in_chs)
        
        self.ens_model1 = FeatureEnsembleNModelsNoTrim([model1, model2])  # clip classification head

        # Create a list of models for separate inputs if separate_inputs is specified
        if self.separate_inputs is not None:
            self.ensemble_models = nn.ModuleList([deepcopy(self.ens_model1) for _ in range(self.separate_inputs)])

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

                # Collect features
                features_list.append(features_i)

            # Train the model using all channels
            features_all_chs = self.model3(x)
            features_list.append(features_all_chs)
            
            # Concatenate features along the channel dimension
            features = torch.cat(features_list, dim=1)

        else:
            features = self.ens_model1(x)

        # Pass the features through the classification head
        out = self.classification(features)

        return out



# ===========================================================================================================