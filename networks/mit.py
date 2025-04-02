"""
Author@ Mrinal Kanti Dhar
November 14, 2024
"""

import sys
sys.path.append("/research/m324371/Project/adnexal/utils/")

import torch
import torch.nn as nn
from copy import deepcopy
from classification_head import ClassificationHead
from base_models_features_collection import base_models_features_only

class MiT(nn.Module):
    """This model takes a base model and creates multiple copies based on the number of channels in the input image. 
    During training, it first splits all channels and uses each copy to train on each split. Finally, it concatenates 
    all channels and sends them to the classification head."""
    
    def __init__(self, 
                 name, 
                 num_classes:int=None, 
                 out_channels:list=None, 
                 pretrain:bool=True, # currently only supports pretrained model
                 dropout: float = 0.3,
                 in_chs:int=None,
                 cls_activation: str = 'leakyrelu',# for instance [1024, 512, 256]. Used in classification head
                 separate_inputs:int=None): # separate_inputs defines the number of inputs
        
        super(MiT, self).__init__()

        self.separate_inputs = separate_inputs

        # Load the base model
        self.base_model = base_models_features_only(name, pretrain, num_classes, in_chs)

        # Create a list of models for separate inputs if separate_inputs is specified
        if self.separate_inputs is not None:
            self.ensemble_models = nn.ModuleList([deepcopy(self.base_model) for _ in range(self.separate_inputs)])

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
                features_i = self.ensemble_models[i](xi_3ch).last_hidden_state # Shape: (batch_size, 512, 7, 7) for (224,224)

                # Collect featureas
                features_list.append(features_i)

            # Concatenate features along the channel dimension
            features = torch.cat(features_list, dim=1)

        else:
            features = self.base_model(x).last_hidden_state # Shape: (batch_size, 512, 7, 7) for (224,224)

        # Pass the features through the classification head
        out = self.classification(features)

        return out
