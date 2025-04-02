#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Author@ Mrinal Kanti Dhar
October 30, 2024
"""

import sys
sys.path.append("/research/m324371/Project/adnexal/utils/")

import torch
import torch.nn as nn

from classification_head import ClassificationHeadWithoutFlatten
from feature_ensemble_2models import FeatureEnsemble2models
import attention

from ensemble_type1 import EnsembleResNet18Ft512_EfficientNetB2SFt1408V2
from radiomic_nets import RadiomicMLP


# In[10]:


class RadiomicMLP_EnsembleResNet18Ft512_EfficientNetB2SFt1408(nn.Module):
    """ Ensembles Radiomic features from RadiomicMLP with EnsembleResNet18Ft512_EfficientNetB2SFt1408 
        EnsembleResNet18Ft512_EfficientNetB2SFt1408 ensembles ResNet18 with 512 features and EfficientNetB2 with 1408 features
    """
    def __init__(self,
                 num_classes:int,
                 out_channels:list=None,  # for instance [1024, 512, 256]. Used in classification head
                 pretrain:bool=True,
                 dropout:float=0.3,
                 in_chs:int=None,
                 separate_inputs:int=None, # separate_inputs defines the number of inputs

                 radiomic_dims:list=None, 
                 radiomic_activation:str='leakyrelu', 
                 radiomic_attention=None, 
                 radiomic_dropout:float=None,
                ):  
        super(RadiomicMLP_EnsembleResNet18Ft512_EfficientNetB2SFt1408, self).__init__()

        "Prepare deep learning model (up to feature extraction)"
        # Initialize deep learning model
        self.dl_model = EnsembleResNet18Ft512_EfficientNetB2SFt1408V2(num_classes, out_channels, pretrain, dropout, in_chs, separate_inputs, only_feature_extraction=True) 
        # Trim classification head
        # self.dl_feature_extractor = nn.Sequential(*list(dl_model.children()))[:-1] 

        # modules = []
        # for layer in dl_model.children():
        #     if isinstance(layer, nn.ModuleList):
        #         modules.extend(layer)  # Flatten out ModuleList layers
        #     else:
        #         modules.append(layer)
        # self.dl_feature_extractor = nn.Sequential(*modules[:-1])  # Exclude the classification head
        
        # Do batch normalization
        # self.bn = nn.BatchNorm2d(num_features=1792)
        self.ln = nn.LayerNorm([1792,7,7]) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PAY ATTENTION !!!!

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (out_channels x 1 x 1)
        
        # Find no. of output features using dummy input
        dummy_inp = torch.rand(1, in_chs, 224, 224)
        dummy_out = self.dl_model(dummy_inp)
        dl_feature_dim = dummy_out.shape[1] # no. of channels
        
        "Prepare radiomic model"
        # Get attention module
        if radiomic_attention is not None: radiomic_attention = getattr(attention, radiomic_attention) 
        self.radiomic_model = RadiomicMLP(radiomic_dims, radiomic_activation, radiomic_attention, radiomic_dropout)

        "Prepare classification head"
        # The input features of the classification head is dl_feature_dim + radiomic_out_dim
        in_features_for_cl = dl_feature_dim + radiomic_dims[-1]
        cl_dims = [in_features_for_cl] + out_channels # classification dimensions
        
        # We assume that the features are already flattened
        self.classification = ClassificationHeadWithoutFlatten(num_classes=num_classes,
                                                 out_channels=cl_dims,
                                                 dropout=dropout)

    def forward(self, data_img, data_radiomic):
        # Extract features
        dl_features = self.dl_model(data_img)
        dl_features = self.ln(dl_features)  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PAY ATTENTION !!!!
        dl_features = self.avg_pool(dl_features)
        dl_features = dl_features.view(dl_features.size(0), -1) # Shape: (Batch_size, features)

        radiomic_features = self.radiomic_model(data_radiomic) # Shape: (batch_size * num_features, mlp_output_dim)
        radiomic_features = radiomic_features.view(data_radiomic.size(0), -1) # Shape: (Batch_size, features)
        
        # Concatenate DL and radiomic features
        combined_features = torch.cat((dl_features, radiomic_features), dim=1)

        # Classification
        output = self.classification(combined_features)

        return output
     
    


# In[11]:


if __name__ == "__main__":
    image_inpput = torch.rand(4, 3, 224, 224)
    num_classes = 2
    out_channels = [5376, 512, 256]
    pretrain = True
    dropout = 0.3
    separate_inputs = 3
    in_chs = 3
    
    radiomic_input = torch.randn(4, 1, 10)
    radiomic_dims = [10, 64, 128, 256, 512]
    radiomic_activation = 'leakyrelu'
    radiomic_dropout = 0.3
    radiomic_attention = "BasicAttention" # define the attention module to be used
    
    model = RadiomicMLP_EnsembleResNet18Ft512_EfficientNetB2SFt1408(num_classes, out_channels, pretrain, dropout, in_chs, separate_inputs,
                                                                    radiomic_dims, radiomic_activation, 
                                                                    radiomic_attention, radiomic_dropout)
    
    out = model(image_inpput, radiomic_input)
    print(out.shape) # torch.Size([4, 2])


# In[ ]:




