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

# ==================================================================================================================
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
        # self.ln = nn.LayerNorm([1792,7,7]) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PAY ATTENTION !!!!
        self.ln = nn.LayerNorm([5376,7,7]) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PAY ATTENTION !!!! If separate inputs

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (batch, out_channels x 1 x 1)
        
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
     

# if __name__ == "__main__":
#     image_inpput = torch.rand(4, 3, 224, 224)
#     num_classes = 2
#     out_channels = [5376, 512, 256]
#     pretrain = True
#     dropout = 0.3
#     separate_inputs = 3
#     in_chs = 3
    
#     radiomic_input = torch.randn(4, 1, 10)
#     radiomic_dims = [10, 64, 128, 256, 512]
#     radiomic_activation = 'leakyrelu'
#     radiomic_dropout = 0.3
#     radiomic_attention = "BasicAttention" # define the attention module to be used
    
#     model = RadiomicMLP_EnsembleResNet18Ft512_EfficientNetB2SFt1408(num_classes, out_channels, pretrain, dropout, in_chs, separate_inputs,
#                                                                     radiomic_dims, radiomic_activation, 
#                                                                     radiomic_attention, radiomic_dropout)
    
#     out = model(image_inpput, radiomic_input)
#     print(out.shape) # torch.Size([4, 2])


# ==================================================================================================================
"""
Author@ Mrinal Kanti Dhar
December 04, 2024
"""

class RadiomicMLP_EnsembleResNet18Ft512_EfficientNetB2SFt1408V2(nn.Module):
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

                 ft_normalization:list=[None, None], # either None, "in", "ln", or "bn". 
                 # The 1st normalization is used to normalize the final feature map. The 2nd normalization is used to normalize
                 # the concatenated DL and radiomic flattened features. So, don't use instance normalization for the 2nd normalization.
                 cls_activation:str='leakyrelu', 
                ):  
        super(RadiomicMLP_EnsembleResNet18Ft512_EfficientNetB2SFt1408V2, self).__init__()

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
        
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (batch, out_channels x 1 x 1)
        
        # Find no. of output features using dummy input
        dummy_inp = torch.rand(1, in_chs, 224, 224)
        dummy_out = self.dl_model(dummy_inp) # shape: batch x ch x h x w
        dl_feature_dim = dummy_out.shape[1] # no. of channels

        "Prepare radiomic model"
        # Get attention module
        if radiomic_attention is not None: radiomic_attention = getattr(attention, radiomic_attention) 
        self.radiomic_model = RadiomicMLP(radiomic_dims, radiomic_activation, radiomic_attention, radiomic_dropout)

        "Prepare classification head"
        # The input features of the classification head is dl_feature_dim + radiomic_out_dim
        in_features_for_cl = dummy_out.shape[1] + radiomic_dims[-1]
        cl_dims = [in_features_for_cl] + out_channels # classification dimensions 
        
        # We assume that the features are already flattened
        self.classification = ClassificationHeadWithoutFlatten(num_classes=num_classes,
                                                 out_channels=cl_dims,
                                                 activation=cls_activation,
                                                 dropout=dropout)

        "Prepare normalization - one to normalize DL features, and the other to normalize combined features"
        # Normalize DL features 
        if ft_normalization[0] == None:
            self.norm1 = None
        elif ft_normalization[0] == "in":
            self.norm1 = nn.InstanceNorm2d(dummy_out.shape[1], affine=True) # affine adds learnable scale/shift
        elif ft_normalization[0] == "ln":
            self.norm1 = nn.LayerNorm([dummy_out.shape[1],dummy_out.shape[2],dummy_out.shape[3]]) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PAY ATTENTION !!!! If separate inputs
        elif ft_normalization[0] == "bn":
            self.norm1 = nn.BatchNorm2d(num_features=dummy_out.shape[1])
        else:
            raise ValueError("Wrong keyword for normalization. Permitted keywords are - None, in, ln, and bn.")    

        # Normalize combined features 
        if ft_normalization[1] == None:
            self.norm2 = None
        elif ft_normalization[1] == "ln":
            self.norm2 = nn.LayerNorm(in_features_for_cl) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PAY ATTENTION !!!! If separate inputs
        elif ft_normalization[1] == "bn":
            self.norm2 = nn.BatchNorm1d(num_features=in_features_for_cl) 
        else:
            raise ValueError("Wrong keyword for normalization. Permitted keywords are - None, ln, and bn.")

    def forward(self, data_img, data_radiomic):
        # Extract features
        dl_features = self.dl_model(data_img)
        if self.norm1 is not None: dl_features = self.norm1(dl_features)  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< PAY ATTENTION !!!!
        dl_features = self.avg_pool(dl_features) # Output size: (batch, out_channels x 1 x 1)
        dl_features = dl_features.view(dl_features.size(0), -1) # Shape: (Batch_size, features)

        radiomic_features = self.radiomic_model(data_radiomic) # Shape: (batch_size * num_features, mlp_output_dim)
        radiomic_features = radiomic_features.view(data_radiomic.size(0), -1) # Shape: (Batch_size, features)
        
        # Concatenate DL and radiomic features
        combined_features = torch.cat((dl_features, radiomic_features), dim=1)
        if self.norm2 is not None: combined_features = self.norm2(combined_features)

        # Classification
        output = self.classification(combined_features)

        return output
     
    
# if __name__ == "__main__":
#     image_inpput = torch.rand(4, 3, 224, 224)
#     num_classes = 2
#     out_channels = [5376, 512, 256]
#     pretrain = True
#     dropout = 0.3
#     separate_inputs = 3
#     in_chs = 3
    
#     radiomic_input = torch.randn(4, 1, 10)
#     radiomic_dims = [10, 64, 128, 256, 512]
#     radiomic_activation = 'leakyrelu'
#     radiomic_dropout = 0.3
#     radiomic_attention = "BasicAttention" # define the attention module to be used
    
#     ft_normalization = ["in", "bn"] # either None, "ln", or "bn"
#     cls_activation = "leakyrelu"
    
#     model = RadiomicMLP_EnsembleResNet18Ft512_EfficientNetB2SFt1408V2(num_classes, out_channels, pretrain, dropout, in_chs, separate_inputs,
#                                                                     radiomic_dims, radiomic_activation, radiomic_attention, radiomic_dropout,
#                                                                    ft_normalization, cls_activation)
    
#     out = model(image_inpput, radiomic_input)
#     print(out.shape) # torch.Size([4, 2])


# ==================================================================================================================
"""
Author@ Mrinal Kanti Dhar
January 16, 2025
"""
import sys
sys.path.append("/research/m324371/Project/adnexal/networks/") 
from ensemble_type1 import EnsembleResNet18Ft512_EfficientNetB2SFt1408

class RadiomicMLP_PretrainedEnsembleResNet18Ft512_EfficientNetB2SFt1408(nn.Module):
    def __init__(self,
                 num_classes:int,
                 out_channels:list=None,
                 pretrain:bool=True,
                 pretrained_dir:str=None,  # Path to pretrained model
                 dropout:float=0.3,
                 in_chs:int=None,
                 separate_inputs:int=None,
                 radiomic_dims:list=None, 
                 radiomic_activation:str='leakyrelu', 
                 radiomic_attention=None, 
                 radiomic_dropout:float=None,
                 ft_normalization:list=[None, None],
                 cls_activation:str='leakyrelu'):  
        super(RadiomicMLP_PretrainedEnsembleResNet18Ft512_EfficientNetB2SFt1408, self).__init__()

        # Initialize the deep learning model
        self.dl_model = EnsembleResNet18Ft512_EfficientNetB2SFt1408(num_classes, 
                                                                      out_channels, 
                                                                      pretrain, 
                                                                      dropout, 
                                                                      in_chs, 
                                                                      separate_inputs,)
        
        # Load pretrained parameters if provided
        if pretrained_dir:
            print(f"Loading pretrained model from: {pretrained_dir}")
            checkpoint = torch.load(pretrained_dir)
            # state_dict = torch.load(pretrained_dir, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.dl_model.load_state_dict(checkpoint['state_dict'])
            # self.dl_model.load_state_dict(state_dict)
        
        # Freeze the pretrained model parameters
        for param in self.dl_model.parameters():
            param.requires_grad = False

        # Remove classification layers and keep till flattening layer
        self.dl_model.classification.fc = nn.Identity() # keep till flattening 
        
        # Set model to evaluation mode
        self.dl_model.eval()

        # Global average pooling
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (batch, out_channels x 1 x 1)
        
        # Prepare radiomic model
        if radiomic_attention is not None:
            radiomic_attention = getattr(attention, radiomic_attention)
        self.radiomic_model = RadiomicMLP(radiomic_dims, radiomic_activation, radiomic_attention, radiomic_dropout)

        # Classification head
        dummy_inp = torch.rand(1, in_chs, 224, 224)
        dummy_out = self.dl_model(dummy_inp) # torch.Size([1, 5376])
        dl_feature_dim = dummy_out.shape[1]
        in_features_for_cl = dl_feature_dim + radiomic_dims[-1]
        cl_dims = [in_features_for_cl] + out_channels
        
        self.classification = ClassificationHeadWithoutFlatten(num_classes=num_classes,
                                                               out_channels=cl_dims,
                                                               activation=cls_activation,
                                                               dropout=dropout)

        # Normalization layers
        self.norm1 = self._get_normalization(ft_normalization[0], dummy_out.shape)
        self.norm2 = self._get_normalization(ft_normalization[1], (in_features_for_cl,))
    
    def _get_normalization(self, norm_type, shape):
        if norm_type is None:
            return None
        elif norm_type == "in":
            return nn.InstanceNorm2d(shape[1], affine=True)
        elif norm_type == "ln":
            return nn.LayerNorm(shape)
        elif norm_type == "bn":
            return nn.BatchNorm2d(shape[1]) if len(shape) == 4 else nn.BatchNorm1d(shape[0])
        else:
            raise ValueError("Invalid normalization type.")
    
    def forward(self, data_img, data_radiomic):
        # Extract features
        with torch.no_grad():  # Ensure pretrained model is frozen
            dl_features = self.dl_model(data_img)
            if self.norm1 is not None:
                dl_features = self.norm1(dl_features)
            # dl_features = self.avg_pool(dl_features)
            # dl_features = dl_features.view(dl_features.size(0), -1)

        # Radiomic features
        radiomic_features = self.radiomic_model(data_radiomic)
        radiomic_features = radiomic_features.view(data_radiomic.size(0), -1)
        
        # Concatenate DL and radiomic features
        combined_features = torch.cat((dl_features, radiomic_features), dim=1)
        if self.norm2 is not None:
            combined_features = self.norm2(combined_features)

        # Classification
        output = self.classification(combined_features)
        return output

if __name__ == "__main__":
    image_inpput = torch.rand(4, 3, 224, 224)
    num_classes = 2
    out_channels = [5376, 1024, 512, 256]
    pretrain = True
    dropout = 0.3
    separate_inputs = 3
    in_chs = 3
    pretrained_dir='/research/m324371/Project/adnexal/results/EnsembleResNet18Ft512_EfficientNetB2SFt1408_2024-10-25_17-31-23/checkpoints/EnsembleResNet18Ft512_EfficientNetB2SFt1408_2024-10-25_19-15-38/best_model.pth'
    
    radiomic_input = torch.randn(4, 1, 10)
    radiomic_dims = [10, 64, 128, 256, 512]
    radiomic_activation = 'leakyrelu'
    radiomic_dropout = 0.3
    radiomic_attention = "BasicAttention" # define the attention module to be used
    
    ft_normalization = [None, "ln"] # either None, "ln", or "bn"
    cls_activation = "leakyrelu"
    
    model = RadiomicMLP_PretrainedEnsembleResNet18Ft512_EfficientNetB2SFt1408(num_classes, out_channels, pretrain, pretrained_dir, dropout, in_chs, separate_inputs,
                                                                    radiomic_dims, radiomic_activation, radiomic_attention, radiomic_dropout,
                                                                   ft_normalization, cls_activation)
    
    out = model(image_inpput, radiomic_input)
    print(out.shape) # torch.Size([4, 2])

# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================




