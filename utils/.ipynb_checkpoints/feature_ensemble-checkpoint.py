import sys
sys.path.append("/research/m324371/Project/adnexal/networks/")

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================================================================
# class FeatureEnsembleNModelsNoTrim(nn.Module):
#     """Ensemble feature maps of N models. Assumes the classification heads
#     are already removed from the models, so no trimming operation is done here.

#     Args:
#         - models: A list of models to be ensembled

#     Returns:
#         - combined_features: A feature map combining all input models
#     """
    
#     def __init__(self, models):
#         super(FeatureEnsembleNModelsNoTrim, self).__init__()
        
#         self.models = nn.ModuleList(models)

#     def forward(self, x):
#         features_list = []
        
#         # Extract features from each model
#         for model in self.models:
#             features = model(x)
#             features_list.append(features)
        
#         if len(features_list) > 1:
#             # Resize feature maps if needed to match the spatial dimensions of the first model's output
#             base_shape = features_list[0].shape[2:]
#             for i in range(1, len(features_list)):
#                 if features_list[i].shape[2:] != base_shape:
#                     features_list[i] = F.interpolate(features_list[i], size=base_shape, mode='bilinear', align_corners=False)

#         # Concatenate the feature maps along the channel dimension
#         combined_features = torch.cat(features_list, dim=1)

#         return combined_features

class FeatureEnsembleNModelsNoTrim(nn.Module):
    def __init__(self, models):
        super(FeatureEnsembleNModelsNoTrim, self).__init__()
        
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # If there's only one model, directly return its output
        if len(self.models) == 1:
            return self.models[0](x)

        features_list = []
        for model in self.models:
            features = model(x)
            features_list.append(features)

        # Resize feature maps if needed
        base_shape = features_list[0].shape[2:]
        for i in range(1, len(features_list)):
            if features_list[i].shape[2:] != base_shape:
                features_list[i] = F.interpolate(features_list[i], size=base_shape, mode='bilinear', align_corners=False)

        # Concatenate the feature maps along the channel dimension
        combined_features = torch.cat(features_list, dim=1)
        return combined_features


# if __name__ == '__main__':
#     # Example models 
#     model1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU())
#     model2 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU())
#     model3 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU())
    
#     # List of models
#     models = [model1, model2, model3]
    
#     # Initialize the ensemble class
#     ensemble_model = FeatureEnsembleNModelsNoTrim(models)
    
#     # Example input tensor (batch size, channels, height, width)
#     input_tensor = torch.randn(1, 3, 64, 64)
    
#     # Forward pass through the ensemble
#     output = ensemble_model(input_tensor)
    
#     # Print the output shape
#     print("Output shape:", output.shape)



# ================================================================================================================
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

        self.trimmed_model1 = nn.Sequential(*list(model1.children()))[:-trim1] # trim first model
        self.trimmed_model2 = nn.Sequential(*list(model2.children()))[:-trim2] # trim second model

    def forward(self, x):

        features1 = self.trimmed_model1(x)
        features2 = self.trimmed_model2(x)

        # Ensure both feature maps have the same spatial dimensions
        if features1.shape[2:] != features2.shape[2:]:
            # Resize feature2's output to match feature1's spatial dimensions
            features2 = F.interpolate(features2, size=features1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate the feature maps along the channel dimension
        combined_features = torch.cat((features1, features2), dim=1) 
        
        return combined_features

# if __name__ == "__main__":
#     from res50pscse_512x28x28 import ResNet50Pscse_512x28x28
#     from enetb2lpscse_384x28x28 import EfficientNetB2LPscse_384x28x28

#     inp=torch.rand(1, 3, 224, 224)
#     num_classes=2
#     out_channels=[1024, 512, 256]
#     pretrain=True
#     dropout=0.3
#     activation='leakyrelu'
#     reduction=16
    
#     model1 = ResNet50Pscse_512x28x28(num_classes, out_channels, pretrain, dropout, activation, reduction) 
#     model2 = EfficientNetB2LPscse_384x28x28(num_classes, out_channels, pretrain, dropout, activation, reduction)
    
#     feature_ensembled_model = FeatureEnsemble2models(model1, model2, trim1=1, trim2=1) # trim classification head
#     out = feature_ensembled_model(inp)
    
#     print(out.shape) # torch.Size([1, 2048, 7, 7])



# ================================================================================================================
class FeatureEnsemble3models(nn.Module):
    """ Ensemble feature maps of three models 
    Args:
        - model1: First model
        - model2: Second model
        - model3: Third model
        - trim1: (int) No. of blocks to be trimmed from the end of model1, usually used to trim classification head
        - trim2: (int) No. of blocks to be trimmed from the end of model2, usually used to trim classification head
        - trim3: (int) No. of blocks to be trimmed from the end of model3, usually used to trim classification head

    Returns:
        - combined_features: A feature map combining model1, model2, and model3
    """
    
    def __init__(self, model1, model2, model3, trim1:int, trim2:int, trim3:int):
        super(FeatureEnsemble3models, self).__init__()

        self.trimmed_model1 = nn.Sequential(*list(model1.children()))[:-trim1] # trim first model
        self.trimmed_model2 = nn.Sequential(*list(model2.children()))[:-trim2] # trim second model
        self.trimmed_model3 = nn.Sequential(*list(model3.children()))[:-trim3] # trim third model

    def forward(self, x):

        features1 = self.trimmed_model1(x)
        features2 = self.trimmed_model2(x)
        features3 = self.trimmed_model3(x)

        # Ensure all feature maps have the same spatial dimensions
        if features1.shape[2:] != features2.shape[2:]:
            # Resize feature2's output to match feature1's spatial dimensions
            features2 = F.interpolate(features2, size=features1.shape[2:], mode='bilinear', align_corners=False)
            
        if features1.shape[2:] != features3.shape[2:]:    
            # Resize feature3's output to match feature1's spatial dimensions
            features3 = F.interpolate(features3, size=features1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate the feature maps along the channel dimension
        combined_features = torch.cat((features1, features2, features3), dim=1) 
        
        return combined_features




# ================================================================================================================
class FeatureEnsemble3modelsNoTrim(nn.Module):
    """ Ensemble feature maps of three models. However, it assumes that the classification heads
    are already removed from the models. So, no trimming operation will be done here. 
     
    Args:
        - model1: First model
        - model2: Second model
        - model3: Third model

    Returns:
        - combined_features: A feature map combining model1, model2, and model3
    """
    
    def __init__(self, model1, model2, model3):
        super(FeatureEnsemble3modelsNoTrim, self).__init__()
        
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):

        features1 = self.model1(x)
        features2 = self.model2(x)
        features3 = self.model3(x)

        # Ensure all feature maps have the same spatial dimensions
        if features1.shape[2:] != features2.shape[2:]:
            # Resize feature2's output to match feature1's spatial dimensions
            features2 = F.interpolate(features2, size=features1.shape[2:], mode='bilinear', align_corners=False)
            
        if features1.shape[2:] != features3.shape[2:]:    
            # Resize feature3's output to match feature1's spatial dimensions
            features3 = F.interpolate(features3, size=features1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate the feature maps along the channel dimension
        combined_features = torch.cat((features1, features2, features3), dim=1) 
        
        return combined_features
        
        
# =====================================================================================================
