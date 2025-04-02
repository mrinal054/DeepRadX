import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import transformers

def base_models_features_only(name, pretrain:bool=True, num_classes:int=None, in_chs:int=None):
    """
    Args:
        - name: Name of the pretrained model
        - pretrain: Whether to load pretrained weights
        - num_classes: No. of classes
        - in_chs: No. of input channels

    Return:
        - Pretrained model features only
    """

    pretrained_names = ['resnet18', 'resnet50', 'efficientnet_v2_s', 'efficientnet_v2_l', 'mobilenet_v3_large', 'vgg19_bn', 'maxvit_t', 'convnext_base', 'convnext_large', 'swin_v2_b', 'mit_b3']
    
    if name == 'resnet18':
        base_model = models.resnet18(weights="DEFAULT") if pretrain else models.resnet18(weights=None)
        base_model = nn.Sequential(*list(base_model.children()))[:-2] # remove avg pooling and classification layers

        # Modify the first convolutional layer to accept a single channel
        if in_chs == 1:
            base_model.conv1 = nn.Conv2d(1, base_model.conv1.out_channels, kernel_size=base_model.conv1.kernel_size, 
                                stride=base_model.conv1.stride, padding=base_model.conv1.padding, bias=False)
        
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.conv1.weight = torch.nn.Parameter(base_model.conv1.weight.mean(dim=1, keepdim=True))
    
    elif name == 'efficientnet_v2_s':
    
        base_model = models.efficientnet_v2_s(weights="DEFAULT") if pretrain else models.efficientnet_v2_s(weights=None)
        base_model = nn.Sequential(*list(base_model.children()))[:-2] # remove avg pooling and classification layers

        # Modify the first layer to adjust the input channels
        if in_chs == 1:
            base_model.features[0][0] = nn.Conv2d(1, base_model.features[0][0].out_channels, kernel_size=base_model.features[0][0].kernel_size, 
                                                  stride=base_model.features[0][0].stride, padding=base_model.features[0][0].padding, bias=False)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.features[0][0].weight = torch.nn.Parameter(base_model.features[0][0].weight.mean(dim=1, keepdim=True))
    
    elif name == 'convnext_large':
        base_model = models.convnext_large(weights="DEFAULT") if pretrain else models.convnext_large(weights=None)
        base_model = nn.Sequential(*list(base_model.children()))[:-2] # remove avg pooling and classification layers

        if in_chs == 1:
            # Modify the first layer to adjust the input channels
            base_model.features[0][0] = nn.Conv2d(1, base_model.features[0][0].out_channels, kernel_size=base_model.features[0][0].kernel_size, stride=base_model.features[0][0].stride)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.features[0][0].weight = torch.nn.Parameter(base_model.features[0][0].weight.mean(dim=1, keepdim=True))

    elif name == 'maxvit_t':
        base_model = models.maxvit_t(weights="DEFAULT") if pretrain else models.maxvit_t(weights=None)
        base_model.classifier = nn.Identity()  # Replace the classification layer with an identity layer
        
        if in_chs == 1:
            # Modify the first layer to adjust the input channels
            base_model.stem[0][0] = nn.Conv2d(1, base_model.stem[0][0].out_channels, kernel_size=base_model.stem[0][0].kernel_size, 
                                              stride=base_model.stem[0][0].stride, padding=base_model.stem[0][0].padding, bias=False)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.stem[0][0].weight = torch.nn.Parameter(base_model.stem[0][0].weight.mean(dim=1, keepdim=True))

    elif name == 'swin_v2_b':

        base_model = models.swin_v2_b(weights="DEFAULT") if pretrain else models.swin_v2_b(weights=None)
        base_model = nn.Sequential(*list(base_model.children()))[:-3] # remove avg pooling and classification layers
        
        if in_chs == 1:
            # Modify the first layer to adjust the input channels
            base_model.features[0][0] = nn.Conv2d(1, base_model.features[0][0].out_channels, kernel_size=base_model.features[0][0].kernel_size, stride=base_model.features[0][0].stride)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.features[0][0].weight = torch.nn.Parameter(base_model.features[0][0].weight.mean(dim=1, keepdim=True))
                
    elif name == 'mobilenet_v3_large':
    
        base_model = models.mobilenet_v3_large(weights="DEFAULT") if pretrain else models.mobilenet_v3_large(weights=None)
        base_model = nn.Sequential(*list(base_model.children()))[:-2] # remove avg pooling and classification layers

        if in_chs == 1:
            # Modify the first layer to adjust the input channels
            base_model.features[0][0] = nn.Conv2d(1, base_model.features[0][0].out_channels, 
                                                  kernel_size=base_model.features[0][0].kernel_size, stride=base_model.features[0][0].stride, padding=base_model.features[0][0].padding, bias=False)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.features[0][0].weight = torch.nn.Parameter(base_model.features[0][0].weight.mean(dim=1, keepdim=True))
                
    elif name == 'mobilenet_v3_small':
        
        base_model = models.mobilenet_v3_small(weights="DEFAULT") if pretrain else models.mobilenet_v3_small(weights=None)
        base_model = nn.Sequential(*list(base_model.children()))[:-2] # remove avg pooling and classification layers

        if in_chs == 1:
            # Modify the first layer to adjust the input channels
            base_model.features[0][0] = nn.Conv2d(1, base_model.features[0][0].out_channels, 
                                                  kernel_size=base_model.features[0][0].kernel_size, stride=base_model.features[0][0].stride, padding=base_model.features[0][0].padding, bias=False)
            
            # Adjust the weights by averaging them (to retain as much as possible of the pretraining)
            with torch.no_grad():
                base_model.features[0][0].weight = torch.nn.Parameter(base_model.features[0][0].weight.mean(dim=1, keepdim=True))
        
    elif name == 'mit_b3':
        base_model = transformers.SegformerModel.from_pretrained("nvidia/mit-b3")
        # Currently avoiding in_chs
        
    else:
        print(f"{name} is not in pretrained names: {pretrained_names}")
        
    return base_model



            