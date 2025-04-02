
def model_params(name, config=None):
    """ Register model parameters here """
    
    # Create a dictionary to store model parameters
    param = dict()
    
    if name == "EnsembleEnet2SepIn":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["dropout"] = config.model.dropout
        
    elif name == "EnsembleRes18SepIn":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["dropout"] = config.model.dropout
        
    elif name == "EnsembleRes18Enetb2s":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["dropout"] = config.model.dropout

    elif name == "ResNet50Pscse_256x28x28":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout
        param["activation"] = config.model.activation
        param["reduction"] = config.model.reduction
        
    elif name == "ResNet50Pscse_512x28x28":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout
        param["activation"] = config.model.activation
        param["reduction"] = config.model.reduction
        
    elif name == "EfficientNetB2LPscse_384x28x28":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout
        param["activation"] = config.model.activation
        param["reduction"] = config.model.reduction
        
    elif name == "EnsembleResNet50_512x28PscseEfficientNetB2Pscse384X28":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout
        param["activation"] = config.model.activation
        param["reduction"] = config.model.reduction   
        param["separate_inputs"] = config.model.separate_inputs
        
    elif name == "EnsembleResNet18Ft512_EfficientNetB2SFt1408":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< temporarily changed for doppler
        param["separate_inputs"] = config.model.separate_inputs
        
    elif name == "EnsembleResNet18Ft512_EfficientNetB2SFt1408V2":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat)
        param["separate_inputs"] = config.model.separate_inputs
        param["only_feature_extraction"] = config.model.only_feature_extraction

    elif name == "RadiomicMLP_EnsembleResNet18Ft512_EfficientNetB2SFt1408":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat)
        param["separate_inputs"] = config.model.separate_inputs
        param["radiomic_dims"] = config.model.radiomic_dims
        param["radiomic_activation"] = config.model.radiomic_activation
        param["radiomic_attention"] = config.model.radiomic_attention
        param["radiomic_dropout"] = config.model.radiomic_dropout

    elif name == "MiT":
        param["name"] = config.model.subname
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat)
        param["cls_activation"] = config.model.cls_activation
        param["separate_inputs"] = config.model.separate_inputs

    elif name == "EnsembleModelsV1":
        param["names"] = config.model.subnames
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat)
        param["cls_activation"] = config.model.cls_activation
        param["input_seq"] = config.model.input_seq
        
    elif name == "EnsembleModelsV2":
        param["names"] = config.model.subnames
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat)
        param["cls_activation"] = config.model.cls_activation
        param["input_seq"] = config.model.input_seq
        param["preensemble"] = config.model.preensemble

    elif name == "EnsembleResNet18Ft512_MBV3LFt960":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat)
        param["cls_activation"] = config.model.cls_activation
        param["separate_inputs"] = config.model.separate_inputs

    elif name == "EnsembleEfficientNetB2SFt1408_MBV3LFt960":
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat)
        param["cls_activation"] = config.model.cls_activation
        param["separate_inputs"] = config.model.separate_inputs

    elif name == "TwoPlusOneEnsemble":
        param["names"] = config.model.subnames
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat)
        param["cls_activation"] = config.model.cls_activation
        param["separate_inputs"] = config.model.separate_inputs
    
    elif name == "BaseModelSepIn":
        param["name"] = config.model.subname
        param["num_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["pretrain"] = config.model.pretrain
        param["dropout"] = config.model.dropout 
        param["in_chs"] = len(config.data.concat)
        param["cls_activation"] = config.model.cls_activation
        param["separate_inputs"] = config.model.separate_inputs
        
    elif name == "base_models":
        param["name"] = config.model.subname
        param["pretrain"] = config.model.pretrain
        param["num_classes"] = config.train.n_classes
        param["in_chs"] = len(config.data.concat)
        
    else:
        raise ValueError(f"{name} is not found in supported model list")

    return param
        

    