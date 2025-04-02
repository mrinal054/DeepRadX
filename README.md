# adnexal_masses_DL

## <img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="50px" /> &nbsp; Latest News

:boom: **Adnexal masses classifier v0 released.** <br>
:boom: **New trainerv7 released. Supports tensorboard.** <br>
:boom: **Combined radiomic and deep learning features (trainerv8) are coming soon.** <br>

## Overview
This repository presents a deep learning-based framework specifically designed for the classification of cancer from adnexal masses. The model leverages advanced neural network architectures to accurately distinguish cancerous from non-cancerous cases, making it a valuable tool for early diagnosis and clinical decision-making. While the primary focus is on adnexal masses, the modular design and robust architecture allow for easy adaptation to other classification problems across different medical or non-medical datasets. The framework incorporates best practices in data preprocessing, augmentation, and model training, ensuring high performance and generalizability. This versatility underscores its potential as a foundational model for various classification tasks.

## Directory setup
To use this framework, the directories should be structured as follows:
```
root 
    |-- adnexal.py 
    |-- config 
        |-- template.yaml 
    |-- dataloader
    |-- losses
    |-- network_parameters
        |-- params.py
    |-- networks 
        |-- model1.py
        |-- model2.py
                :
        |-- nets.py
    |-- tutorials
    |-- utils
```
## How to run?
It is important to properly configure the `template.yaml` file. Carefully read the comments in the template file. There are different template files available in the config directory; however, one can create their own config file while keeping the structure the same. <br>

The `phase` option in the template file allows the user to determine the phase—whether it is for training, testing, or both. During evaluation, the `k-fold` models can be used in two ways to make predictions: by selecting the best model among all k-folds or  by taking the average of the predictions made by the best models from each `k-fold`. <br>

To run deep learning models, use the following command:
```python
python trainer.py --config ./config/template.yaml
```
Or, if you have multi-core GPU:<br>
```python
CUDA_VISIBLE_DEVICES=2 nohup python trainer.py --config ./config/template.yaml >log_train.log &
```


## Models
* [base_models](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/base_models_collection.py?ref_type=heads) <br> It contains popular pretrained models. It currently supports following models -`resnet18`, `resnet50`, `efficientnet_v2_s`, `efficientnet_v2_l`, `mobilenet_v3_large`, `vgg19_bn`, `maxvit_t`, `convnext_base`.
* [ResNet50Pscse_256x28x28](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/res50pscse_256x28x28.py?ref_type=heads) <br> This model combines ResNet50 and P-scSE attention module. If the original image size is `224x224`, it extract ResNet50 up to `256x28x28` followed by two blocks of P-scSE, activation, dropout, and batch normalization. In each block, dimension is reduced by half. So, final feature maps will be reduced to `7x7` before passing to classification head. 
* [EfficientNetB2LPscse_384x28x28](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/enetb2lpscse_384x28x28.py?ref_type=heads) <br> This model is similar to `ResNet50Pscse_256x28x28`, except for it extracts layers from `EfficientNetB2L` up to `384x28x28`.
* [EnsembleRes18Enetb2s](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_res18_enetb2s.py?ref_type=heads) <br> This model ensembles `ResNet18` and `EffcientNetB2S`. 
* [EnsembleRes18SepIn](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_res18_sep_input.py?ref_type=heads) <br> This model takes an input image, creates `ResNet18` for each channel of the image, and finally concatenates features received from each model before passing to the classification head. 
* [EnsembleEnet2SepIn](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_enet2_sep_input.py?ref_type=heads) <br> This model takes an input image, creates `EfficientNetB2` for each channel of the image, and finally concatenates features received from each model before passing to the classification head. 
* [EnsembleResNet18Ft512_EfficientNetB2SFt1408](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_type1.py?ref_type=heads) <br> This model ensembles ResNet18 with 512 features and EfficientNetB2 with 1408 features. It supports `separate_inputs`.
* [EnsembleResNet18Ft512_EfficientNetB2SFt1408V2](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_type1.py?ref_type=heads) <br> This is the modified version of the previous model. It ensembles ResNet18 with 512 features and EfficientNetB2 with 1408 features. V2 added a new argument `only_feature_extraction`. It decides whether the model be a feature extractor or a classifier. It supports `separate_inputs`.
* [EnsembleResNet50_512x28PscseEfficientNetB2Pscse384X28](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_type1.py?ref_type=heads) <br> This model ensembles `ResNet50Pscse_512x28x28` and `EfficientNetB2LPscse_384x28x28`. It supports `separate_inputs`.
* [RadiomicMLP](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/radiomic_nets.py?ref_type=heads) <br> This model takes radiomic features as input. Based on the input argument `radiomic_dims`, it decides wheter to return the radiomic features performing some reshaping or pass it to a block consisting of fully-connected layers, attention module, and dropout. 
* [RadiomicMLP_EnsembleResNet18Ft512_EfficientNetB2SFt1408](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_radiomics.py?ref_type=heads) <br> This model ensembles radiomic features from `RadiomicMLP` with `EnsembleResNet18Ft512_EfficientNetB2SFt1408`. 

## Custom models
If you want to integrate a customized model to this pipeline, follow these guidelines. Let's assume you have created model called `MyCustomNet` in `my_custom_net.py`. Also, assume that it takes three input parameters - `num_classes`, `activation`, and `dropout`. <br><br> 
Now, follow these steps: <br><br>
**Step 1:** Place `my_custom_net.py` in `networks` directory. <br>
**Step 2:** Register `MyCustomNet` in `nets.py` within networks directory. Import it in `nets.py` and add it to `__all__`. <br>
**Step 3:** Register input parameters of the model `params.py` in `network_parameters` directory. For `MyCustomNet`, it will be: 
```python
elif name == "MyCustomNet":
        param["num_classes"] = 2
        param["activation"] = 'leakyrelu'
        param["dropout"] = 0.3
```
You can also define the model parameters in the template to avoid hard coding. In this case, it would look like this:

```python
elif name == "MyCustomNet":
        param["num_classes"] = config.train.n_classes
        param["activation"] = config.model.activation
        param["dropout"] = config.model.dropout
```
This approach helps avoid hard coding the parameters.

### Outputs
A successful run will store the following information in the `result` directory: <br>
* Image-wise validation result for each fold
* Classification report for each fold
* Training and validation loss curve for each fold
* Checkpoints for each fold
* ROC for each fold
* Image-wise evaluation for test images
* Classification report for test images
* TensorBoard containing losses and metrics

## Results

| Model | AD | Fluid | Solid | Doppler | Accuracy | Precision | Recall | F1-score | AUC | Param (M)|
| :---: | :---: |  :---: |  :---: |  :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
| [ResNet18](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/base_models_collection.py?ref_type=heads) | &check; | &cross; | &cross; | &cross; | 0.849 | 0.849 | 0.849 | 0.849 | 0.87 | - | 
| [ResNet18](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_res18_sep_input.py?ref_type=heads) | &check; | &check; | &check; | &cross; | 0.872 | 0.872 | 0.872 | 0.872 | 0.93 | - |
| [EfficientNetB2](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/base_models_collection.py?ref_type=heads) | &check; | &cross; | &cross; | &cross; | 0.854 | 0.857 | 0.854 | 0.854 | 0.93 | - | 
| [ResNet18+EfficientNetB2](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_res18_enetb2s.py?ref_type=heads) | &check; | &cross; | &cross; | &cross; | 0.86 | 0.86 | 0.86 | 0.86 | **0.94** | - | 
[ResNet18Ft512+EfficientNetB2SFt1408](https://gitlab.mayo.edu/kline-lab/adnexal_masses_dl/-/blob/main/networks/ensemble_type1.py?ref_type=heads) | &check; | &check; | &check; | &cross; | **0.89** | **0.894** | **0.890** | **0.891** | **0.94** | - |
