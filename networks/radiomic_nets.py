# Append utils
import sys
sys.path.append("/research/m324371/Project/adnexal/utils/")

import torch
import torch.nn as nn
from activations import activation_function


# Define the MLP for radiomic features
class RadiomicMLP(nn.Module):
    def __init__(self, radiomic_dims:list, activation:str='leakyrelu', attention=None, dropout:float=None):
        """
        Args:
        - radiomic_dims: A list in the format: [in_dim, hidden_dim, out_dim] (e.g. [100, 64, 128])
        - activation: Activation function
        - Attention module
        - dropout

        Return:
        - An MLP network
        """
        super(RadiomicMLP, self).__init__()

        self.radiomic_dims = radiomic_dims
        
        layers = []
        
        # Create fully connected layers based on radiomic_dims. If it has only one value, that means
        # it is input dims only. So, no need to create MLP. 
        if self.radiomic_dims is not None:
            for i in range(len(self.radiomic_dims) - 1):
                layers.append(nn.Linear(self.radiomic_dims[i], self.radiomic_dims[i + 1]))
                if attention is not None: layers.append(attention(feature_dim=self.radiomic_dims[i + 1]))
                layers.append(activation_function(activation))
                # if dropout is not None: layers.append(nn.Dropout(dropout))
                if dropout is not None and i < len(self.radiomic_dims) - 2: layers.append(nn.Dropout(dropout)) # Only add dropout to hidden layers, not the output layer.
            
            self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        batch_size, num_features, feature_length = x.shape
        x = x.view(batch_size * num_features, feature_length) # Shape: (batch_size * num_features, mlp_output_dim)
        if self.radiomic_dims is not None:
            x = self.fc(x)

        return x

        
if __name__ == "__main__":

    from attention import BasicAttention
    
    # Test the model
    radiomic_input = torch.randn(4, 10, 100)
    
    radiomic_dims = [100, 64, 128, 256, 512] # None  
    activation = 'leakyrelu'
    dropout = 0.2

    attention = BasicAttention # define the attention module to be used
        
    model = RadiomicMLP(radiomic_dims, activation, attention, dropout)  # Create the model 
    
    out = model(radiomic_input)  # Forward pass
    print(out.shape)  # torch.Size([40, 512])

