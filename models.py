"""
Implements the wav2vec 2.0 model.
The model starts with a 1D convolutional layer, and is then followed by a transformer.
"""

import numpy as np
from torch import nn

class Wav2Vec(nn.Module):
    """
    Implements the wav2vec 2.0 architecture.
    The model starts with a 1D convolutional layer,
    and is then followed by a transformer.
    """

    def __init__(
        self,
        input_dim = 1024,
        embed_reduc_factor = 2, # Factor to reduce the input size by
        conv_width = 3,
        n_layers = 6,
        dropout = 0.5,
        include_conv = True,
        include_transformer = True):
        """
        Initializes the model by setting up the layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim / embed_reduc_factor
        self.n_layers = n_layers
        self.dropout = dropout
        
        conv_padding = (conv_width - 1) // 2
        conv_stride = embed_reduc_factor

        self.hidden_dim = int(np.floor((input_dim + 2 * conv_padding - (conv_width - 1) - 1) \
            / conv_stride + 1))

        # Initialize the layers
        if include_conv:
            self.conv = nn.Conv1d(
                in_channels = 1,
                out_channels = 1,
                kernel_size = conv_width,
                padding = conv_padding,
                stride = conv_stride)

        # Look at this: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        # And follow this tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        if include_transformer:
            print(self.hidden_dim)
            self.transformer = nn.Transformer(
                d_model = self.hidden_dim,
                nhead = 8,
                num_encoder_layers = n_layers,
                num_decoder_layers = n_layers,
                dim_feedforward = self.hidden_dim*4,
                dropout = dropout)
    
    def forward(self, x):
        """
        Forward pass through the model.
        """
        if self.conv:
            x = self.conv(x)
        # S x B x E
        if self.transformer:
            x = self.transformer(x)
        return x
