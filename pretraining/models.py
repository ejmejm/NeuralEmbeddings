import math

import numpy as np
import torch
from torch import nn, Tensor

# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Wav2Vec(nn.Module):
    """
    Similar to the Wav2Vec architecture.
    The model starts with a 1D convolutional layer,
    and is then followed by a transformer.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        input_channels: int = 1,
        embedding_dim: int = 64,
        embed_reduc_factor: int = 2, # Factor to reduce the input size by
        conv_width: int = 3,
        n_layers: int = 6,
        dropout: float = 0.5,
        n_head: int = 8,
        include_conv: bool = True,
        include_transformer: bool = True):
        """
        Initializes conv and transformer encoder layers.
        """

        super().__init__()
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        # self.hidden_dim = input_dim / embed_reduc_factor
        self.n_layers = n_layers
        self.dropout = dropout
        self.include_conv = include_conv
        self.include_transformer = include_transformer

        # Initialize the layers
        if include_conv:
            conv_padding = (conv_width - 1) // 2
            conv_stride = embed_reduc_factor

            self.embed_seq_len = int(np.floor((input_dim + 2 * conv_padding - (conv_width - 1) - 1) \
                / conv_stride + 1))

            self.conv = nn.Conv1d(
                in_channels = self.input_channels,
                out_channels = embedding_dim,
                kernel_size = conv_width,
                padding = conv_padding,
                stride = conv_stride)
        else:
            self.embed_seq_len = self.input_dim
            self.embedding_dim = self.input_channels

        # Look at this: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        # And follow this tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        if include_transformer:
            print('Max embedding seq len:', self.embed_seq_len)
            transformer_layer = nn.TransformerEncoderLayer(
                d_model = self.embedding_dim,
                nhead = n_head,
                dim_feedforward = self.embed_seq_len,
                dropout = dropout,
                activation = 'relu',
                batch_first=True
            )
                
            # TODO: Try addinng a normalization layer as a param to the transformer
            self.encoder = nn.TransformerEncoder(transformer_layer, n_layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        """
        # TODO: Add positional embeddings (check word2vec 2.0 paper)

        # TODO: Scale input by 1/sqrt(embedding_dim)?

        # TODO: Figure out how we want to do masking
        # Check Multiheaded Attention for masking: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html?highlight=multiheadattention#torch.nn.MultiheadAttention

        if self.include_conv:
            x = self.conv(x)
        x = x.transpose(1, 2)
        print('Post conv shape:', x.shape)

        # S x B x E
        if self.include_transformer:
            x = self.encoder(x) # Mask could go here
        return x
