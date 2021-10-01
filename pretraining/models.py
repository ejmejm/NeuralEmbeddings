import math
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor

# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# TODO: Change to positional encoding used by wav2vec 2.0
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) \
            * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(1, max_len, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
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

        # Initialize the positional encoding layer
        self.pos_enc = PositionalEncoding(
            embedding_dim, 0.25 * dropout, self.embed_seq_len)

        # Look at this: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        # And follow this tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        if include_transformer:
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

    def get_device(self):
        return next(self.parameters()).device
    
    def forward(self, x: Tensor, sm_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x: input tensor, shape [batch_size, seq_len, n_channels].
            sm_mask: mask for sequence modeling where 1 = keep and 0 = mask,
                shape [batch_size, seq_len].

        Returns:
            Tensor, shape [batch_size, seq_len, embedding_dim].
        """
        # TODO: Figure out how we want to do masking for padded inputs
        # Check Multiheaded Attention for masking: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html?highlight=multiheadattention#torch.nn.MultiheadAttention
        
        if self.include_conv:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)

        # TODO: Double check positional encodings are working correctly
        # Brief testing made it seem like they may be hindering performance
        x = self.pos_enc(x)

        # Apply the mask for masked sequence modeling if applicable
        if sm_mask is not None:
            x *= 1 - sm_mask.unsqueeze(2)

        # B x S x E
        if self.include_transformer:
            x = self.encoder(x) # Mask could go here
        return x
