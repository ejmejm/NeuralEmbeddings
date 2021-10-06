import math
from typing import Callable, Dict, List, Optional

import numpy as np
from einops import rearrange
import torch
from torch import nn, Tensor
import torch.nn.functional as F


# Tokens are used to mark important parts of input sequences
PAD_TOKEN = '<pad>' # Padding after end of sequence
PRIMARY_TOKEN = '<primary>' # Start of primary inputs
CALIB_TOKEN = '<calib>' # Start of calibration inputs
MASK_TOKEN = '<mask>' # Masks for masked sequence modeling

TOKEN_TO_IDX = {
    PAD_TOKEN: 0,
    PRIMARY_TOKEN: 1,
    CALIB_TOKEN: 2,
    MASK_TOKEN: 3
}

def get_token_embeddings(tokens: List[str], embedding_dim: int, device: str) -> Tensor:
    """
    Returns the embedding for a token.

    Args:
        tokens: List of tokens.
        embedding_dim: dimension of the embedding.
        device: device to get embedding on.

    Returns:
        Tensor, shape (n_tokens, embedding_dim).
    """
    tokens = [token.lower() for token in tokens]
    token_ids = []
    for token in TOKEN_TO_IDX:
        if token.lower() in tokens:
            token_ids.append(TOKEN_TO_IDX[token.lower()])
        else:
            raise ValueError(f'List contains unknown token: {tokens}')

    return F.one_hot(torch.tensor(token_ids, device=device), embedding_dim)

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
        x = x + self.pe[:, :x.size(0)]
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
    
    def forward(self, x: Tensor, sm_mask: Optional[Tensor] = None,
        embed_hook: Callable[[Tensor], Tensor] = None) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x: input tensor, shape [batch_size, seq_len, n_channels].
            sm_mask: mask for sequence modeling where 1 = keep and 0 = mask,
                shape [batch_size, seq_len].
            embed_hook: function to apply to the embeddings before passing them
                through the transformer. 

        Returns:
            Tensor, shape [batch_size, seq_len, embedding_dim].
        """
        # TODO: Figure out how we want to do masking for padded inputs
        # Check Multiheaded Attention for masking: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html?highlight=multiheadattention#torch.nn.MultiheadAttention
        
        if self.include_conv:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)

        if embed_hook is not None:
            x = embed_hook(x)

        # B x S x E
        targets = None
        if self.include_transformer:
            # TODO: Double check positional encodings are working correctly
            # Update: I think I fixed the issue, but I haven tested extensively yet
            x = self.pos_enc(x)

            # Apply the mask for masked sequence modeling if applicable
            if sm_mask is not None:
                targets = x
                x = x * (1 - sm_mask.unsqueeze(2))

            x = self.encoder(x) # Mask could go here
            # print(x)

        return {
            'embeddings': x,
            'targets': targets
        }

def apply_channel_combine_func(func_str, data):
    if func_str == 'mean':
        return data.mean(dim=1)
    elif func_str == 'logsumexp':
        return torch.log(torch.exp(data).sum(dim=1))
    else:
        raise ValueError(f'Unknown channel combine function: {func_str}')

class NeuroSignalEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Initialize the single channel encoder
        conv_config = config['primary_conv']
        encoder_config = config['single_channel_encoder']
        self.sc_encoder = Wav2Vec(
            input_dim = config['max_primary_input_len'],
            embedding_dim = config['embedding_dim'],
            embed_reduc_factor = conv_config['stride'],
            conv_width = conv_config['filter_size'],
            n_layers = encoder_config['n_layers'],
            dropout = encoder_config['dropout'],
            n_head = encoder_config['n_head'],
            include_conv = conv_config['enabled'],
            include_transformer = encoder_config['enabled'])
        self.embed_seq_len = self.sc_encoder.embed_seq_len

        # Initialize the mixed channel encoder
        encoder_config = config['mixed_channel_encoder']
        input_channels = config['embedding_dim'] if \
            config['single_channel_encoder']['enabled'] else 1
        self.mc_encoder = Wav2Vec(
            input_dim = self.sc_encoder.embed_seq_len,
            input_channels = input_channels,
            embedding_dim = config['embedding_dim'],
            n_layers = encoder_config['n_layers'],
            dropout = encoder_config['dropout'],
            n_head = encoder_config['n_head'],
            include_conv = False,
            include_transformer = encoder_config['enabled'])

        # Initialize the calibration model
        conv_config = config['calibration_conv']
        encoder_config = config['calibration_encoder']
        self.calibration_model = Wav2Vec(
            input_dim = config['max_calibration_input_len'],
            embedding_dim = config['embedding_dim'],
            embed_reduc_factor = conv_config['stride'],
            conv_width = conv_config['filter_size'],
            n_layers = encoder_config['n_layers'],
            dropout = encoder_config['dropout'],
            n_head = encoder_config['n_head'],
            include_conv = conv_config['enabled'],
            include_transformer = encoder_config['enabled'])

        # Weighting linear layer for calibration

        self.token_to_idx = {token: torch.tensor(val, dtype=torch.long) for token, val in TOKEN_TO_IDX.items()}
        # 

        # Create special tokens and special token embeddings layer
        self.token_embeddings = nn.Embedding(
            num_embeddings = len(TOKEN_TO_IDX),
            embedding_dim = config['embedding_dim'],
            padding_idx = 0)

    def _format_embeddings(self, primary_embeds: Tensor,
        calib_embeds: Tensor = None) -> Tensor:
        """
        Format the embeddings for the model by combining primary
        and calibration inputs, and by addings appropriate tags.

        Args:
            primary_embeds: embeddings for the primary sequence, shape [batch_size, seq_len, embedding_dim].
            calib_embeds: embeddings for the calibration sequence, shape [seq_len, embedding_dim].

        Returns:
            Tensor, shape [batch_size, seq_len, embedding_dim].
        """
        primary_tag_embeds = self.token_embeddings(
            torch.tensor([TOKEN_TO_IDX[PRIMARY_TOKEN]], dtype=torch.long))
        primary_embeds = torch.cat([primary_embeds, primary_tag_embeds], dim=2)

        if calib_embeds is not None:
            calib_tag_embeds = self.token_embeddings(
                torch.tensor([TOKEN_TO_IDX['calibration_tag']], dtype=torch.long))
            calib_embeds = torch.cat([calib_embeds, calib_tag_embeds], dim=2)

        return torch.cat([primary_embeds, calib_embeds], dim=1)

        # Combine the primary and calibration embeddings
        if calib_embeds is not None:
            embeds = torch.cat([primary_embeds, calib_embeds], dim=1)
        else:
            embeds = primary_embeds

        # Add special tokens
        embeds = torch.cat([
            embeds,
            self.token_embeddings(torch.tensor([TOKEN_TO_IDX['<PAD>']] * embeds.shape[0])).unsqueeze(1)
        ], dim=1)

        # Add tags
        embeds = torch.cat([
            embeds,
            self.token_embeddings(torch.tensor([TOKEN_TO_IDX['<PRIMARY>']] * embeds.shape[0])).unsqueeze(1)
        ], dim=1)

        if calib_embeds is not None:
            embeds = torch.cat([
                embeds,
                self.token_embeddings(torch.tensor([TOKEN_TO_IDX['<CALIBRATION>']] * embeds.shape[0])).unsqueeze(1)
            ], dim=1)

        return embeds

    def forward(
        self,
        primary_input: Tensor,
        sc_sm_mask: Optional[Tensor] = None,
        mc_sm_mask: Optional[Tensor] = None,
        calibration_input: Optional[Tensor] = None) -> Tensor:
        """
        Generates emebeddings for the primary_input, using the calibration_input
        to provide extra info about channels.

        Args:
            primary_input: input tensor, shape [batch_size, seq_len, n_channels].
            sc_sm_mask: mask for single-channel sequence modeling
                where 1 = keep and 0 = mask, shape [batch_size, seq_len].
            mc_sm_mask: mask for multi-channel sequence modeling
                where 1 = keep and 0 = mask, shape [batch_size, seq_len].
            calibration_input: input tensor for calibration,
                shape [batch_size, seq_len, n_channels].

        Returns:
            Tensor, shape [batch_size, seq_len, embedding_dim].
        """
        n_calib_channels = calibration_input.shape[2]
        n_channels = primary_input.shape[2]
        assert n_channels == n_calib_channels, \
            f'Primary input channels ({n_channels}) must match calibration input channels ({n_calib_channels})'

        # Format calibration inputs and run them through the calibration model
        if calibration_input is not None:
            calibration_input = rearrange(calibration_input, 'b s c -> (b c) s 1')
            calib_embeds = self.calibration_model(
                calibration_input, sm_mask=None)['embeddings']
            # Note: calibration input and primary input batches are currently not aligned
            # All calibration batches are combined sequentially and prepended to all primary batches
            calib_embeds = rearrange(calib_embeds, '(b c) s e -> 1 c (b s) e', c=n_channels)



        else:
            pass

            
        n_channels = primary_input.shape[2]
        primary_input = rearrange(primary_input, 'b s c -> (b c) s 1')

        # Format inputs and run through the model
        sc_outputs = self.sc_encoder(primary_input, sc_sm_mask)
        sc_embeds = sc_outputs['embeddings']
        sc_targets = sc_outputs['targets']
        sc_embeds = rearrange(sc_embeds, '(b c) s e -> b c s e', c=n_channels)

        # Combine channels into a mixed channel
        mc_embeds = apply_channel_combine_func(
            self.config['channel_combine_func'], sc_embeds)

        mc_outputs = self.mc_encoder(mc_embeds, mc_sm_mask)
        output_embeds = mc_outputs['embeddings']
        mc_targets = mc_outputs['targets']

        return {
            'embeddings': output_embeds,
            'mc_embeddings': output_embeds,
            'mc_targets': mc_targets,
            'sc_embeddings': sc_embeds,
            'sc_targets': sc_targets
        }

