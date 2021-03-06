import math
from typing import Callable, Dict, List, Optional

import numpy as np
from einops import rearrange, repeat
import torch
from torch import nn, Tensor
import torch.nn.functional as F


# Tokens are used to mark important parts of input sequences
PAD_TOKEN = '<pad>' # Padding after end of sequence (currently unused)
PRIMARY_TOKEN = '<primary>' # Start of primary inputs
CALIB_TOKEN = '<calib>' # Start of calibration inputs
MASK_TOKEN = '<mask>' # Masks for masked sequence modeling (currently unused)

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
        x = x + self.pe[:, :x.size(1)]
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
        n_convs: int = 1,
        conv_width: int = 3,
        n_layers: int = 6,
        dropout: float = 0.5,
        n_head: int = 8,
        feedforward_dim: int = 2048,
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
        if include_conv and n_convs > 0:
            conv_padding = (conv_width - 1) // 2
            conv_stride = embed_reduc_factor

            # Length of embedding sequences, which is calculated based on
            # the maximum input size of the model + the convolutional params
            self.embed_seq_len = input_dim
            for _ in range(n_convs):
                self.embed_seq_len = int(np.floor((self.embed_seq_len + 2 * conv_padding - (conv_width - 1) - 1) \
                    / conv_stride + 1))

            # Create the conv layers
            self.convs = []
            first_conv = nn.Conv1d(
                in_channels = self.input_channels,
                out_channels = embedding_dim,
                kernel_size = conv_width,
                padding = conv_padding,
                stride = conv_stride)
            self.convs.append(first_conv)
            self.convs.append(nn.ReLU())
            for _ in range(1, n_convs):
                conv = nn.Conv1d(
                    in_channels = embedding_dim,
                    out_channels = embedding_dim,
                    kernel_size = conv_width,
                    padding = conv_padding,
                    stride = conv_stride)
                self.convs.append(conv)
                self.convs.append(nn.ReLU())

            self.convs.append(nn.Dropout(p=dropout))
            self.convs = nn.Sequential(*self.convs)
        else:
            self.embed_seq_len = self.input_dim
            self.embedding_dim = self.input_channels

        # Initialize the positional encoding layer
        # self.pos_enc = PositionalEncoding(
        #     embedding_dim, 0.25 * dropout, self.embed_seq_len)

        # Reference for understanding how this works:
        # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        if include_transformer:
            transformer_layer = nn.TransformerEncoderLayer(
                d_model = self.embedding_dim,
                nhead = n_head,
                dim_feedforward = feedforward_dim,
                dropout = dropout,
                activation = 'relu',
                batch_first=True
            )
                
            # TODO: Try addinng a normalization layer as a param to the transformer
            self.encoder = nn.TransformerEncoder(transformer_layer, n_layers)

        self.lstm_head = None

    def from_config(config):
        """
        Initializes a wav2vec model from a config dict.
        This should only be used when the main model is a wav2vec one.
        """
        w2v_config = config['wav2vec']
        if 'transformer_enabled' not in w2v_config or \
           w2v_config['transformer_enabled'] is None:
            transformer_enabled = True
        else:
            transformer_enabled = w2v_config['transformer_enabled']

        model = Wav2Vec(
            input_dim = config['max_primary_input_len'] + \
                config['max_calibration_input_len'],
            embedding_dim = config['embedding_dim'],
            input_channels = w2v_config['n_input_channels'],
            embed_reduc_factor = w2v_config['stride'],
            n_convs = w2v_config['n_convs'],
            conv_width = w2v_config['filter_size'],
            n_layers = w2v_config['n_layers'],
            dropout = w2v_config['dropout'],
            n_head = w2v_config['n_head'],
            feedforward_dim = w2v_config['feedforward_dim'],
            include_conv = True,
            include_transformer = transformer_enabled)
        return model

    def get_device(self):
        return next(self.parameters()).device

    def add_lstm_head(self, output_dim: int) -> None:
        """
        Add an LSTM head to the model.
        This is required for CPC, and optional when
        training downstream tasks.

        Args:
            output_dim: dimension of the LSTM head output.
        """
        self.lstm_head = nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = output_dim,
            num_layers = 1,
            batch_first = True)
    
    def forward(self, x: Tensor, sm_mask: Optional[Tensor] = None,
        embed_hook: Callable[[Tensor], Tensor] = None) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x: input tensor, shape [batch_size, seq_len, n_channels].
            sm_mask: mask for sequence modeling where 1 = mask and 0 = keep,
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
            x = self.convs(x)
            x = x.transpose(1, 2)

        if embed_hook is not None:
            x = embed_hook(x)

        # B x S x E
        targets = None
        if self.include_transformer:
            # Apply the mask for masked sequence modeling if applicable
            if sm_mask is not None:
                targets = x
                # TODO: Fix the issue where special tags can be masked
                x = x * (1 - sm_mask.unsqueeze(2))

            # TODO: Double check positional encodings are working correctly and applied in correct place
            # Update: I think I fixed the issue, but I haven't tested extensively yet
            # x = self.pos_enc(x)

            x = self.encoder(x) # Mask could go here

        ### LSTM head ###

        if self.lstm_head is not None:
            # Run the embeddings through the LSTM head
            lstm_embeds, lstm_hidden = self.lstm_head(x)
        else:
            lstm_embeds = None
            lstm_hidden = None

        ### Returns ###

        return {
            'embeddings': x,
            'targets': targets,
            'lstm_embeddings': lstm_embeds,
            'lstm_hidden': lstm_hidden,
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
        calib_config = config['calibration_module']
        sc_config = config['single_channel_module']
        mc_config = config['mixed_channel_module']
        self.embedding_dim = config['embedding_dim']

        # Config validation
        assert sc_config['enabled'] or mc_config['enabled'], \
            'Must enable at least one of the single channel or mixed channel modules!'
        assert sc_config['enabled'] or not calib_config['enabled'], \
            'Cannot use the calibration model without a single-channel encoder!'

        # Initialize the calibration model
        if calib_config['enabled']:
            self.calibration_model = Wav2Vec(
                input_dim = config['max_calibration_input_len'],
                embedding_dim = config['embedding_dim'],
                embed_reduc_factor = calib_config['stride'],
                n_convs = calib_config['n_convs'],
                conv_width = calib_config['filter_size'],
                n_layers = calib_config['n_layers'],
                dropout = calib_config['dropout'],
                n_head = calib_config['n_head'],
                feedforward_dim = calib_config['feedforward_dim'],
                include_conv = True,
                include_transformer = True)
            self.calib_embed_seq_len = self.calibration_model.embed_seq_len
        else:
            self.calibration_model = None
            self.calib_embed_seq_len = None


        # Calculate input dim to single channel encoder
        # Need to increase it to account for the calibration input and tags
        if self.calibration_model is not None:
            calib_seq_len = self.calibration_model.embed_seq_len
            # Maybe should use ceil here
            adjusted_input_dim = int(calib_seq_len * sc_config['stride']) # Calib input
            adjusted_input_dim += sc_config['stride'] # Calib tag
        else:
            adjusted_input_dim = 0
            
        adjusted_input_dim += config['max_primary_input_len'] # Primary input
        adjusted_input_dim += sc_config['stride'] # Primary tag
        
        # Initialize the single channel encoder
        if sc_config['enabled']:
            self.sc_encoder = Wav2Vec(
                input_dim = adjusted_input_dim,
                embedding_dim = config['embedding_dim'],
                embed_reduc_factor = sc_config['stride'],
                n_convs = sc_config['n_convs'],
                conv_width = sc_config['filter_size'],
                n_layers = sc_config['n_layers'],
                dropout = sc_config['dropout'],
                n_head = sc_config['n_head'],
                feedforward_dim = sc_config['feedforward_dim'],
                include_conv = True,
                include_transformer = True)
            self.embed_seq_len = self.sc_encoder.embed_seq_len
        else:
            self.sc_encoder = None

        # Initialize the mixed channel encoder
        if mc_config['enabled']:
            if sc_config['enabled']:
                input_dim = self.sc_encoder.embed_seq_len
                input_channels = config['embedding_dim']
            else:
                input_dim = adjusted_input_dim
                input_channels = 1

            self.mc_encoder = Wav2Vec(
                input_dim = input_dim,
                input_channels = input_channels,
                embedding_dim = config['embedding_dim'],
                embed_reduc_factor = mc_config['stride'],
                n_convs = mc_config['n_convs'],
                conv_width = mc_config['filter_size'],
                n_layers = mc_config['n_layers'],
                dropout = mc_config['dropout'],
                n_head = mc_config['n_head'],
                feedforward_dim = mc_config['feedforward_dim'],
                include_conv = not sc_config['enabled'],
                include_transformer = True)
            self.embed_seq_len = self.mc_encoder.embed_seq_len
        else:
            self.mc_encoder = None

        # LSTM head used for summarizing the outputs
        # Can be added after initialization of the model
        self.lstm_head = None

        # Register index tensors as buffers
        # This way, their devices will be updated with the model's device
        for token, idx in TOKEN_TO_IDX.items():
            idx_tensor = torch.tensor(idx, dtype=torch.long)
            self.register_buffer(f'{token}_idx', idx_tensor)
        
        # Create special tokens and special token embeddings layer
        self.token_embeddings = nn.Embedding(
            num_embeddings = len(TOKEN_TO_IDX),
            embedding_dim = config['embedding_dim'],
            padding_idx = 0)

    def _token_to_idx(self, token: str) -> Tensor:
        return self.__getattr__(f'{token}_idx')

    def _get_token_embeddings(self, tokens: List[str]) -> List[Tensor]:
        """
        Get the embeddings for a list of tokens.

        Args:
            tokens: list of tokens

        Returns:
            List of embeddings, one for each token.
        """
        token_idxs = [self._token_to_idx(token) for token in tokens]
        token_idxs = torch.stack(token_idxs, dim=0).unsqueeze(0)
        return self.token_embeddings(token_idxs).squeeze(0)

    def _format_embeddings(
        self,
        primary_embeds: Tensor,
        calib_embeds: Tensor = None) -> Tensor:
        """
        Format the embeddings for the model by combining primary
        and calibration inputs, and by prepending appropriate tags.

        Args:
            primary_embeds: embeddings for the primary sequence, shape [batch_size, seq_len, embedding_dim].
            calib_embeds: embeddings for the calibration sequence, shape [seq_len, embedding_dim].

        Returns:
            Tensor, shape [batch_size, seq_len, embedding_dim].
        """
        primary_tag_embeds, calib_tag_embeds = \
            self._get_token_embeddings([PRIMARY_TOKEN, CALIB_TOKEN])

        primary_tag_embeds = torch.vstack([primary_tag_embeds] * primary_embeds.shape[0])
        primary_tag_embeds = primary_tag_embeds.unsqueeze(1)

        embeds = torch.cat([primary_tag_embeds, primary_embeds], dim=1)

        if calib_embeds is not None:
            calib_tag_embeds = calib_tag_embeds.unsqueeze(0)
            calib_embeds = torch.cat([calib_tag_embeds, calib_embeds], dim=0)
            calib_embeds = torch.stack([calib_embeds] * primary_embeds.shape[0], dim=0)
            embeds = torch.cat([calib_embeds, embeds], dim=1)

        # Could add padding here if not done before passing inputs to the model

        return embeds

    def _format_embeddings(
        self,
        primary_embeds: Tensor,
        calib_embeds: Tensor = None) -> Tensor:
        """
        Format the embeddings for the model by combining primary
        and calibration inputs, and by addings appropriate tags.

        Args:
            primary_embeds: embeddings for the primary sequence,
                shape [batch_size * n_channels, seq_len, embedding_dim].
            calib_embeds: embeddings for the calibration sequence,
                shape [n_channels, seq_len, embedding_dim].

        Returns:
            Tensor, shape [batch_size * n_channels, seq_len, embedding_dim].
        """
        primary_tag_embeds, calib_tag_embeds = \
            self._get_token_embeddings([PRIMARY_TOKEN, CALIB_TOKEN])

        primary_tag_embeds = primary_tag_embeds.unsqueeze(0) # seq_len
        primary_tag_embeds = torch.stack( # batch_size * n_channels
            [primary_tag_embeds] * primary_embeds.shape[0], dim=0)

        embeds = torch.cat([primary_tag_embeds, primary_embeds], dim=1)

        if calib_embeds is not None:
            calib_tag_embeds = calib_tag_embeds.unsqueeze(0) # seq_len
            calib_tag_embeds = torch.stack( # n_channels
                [calib_tag_embeds] * calib_embeds.shape[0], dim=0)
            calib_embeds = torch.cat([calib_tag_embeds, calib_embeds], dim=1)
            batch_size = int(primary_embeds.shape[0] / calib_embeds.shape[0])
            if batch_size != primary_embeds.shape[0] / calib_embeds.shape[0]:
                raise ValueError(
                    f'Primary embeddings first dimension, ({primary_embeds.shape[0]}) '
                    f'should be a multiple of the calibration channel size, '
                    f'({calib_embeds.shape[0]})')
            calib_embeds = torch.cat( # batch_size
                [calib_embeds] * batch_size, dim=0)
            embeds = torch.cat([calib_embeds, embeds], dim=1)

        # Could add padding here if not done before passing inputs to the model

        return embeds

    def add_lstm_head(self, output_dim: int) -> None:
        """
        Add an LSTM head to the model.
        This is required for CPC, and optional when
        training downstream tasks.

        Args:
            output_dim: dimension of the LSTM head output.
        """
        self.lstm_head = nn.LSTM(
            input_size = self.config['embedding_dim'],
            hidden_size = output_dim,
            num_layers = 1,
            batch_first = True)

    def forward(
        self,
        primary_input: Tensor,
        sc_sm_mask: Optional[Tensor] = None,
        mc_sm_mask: Optional[Tensor] = None,
        calib_sm_mask: Optional[Tensor] = None,
        calibration_input: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Generates emebeddings for the primary_input, using the calibration_input
        to provide extra info about channels.

        Args:
            primary_input: input tensor, shape [batch_size, seq_len, n_channels].
            sc_sm_mask: mask for single-channel sequence modeling
                where 1 = keep and 0 = mask, shape [batch_size, seq_len].
            mc_sm_mask: mask for multi-channel sequence modeling
                where 1 = keep and 0 = mask, shape [batch_size, seq_len].
            calib_sm_mask: mask for calibration sequence modeling
                where 1 = keep and 0 = mask, shape [batch_size, seq_len].
            calibration_input: input tensor for calibration,
                shape [batch_size, seq_len, n_channels].

        Returns:
            Dict, contains embeddings and target values.
        """

        ### Input validation ###

        if calibration_input is not None and self.calibration_model is None:
            raise ValueError('Cannot provide a calibration input when the calibration model is disabled!')

        n_channels = primary_input.shape[2]
        if calibration_input is not None:
            n_calib_channels = calibration_input.shape[2]
            assert n_channels == n_calib_channels, \
                f'Primary input channels ({n_channels}) must match calibration input channels ({n_calib_channels})'

        ### Calibration module ###

        # Format calibration inputs and run them through the calibration model
        if calibration_input is not None:
            calibration_input = rearrange(calibration_input, 'b s c -> (b c) s 1')
            if calib_sm_mask is not None:
                # Expand the mask to fit the rearranged inputs
                calib_sm_mask = repeat(calib_sm_mask, 'b s -> (b c) s', c=n_channels)
            calib_outputs = self.calibration_model(
                calibration_input, sm_mask=calib_sm_mask)
            calib_embeds = calib_outputs['embeddings']

            # Return tensors rearranged to match original input shape order
            # The rearranging makes training easier later
            calib_return_embeds = rearrange(calib_embeds,
                '(b c) s e -> b s c e', c=n_channels)
            
            calib_targets = calib_outputs['targets']
            if calib_targets is not None:
                calib_targets = rearrange(calib_targets,
                    '(b c) s e -> b s c e', c=n_channels)

            # Note: calibration input and primary input batches are currently not aligned
            # All calibration batches are combined sequentially and prepended to all primary batches
            calib_embeds = rearrange(calib_embeds, '(b c) s e -> c (b s) e', c=n_channels)
            # Create hook to prepend calib embeds and add tags
            format_hook = lambda x: self._format_embeddings(x, calib_embeds)
            # 0 for primary, 1 for calibration, 2 for special tokens
        else:
            # Create hook to add tags
            format_hook = lambda x: self._format_embeddings(x)
            calib_return_embeds = None
            calib_targets = None

        ### Single-channel module ###
        
        # Format inputs and run through the model
        primary_input = rearrange(primary_input, 'b s c -> (b c) s 1')
        if self.sc_encoder is not None:
            if sc_sm_mask is not None:
                # Expand the mask to fit the rearranged inputs
                sc_sm_mask = repeat(sc_sm_mask, 'b s -> (b c) s', c=n_channels)
            sc_outputs = self.sc_encoder(
                primary_input,
                sm_mask = sc_sm_mask,
                embed_hook = format_hook)
            sc_embeds = sc_outputs['embeddings']

            # Return tensors rearranged to match original input shape order
            # The rearranging makes training easier later
            sc_return_embeds = rearrange(sc_embeds,
                '(b c) s e -> b s c e', c=n_channels)
                
            sc_targets = sc_outputs['targets']
            if sc_targets is not None:
                sc_targets = rearrange(sc_targets,
                    '(b c) s e -> b s c e', c=n_channels)

            format_hook = None # Make sure it doesn't get used again for mc encoder
        else:
            sc_embeds = primary_input
            sc_return_embeds = None
            sc_targets = None
        
        sc_embeds = rearrange(sc_embeds, '(b c) s e -> b c s e', c=n_channels)

        # Combine channels into a mixed channel
        combined_embeds = apply_channel_combine_func(
            self.config['channel_combine_func'], sc_embeds)
        output_embeds = combined_embeds

        ### Multi-channel module ###

        if self.mc_encoder is not None:
            mc_outputs = self.mc_encoder(
                combined_embeds,
                sm_mask = mc_sm_mask,
                embed_hook = format_hook)
            mc_embeds = mc_outputs['embeddings']
            mc_targets = mc_outputs['targets']
            output_embeds = mc_embeds
        else:
            mc_embeds = None
            mc_targets = None

        primary_out_mask = torch.zeros((output_embeds.shape[0], output_embeds.shape[1]),
            device=output_embeds.device)
        primary_seq_len = sc_embeds.shape[2] - 1
        if self.calibration_model is not None:
            primary_seq_len -= self.calibration_model.embed_seq_len + 1
        primary_out_mask[:, -primary_seq_len:] = 1

        ### LSTM head ###

        if self.lstm_head is not None:
            # Run the embeddings through the LSTM head
            lstm_embeds, lstm_hidden = self.lstm_head(output_embeds)
        else:
            lstm_embeds = None
            lstm_hidden = None

        ### Returns ###

        return {
            # Final embeddings of shape (batch_size, seq_len, embed_dim)
            'embeddings': output_embeds,
            'lstm_embeddings': lstm_embeds,
            'lstm_hidden': lstm_hidden,
            'mc_embeddings': mc_embeds,
            'mc_targets': mc_targets,
            'sc_embeddings': sc_return_embeds,
            'sc_targets': sc_targets,
            'calib_embeddings': calib_return_embeds,
            'calib_targets': calib_targets,
            'primary_embeddings_mask': primary_out_mask,
        }


class NeuroDecoder(nn.Module):
    def __init__(self,
                 config: dict,
                 encoder: Optional[NeuroSignalEncoder] = None):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        self.ds_config = config['downstream']

        # Load the encoder model if necessary
        if encoder is None:
            self.encoder = init_model(config)
            
            # Add LSTM layer to model before loading weights if CPC was used
            if self.config['train_method'].lower() == 'cpc':
                self.encoder.add_lstm_head(self.model_config['lstm_embedding_dim'])
                if self.model_config['save_path'] is not None:
                    print('Loading the pretrained model for downstream finetuning.')
                    self.encoder.load_state_dict(torch.load(self.model_config['save_path']))
            else:
                # MSM was not trained with the LSTM head, so it should be added after
                # the weights for the rest of the model are loaded
                if self.model_config['save_path'] is not None:
                    print('Loading the pretrained model for downstream finetuning.')
                    self.encoder.load_state_dict(torch.load(self.model_config['save_path']))
                self.encoder.add_lstm_head(self.model_config['lstm_embedding_dim'])
        else:
            self.encoder = encoder
            
        if self.ds_config['use_lstm']:
            input_size = self.model_config['lstm_embedding_dim']
            self.decoder = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.ds_config['n_classes']))
        else:
            input_size = self.encoder.embed_seq_len * self.encoder.embedding_dim
            self.decoder = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.ds_config['n_classes']))


    def forward(self,
                primary_input: Tensor,
                calibration_input: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the decoder, that takes stimulus inputs and outputs
        a predicted class.

        Args:
            primary_input: Tensor of shape (batch_size, n_timesteps, n_channels)
            calibration_input: Tensor of shape (1, n_timesteps, n_channels)

        Returns:
            Tensor of shape (batch_size, n_classes)
        """
        # Pass the input signals through the encoder
        if isinstance(self.encoder, NeuroSignalEncoder):
            output_dict = self.encoder(primary_input, calibration_input=calibration_input)

            if not self.ds_config['use_lstm']:
                output_embeds = output_dict['embeddings']
                output_embeds = output_embeds.reshape(output_embeds.shape[0], -1)
            else:
                lstm_embeds = output_dict['lstm_embeddings'] # Sequence of all hidden outputs
                primary_mask = output_dict['primary_embeddings_mask']

                # Select the embeddings corresponding to the primary sequence
                selected_embeds = lstm_embeds.masked_select(primary_mask.unsqueeze(-1).bool())
                primary_embeds = selected_embeds.reshape(lstm_embeds.shape[0], -1, lstm_embeds.shape[2])

                # Select the emebddings at the end of the stimulus response time
                target_output_idx = torch.tensor(
                    (self.ds_config['n_stimulus_samples'] / \
                    self.ds_config['tmax_samples']) \
                    * primary_embeds.shape[1]).ceil().type(torch.int64)
                target_output_idx = target_output_idx.to(primary_embeds.device)
                output_embeds = primary_embeds.index_select(dim=1, index=target_output_idx)
                output_embeds = output_embeds.squeeze(1)
        else:
            if calibration_input is not None:
                primary_input = torch.cat((calibration_input, primary_input), dim=1)
            output_dict = self.encoder(primary_input)
            
            if self.ds_config['use_lstm']:
                output_embeds = output_dict['lstm_hidden'][0].squeeze(0)
            else:
                output_embeds = output_dict['embeddings']
                output_embeds = output_embeds.reshape(output_embeds.shape[0], -1)

        # Pass the embeddings through the decoder
        class_logits = self.decoder(output_embeds)

        return class_logits

def init_model(config: dict) -> nn.Module:
    """
    Initialize a model based on the model type set in the config.
    """
    model_config = config['model']
    if config['model_type'] == 'neuro_signal_encoder':
        return NeuroSignalEncoder(model_config)
    elif config['model_type'] == 'wav2vec':
        return Wav2Vec.from_config(model_config)
    else:
        raise ValueError('Unknown model type: {}'.format(config['model_type']))