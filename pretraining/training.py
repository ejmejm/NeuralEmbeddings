from typing import Optional, Tuple

import numpy as np
import torch
from torch import optim, Tensor
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from models import NeuroSignalEncoder

def generate_mask(mask_shape: Tuple, msm_config: dict, device: str = 'cpu') -> Tensor:
    """
    Generates a random mask for the data.
    
    Args:
        mask_shape: The shape of the mask, should be (batch_size, seq_len).
        msm_config: The masked sequence modeling configuration.
        device: The device to generate the mask on.
    
    Returns:
        The generated mask, shape (batch_size, seq_len).
    """
    # Generate initial mask
    mask = torch.rand(mask_shape[0], mask_shape[1]) < msm_config['mask_prob']
    mask = mask.type(torch.long)

    # Expand masks to the correct sizes
    for batch_idx in range(mask.shape[0]):
        seq_idx = 0
        while seq_idx < mask.shape[1]:
            if mask[batch_idx, seq_idx] == 1:
                mask_len = np.random.randint(
                    msm_config['min_mask_len'], msm_config['max_mask_len'] + 1)
                mask[batch_idx, seq_idx:seq_idx + mask_len] = 1
                seq_idx += mask_len
            else:
                seq_idx += 1
    mask = mask.to(device)

    return mask

def calculate_msm_losses(
    output_dict: dict[str, Tensor],
    sc_mask: Tensor,
    mc_mask: Tensor) -> dict[str, Tensor]:
    """
    Calculates the masked sequence modeling losses.

    Args:
        output_dict: The output dictionary from the model.
        sc_mask: The single-channel model mask.
        mc_mask: The multi-channel model mask.

    Returns:
        A dictionary containing the separate and combined MSM losses.
            - 'loss': The total MSM loss.
            - 'sc_loss': The single-channel MSM loss.
            - 'mc_loss': The multi-channel MSM loss.
    """
    sc_loss = 0
    if output_dict['sc_targets'] is not None:
        sc_embeds = output_dict['sc_embeddings']
        sc_targets = output_dict['sc_targets']
        
        selection_mask = sc_mask.type(torch.bool).unsqueeze(2)
        masked_sc_embeds = sc_embeds.masked_select(selection_mask)
        masked_sc_targets = sc_targets.masked_select(selection_mask)
        sc_loss = mse_loss(masked_sc_embeds, masked_sc_targets.detach(),
            reduce=True, reduction='mean')

    # Calculate MSM loss for the multi-channel encoder if used
    mc_loss = 0
    if output_dict['mc_targets'] is not None:
        mc_embeds = output_dict['mc_embeddings']
        mc_targets = output_dict['mc_targets']
        
        selection_mask = mc_mask.type(torch.bool).unsqueeze(2)
        masked_mc_embeds = mc_embeds.masked_select(selection_mask)
        masked_mc_targets = mc_targets.masked_select(selection_mask)
        mc_loss = mse_loss(masked_mc_embeds, masked_mc_targets.detach(),
            reduce=True, reduction='mean')

    # Calculate the total loss
    loss = sc_loss + mc_loss

    return {
        'loss': loss,
        'sc_loss': sc_loss,
        'mc_loss': mc_loss
    }

def train_with_msm(
    model: NeuroSignalEncoder,
    config: dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None):
    """
    Trains the model with masked sequence modeling.
    1. Generate a random mask for each sequence.
    2. Train the model on the masked sequence.
    3. Repeat.
    
    Args:
        model: The model to train.
        config: The training configuration.
        train_loader: The data loader for training.
        val_loader: The data loader for validation.
    """
    msm_config = config['msm_params']

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['train_epochs']):
        model.train()
        batch_losses = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(config['device'])
            
            mask_shape = (data.shape[0], model.embed_seq_len)
            sc_mask = generate_mask(mask_shape, msm_config, device=config['device'])
            mc_mask = generate_mask(mask_shape, msm_config, device=config['device'])
            output_dict = model(data, sc_sm_mask=sc_mask, mc_sm_mask=mc_mask)

            # Calculate the masked sequence modeling losses
            msm_losses = calculate_msm_losses(output_dict, sc_mask, mc_mask)
            loss = msm_losses['loss']

            batch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % config['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    np.mean(batch_losses[-config['log_interval']:])))
        if val_loader is not None:
            val_loss = validate(model, val_loader, config)
            print('Validation loss: {:.4f}'.format(val_loss))

def validate(model: NeuroSignalEncoder, val_loader: DataLoader, config: dict):
    """
    Validates the model on the validation data.
    
    Args:
        model: The model to validate.
        val_loader: The data loader for validation.
        config: The training configuration.
    """
    msm_config = config['msm_params']
    model.eval()
    val_losses = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(config['device'])

            mask_shape = (data.shape[0], model.embed_seq_len)
            sc_mask = generate_mask(mask_shape, msm_config, device=config['device'])
            mc_mask = generate_mask(mask_shape, msm_config, device=config['device'])
            output_dict = model(data, sc_sm_mask=sc_mask, mc_sm_mask=mc_mask)

            # Calculate the masked sequence modeling losses
            msm_losses = calculate_msm_losses(output_dict, sc_mask, mc_mask)
            val_loss = msm_losses['loss']

            val_losses.append(val_loss.item())
    val_loss = np.mean(val_losses)
    return val_loss