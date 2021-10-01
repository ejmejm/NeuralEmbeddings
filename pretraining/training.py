from typing import Optional

import numpy as np
import torch
from torch import optim, Tensor
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from data_loading import load_data, prepare_meg_data
from models import Wav2Vec

def generate_mask(data: Tensor, msm_config: dict) -> Tensor:
    """
    Generates a random mask for the data.
    
    Args:
        data_shape: The shape of the data.
        msm_config: The configuration for masked sequence modeling.
    """
    # Generate a random mask for the data
    # TODO: Make use of min and max length mask params
    mask = torch.rand(data.shape[0], data.shape[1]) < msm_config['mask_prob']
    mask = mask.to(data.device)

    return mask

def train_with_msm(
    model: Wav2Vec,
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
        for batch_idx, data in enumerate(train_loader):
            data = data.to(config['device'])
            
            target = model(data).detach()
            mask = generate_mask(target, msm_config)
            masked_output = model(data, sm_mask=mask)
            # TODO: Double check this detaching of the target is correct

            # Train the model on the masked sequence
            # TODO: Add a mask for embedding padding
            loss = mse_loss(masked_output, target, reduce=False)
            loss = loss.sum(dim=2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % config['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        if val_loader is not None:
            val_loss = validate(model, val_loader, config)
            print('Validation loss: {:.4f}'.format(val_loss))

def validate(model: Wav2Vec, val_loader: DataLoader, config: dict):
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

            target = model(data).detach()
            mask = generate_mask(target, msm_config)
            masked_output = model(data, sm_mask=mask)
            # TODO: Double check this detaching of the target is correct

            val_loss = mse_loss(masked_output, target, reduce=False)
            val_loss = val_loss.sum(dim=2).mean().item()
            val_losses.append(val_loss)
    val_loss = np.mean(val_losses)
    return val_loss