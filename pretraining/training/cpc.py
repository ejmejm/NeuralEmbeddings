from typing import List, Optional

from einops import rearrange
import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
import wandb

from models import NeuroSignalEncoder


def calculate_cpc_loss(
    output_dict: dict[str, Tensor],
    bilinear_layers: List[nn.Module]) -> float:
    """
    Calculates the contrstive predictive coding loss.

    Args:
        output_dict: Outputs of the encoder model.
        bilinear_layers: Layers used to calculate mutual info.

    Returns:
        The cpc loss.
    """
    # Unpack outputs
    out_embeds = output_dict['embeddings']
    lstm_hiddens = output_dict['lstm_embeddings']
    seq_len = out_embeds.shape[1]
    n_pred_steps = len(bilinear_layers)
    # TODO: Add masking for calibration input and special tokens
    # Rearrange the sequences as batches so they can
    # be passed to the bilinear layer all at once
    batch_embeds = rearrange(out_embeds, 'b s e -> (b s) e')

    # Each index i corresponds to the i-th prediction step avg loss
    cpc_losses = [[] for _ in range(n_pred_steps)]
    # Calculate the cpc loss
    for seq_idx in range(seq_len - n_pred_steps):
        for pred_step in range(n_pred_steps):
            # Duplicate the lstm hidden state for get one for each 
            # output embedding in the entire sequence
            target_hiddens = lstm_hiddens[:, seq_idx:seq_idx+1].repeat(1, seq_len, 1)
            batch_hiddens = rearrange(target_hiddens, 'b s e -> (b s) e')
            # Calculate the mutual information
            # TODO: Speed this up by only using a portion of the full sequence
            mis = bilinear_layers[pred_step](batch_embeds, batch_hiddens)
            # Recover the sequences
            mis = rearrange(mis, '(b s) 1 -> b s',
                b=out_embeds.shape[0], s=seq_len)

            # Calculate the losses
            mis = torch.exp(mis)
            positive_samples = mis[:, seq_idx + pred_step]
            seq_sums = torch.sum(mis, dim=1)
            losses = -torch.log(positive_samples / seq_sums)
            cpc_losses[pred_step].extend(losses)

    cpc_losses = torch.stack([torch.stack(l) for l in cpc_losses])
    cpc_losses = cpc_losses.mean(dim=1)
    return cpc_losses

def log_losses(
    losses: List[float],
    prefix: str = '',
    do_print : bool = True):
    """
    Logs the losses to wandb.

    Args:
        losses: The i-th loss is the loss for the i-th prediction step.
        prefix: A prefix to add to the variable names.
        print: Whether to print the losses.
    """
    # Format loss names
    total_loss = np.sum(losses)
    loss_map = {f'{prefix}{i+1}_step_loss': loss for i, loss in enumerate(losses)}
    total_loss_name = f'{prefix}loss'
    
    # Log to wandb
    loss_map[total_loss_name] = total_loss
    wandb.log(loss_map)

    # Print loss
    if do_print:
        print(f'{total_loss_name}: {total_loss:.5f}')

def train_with_cpc(
    model: NeuroSignalEncoder,
    config: dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None):
    """
    Trains the model with masked sequence modeling.
    1. Run a batch through the model.
    2. Calculate the cpc losses and update the model.
    3. Repeat.
    
    Args:
        model: The model to train.
        config: The training configuration.
        train_loader: The data loader for training.
        val_loader: The data loader for validation.
    """
    cpc_config = config['cpc_params']
    # Add bilinear layers used for calculating the CPC loss
    bilinear_layers = [
        nn.Bilinear(config['model']['embedding_dim'], cpc_config['embedding_dim'],
                    1, bias=False, device=config['device'])
        for _ in range(cpc_config['n_pred_steps'])]
    all_params = list(model.parameters()) + [l.weight for l in bilinear_layers]
    optimizer = optim.Adam(all_params, lr=config['learning_rate'])
    # Create lr scheduler
    if cpc_config['scheduler_enabled']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min')

    # Calculate initial validation loss
    val_losses = validate(model, bilinear_layers, val_loader, config)
    log_losses(val_losses, prefix='val_')

    for epoch in range(config['train_epochs']):
        model.train()
        batch_losses = []
        for batch_idx, data in enumerate(train_loader):
            # Unpack training data
            primary_input = data['primary_input'].to(config['device'])
            calib_input = data['calibration_input'].to(config['device'])

            # Shuffle the primary input
            if config['shuffle_in_batch']:
                primary_input = primary_input[torch.randperm(primary_input.shape[0])]

            # Run model
            output_dict = model(primary_input, calibration_input=calib_input)

            # Calculate the masked sequence modeling losses
            cpc_losses = calculate_cpc_loss(output_dict, bilinear_layers)
            cpc_loss = torch.sum(cpc_losses)
            batch_losses.append(cpc_loss.item())

            # Log epoch
            wandb.log({'epoch': epoch})
            # Log learning rates
            wandb.log({'lr': optimizer.param_groups[0]['lr']})
            # Log losses
            log_losses(cpc_losses.detach().cpu().numpy(), do_print=False)

            # Calcualte losses and update weights
            optimizer.zero_grad()
            cpc_loss.backward()
            optimizer.step()

            # Log the epoch, batch, and loss
            if (batch_idx + 1) % config['log_interval'] == 0 or \
               (batch_idx + 1) == len(train_loader):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch + 1, batch_idx + 1, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader)))
                avg_loss = np.mean(batch_losses[-config['log_interval']:])
                print(f'loss: {avg_loss:.5f}')

            # Do validation at the end of each epoch
            # and every `config['val_interval']` batches
            do_validation = val_loader is not None and \
                    (config['val_interval'] is not None and \
                    (batch_idx + 1) % config['val_interval'] == 0) or \
                    (batch_idx + 1) == len(train_loader)
            if do_validation:
                val_losses = validate(model, bilinear_layers, val_loader, config)
                print('Validation losses:')
                # Log losses
                log_losses(val_losses)
                if cpc_config['scheduler_enabled']:
                    scheduler.step(np.mean(val_losses))
                print()
                model.train()
    
def validate(model: NeuroSignalEncoder, bilinear_layers: List[nn.Module],
             val_loader: DataLoader, config: dict):
    """
    Validates the model on the validation data.
    
    Args:
        model: The model to validate.
        bilinear_layers: Layers used for calculating the CPC loss.
        val_loader: The data loader for validation.
        config: The training configuration.
    """
    model.eval()
    val_losses = []
    with torch.no_grad():
        for data in val_loader:
            # Unpack training data
            primary_input = data['primary_input'].to(config['device'])
            calib_input = data['calibration_input'].to(config['device'])

            # Run model
            output_dict = model(primary_input, calibration_input=calib_input)

            # Calculate the masked sequence modeling losses
            cpc_losses = calculate_cpc_loss(output_dict, bilinear_layers)
            val_losses.append(cpc_losses.detach().cpu().numpy())

    val_losses = np.array(val_losses).mean(axis=0)

    return val_losses