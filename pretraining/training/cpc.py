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
    bilinear_layers: List[nn.Module],
    mi_seq_radius: Optional[int] = None) -> float:
    """
    Calculates the contrstive predictive coding loss.

    Args:
        output_dict: Outputs of the encoder model.
        bilinear_layers: Layers used to calculate mutual info.
        mi_seq_radius: Maximum sequence length to use in summation
            in the denomicatior of the loss.

    Returns:
        The cpc loss.
    """
    # Unpack outputs
    out_embeds = output_dict['embeddings']
    lstm_hiddens = output_dict['lstm_embeddings']

    primary_embeds_mask = output_dict['primary_embeddings_mask']
    primary_embeds_mask = primary_embeds_mask.bool().unsqueeze(-1)

    selected_out_embeds = out_embeds.masked_select(primary_embeds_mask)
    # This will break if different primary seq lengths are used within the same batch
    out_embeds = selected_out_embeds.reshape(out_embeds.shape[0], -1, out_embeds.shape[2])

    selected_lstm_hiddens = lstm_hiddens.masked_select(primary_embeds_mask)
    lstm_hiddens = selected_lstm_hiddens.reshape(lstm_hiddens.shape[0], -1, lstm_hiddens.shape[2])

    seq_len = out_embeds.shape[1]
    n_pred_steps = len(bilinear_layers)
    # Rearrange the sequences as batches so they can
    # be passed to the bilinear layer all at once
    if mi_seq_radius is None:
        batch_embeds = rearrange(out_embeds, 'b s e -> (b s) e')

    # Each index i corresponds to the i-th prediction step avg loss
    cpc_losses = [[] for _ in range(n_pred_steps)]
    # Calculate the cpc loss
    for seq_idx in range(seq_len - 1):
        if mi_seq_radius is None:
            # Normally you need an LSTM output for each embedding sequence element
            mod_seq_len = seq_len
            seq_start_idx = 0
        else:
            # But for if mi_seq_radius is defined, we only need the elements
            # within a radius of the current element
            seq_start_idx = max(0, seq_idx - mi_seq_radius)
            seq_end_idx = min(seq_len, seq_idx + mi_seq_radius + 1)
            mod_seq_len = seq_end_idx - seq_start_idx

            partial_embeds = out_embeds[:, seq_start_idx:seq_end_idx]
            batch_embeds = rearrange(partial_embeds, 'b s e -> (b s) e')
        
        # Duplicate the lstm hidden state for get one for each 
        # output embedding in the entire sequence
        target_hiddens = lstm_hiddens[:, seq_idx:seq_idx+1].repeat(1, mod_seq_len, 1)
        batch_hiddens = rearrange(target_hiddens, 'b s e -> (b s) e')

        for pred_step in range(min(n_pred_steps, seq_len - seq_idx - 1)):
            # Calculate the mutual information
            mis = bilinear_layers[pred_step](batch_embeds, batch_hiddens)
            # Recover the sequences
            mis = rearrange(mis, '(b s) 1 -> b s',
                b=out_embeds.shape[0], s=mod_seq_len)

            # Calculate the losses
            mis = torch.exp(mis)
            positive_samples = mis[:, (seq_idx - seq_start_idx) + pred_step + 1]
            seq_sums = torch.sum(mis, dim=1)
            losses = -torch.log(positive_samples / seq_sums)
            cpc_losses[pred_step].extend(losses)

        if seq_len - seq_idx - 1 < n_pred_steps:
            for i in range(n_pred_steps - (seq_len - seq_idx - 1)):
                cpc_losses[pred_step + i + 1].extend(torch.zeros_like(losses))

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
    model_config = config['model']
    # Add bilinear layers used for calculating the CPC loss
    bilinear_layers = [
        nn.Bilinear(model_config['embedding_dim'], model_config['lstm_embedding_dim'],
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

    n_batches = len(train_loader)
    if 'epoch_early_cutoff' in config and \
            config['epoch_early_cutoff'] is not None:
        n_batches = int(n_batches * config['epoch_early_cutoff'])

    for epoch in range(config['train_epochs']):
        model.train()
        batch_losses = []
        for batch_idx, data in enumerate(train_loader):
            if batch_idx >= n_batches:
                print('Stopping batch early for specified early cutoff')
                break
            
            # Unpack training data
            primary_input = data['primary_input'].to(config['device'])
            if model.calibration_model is not None:
                calib_input = data['calibration_input'].to(config['device'])
            else:
                calib_input = None

            # Shuffle the primary input
            if config['shuffle_in_batch']:
                primary_input = primary_input[torch.randperm(primary_input.shape[0])]

            # Run model
            output_dict = model(primary_input, calibration_input=calib_input)

            # Calculate the masked sequence modeling losses
            cpc_losses = calculate_cpc_loss(output_dict, bilinear_layers,
                cpc_config['mi_seq_radius'])
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
               (batch_idx + 1) == n_batches:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch + 1, batch_idx + 1, n_batches,
                    100. * (batch_idx + 1) / n_batches))
                avg_loss = np.mean(batch_losses[-config['log_interval']:])
                print(f'loss: {avg_loss:.5f}')

            # Do validation at the end of each epoch
            # and every `config['val_interval']` batches
            do_validation = val_loader is not None and \
                    (config['val_interval'] is not None and \
                    (batch_idx + 1) % config['val_interval'] == 0) or \
                    (batch_idx + 1) == n_batches
            if do_validation:
                val_losses = validate(model, bilinear_layers, val_loader, config)
                print('Validation losses:')
                # Log losses
                log_losses(val_losses, prefix='val_')
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
    cpc_config = config['cpc_params']

    model.eval()
    val_losses = []
    with torch.no_grad():
        for data in val_loader:
            # Unpack training data
            primary_input = data['primary_input'].to(config['device'])
            if model.calibration_model is not None:
                calib_input = data['calibration_input'].to(config['device'])
            else:
                calib_input = None

            # Run model
            output_dict = model(primary_input, calibration_input=calib_input)

            # Calculate the masked sequence modeling losses
            cpc_losses = calculate_cpc_loss(output_dict, bilinear_layers,
                cpc_config['mi_seq_radius'])
            val_losses.append(cpc_losses.detach().cpu().numpy())

    val_losses = np.array(val_losses).mean(axis=0)

    return val_losses