from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb


def format_inputs(inputs: List[Tensor], config: Dict) -> List[Tensor]:
    """
    Reshapes the inputs and moves them to the correct device.

    Args:
        inputs: List of tensors of shape (batch_size, n_channels, n_timesteps)
            or (n_channels, n_timesteps)
        config: Dictionary containing the config data

    Returns:
        List of tensors of shape (batch_size, n_timesteps, n_channels)
    """
    new_inputs = []
    for i in range(len(inputs)):
        new_inputs.append(inputs[i])
        if inputs[i] is None:
            continue

        # If the inputs are in the shape (n_channels, n_timesteps)
        if len(new_inputs[i].shape) == 2:
            new_inputs[i] = new_inputs[i].unsqueeze(0)
        # Reshape the inputs to (batch_size, n_timesteps, n_channels)
        new_inputs[i] = new_inputs[i].permute(0, 2, 1)
        # Move the inputs to the correct device
        new_inputs[i] = new_inputs[i].to(config['device'])

    return new_inputs

# TODO: This is hardcoded for the MEG colors dataset, fix that
def format_labels(labels: Tensor, config: Dict) -> Tensor:
    """
    Reshapes the labels, moves them to the correct device,
    and converts the range to the range [0, n_classes - 1].

    Args:
        labels: Tensor of shape (batch_size, 1)
        config: Dictionary containing the config data

    Returns:
        Tensor of shape (batch_size,)
    """
    labels = labels.squeeze(1)
    labels = labels - 1
    labels = labels.to(config['device'])
    return labels

def train_downstream(
    model: nn.Module,
    config: dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None):
    """
    Trains the model on a downstream task.
    The task is currently hardcoded to be the MEG colors dataset.
    
    Args:
        model: The decoder model to train.
        config: The training configuration.
        train_loader: The data loader for training.
        val_loader: The data loader for validation.
        test_loader: The data loader for testing.
    """
    ds_config = config['downstream']

    optimizer = optim.Adam(model.parameters(), lr=ds_config['learning_rate'])

    # Calculate initial validation loss
    val_stats = validate(model, val_loader, config)
    wandb.log({'val_' + k: v for k, v in val_stats.items()})

    n_batches = len(train_loader)
    for epoch in range(ds_config['train_epochs']):
        epoch_losses = []
        epoch_accs = []

        batch_losses = []
        batch_preds = []
        batch_labels = []

        model.train()
        for batch_idx, data in enumerate(train_loader):
            # Get and format the data for the batch
            primary_input = data['primary_input']
            calibration_input = data['calibration_input']
            labels = data['label']

            primary_input, calibration_input = format_inputs(
                (primary_input, calibration_input), config)
            labels = format_labels(labels, config)

            # Run the data through the model
            logits = model(primary_input, calibration_input)

            # Update the result buffers
            preds = logits.argmax(dim=1)
            batch_preds.extend(preds.detach().cpu().numpy())
            batch_labels.extend(labels.detach().cpu().numpy())

            # Calculate the loss and update the model weights
            loss = F.cross_entropy(logits, labels)
            batch_losses.append(loss)

            # Minibatch update because samples can only be passed
            # through the decoder one at a time (current limitation)
            if (batch_idx + 1) % ds_config['batch_size'] == 0 or \
               batch_idx == n_batches - 1:
                batch_loss = torch.mean(torch.stack(batch_losses))

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # Calculate accuracy
                accuracy = np.sum(np.array(batch_preds) == np.array(batch_labels)) \
                    / len(batch_labels)
                batch_loss = batch_loss.item()

                # Log updates to Wandb
                wandb.log({
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'loss': batch_loss,
                    'accuracy': accuracy})

                epoch_losses.append(batch_loss)
                epoch_accs.append(accuracy)

                batch_preds = []
                batch_labels = []
                batch_losses = []

            # Log the epoch, batch, and loss
            if (batch_idx + 1) % (ds_config['batch_size'] * ds_config['log_interval']) == 0 or \
               (batch_idx + 1) == n_batches:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch + 1, batch_idx + 1, n_batches,
                    100. * (batch_idx + 1) / n_batches))
                avg_loss = np.mean(epoch_losses[-ds_config['log_interval']:])
                avg_acc = np.mean(epoch_accs[-ds_config['log_interval']:])
                print(f'loss: {avg_loss:.5f}\tacc: {100*avg_acc:.2f}%')

            # Do validation at the end of each epoch
            # and every `ds_config['val_interval']` batches
            do_validation = val_loader is not None and \
                    (ds_config['val_interval'] is not None and \
                    (batch_idx + 1) % (ds_config['batch_size'] * ds_config['val_interval']) == 0) or \
                    (batch_idx + 1) == n_batches
            if do_validation:
                val_stats = validate(model, val_loader, config)
                # Log losses
                print('Validation stats:')
                print('val_loss: {:.4f}\tval_acc: {:.2f}%'.format(
                    val_stats['loss'], 100*val_stats['accuracy']))
                print()
                wandb.log({'val_' + k: v for k, v in val_stats.items()})
                model.train()
    
    # Test the model and log results
    print('Finished training, now testing...')
    test_stats = validate(model, test_loader, config)
    print('\nTest stats:')
    print('test_loss: {:.4f}\ttest_acc: {:.2f}%'.format(
        test_stats['loss'], 100*test_stats['accuracy']))
    wandb.log({'test_' + k: v for k, v in test_stats.items()})

def validate(model: nn.Module, val_loader: DataLoader, config: dict):
    """
    Tests the model on the validation data.
    
    Args:
        model: The decoder model to run validation on.
        val_loader: The data loader for validation.
        config: The training configuration.
    """
    all_preds = []
    all_labels = []
    all_losses = []

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            # Get and format the data for the batch
            primary_input = data['primary_input']
            calibration_input = data['calibration_input']
            labels = data['label']

            primary_input, calibration_input = format_inputs(
                (primary_input, calibration_input), config)
            labels = format_labels(labels, config)

            # Run the data through the model
            logits = model(primary_input, calibration_input)

            # Update the result buffers
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            # Calculate the loss and update the model weights
            loss = F.cross_entropy(logits, labels)
            all_losses.append(loss.item())

    val_loss = np.mean(all_losses)
    val_accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) \
        / len(all_labels)
    
    return {
        'loss': val_loss,
        'accuracy': val_accuracy
    }

