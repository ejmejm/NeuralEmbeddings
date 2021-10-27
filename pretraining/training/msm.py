from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import optim, Tensor
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import wandb

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
        # If this mask is empty, add a single entry
        if sum(mask[batch_idx]) == 0:
            mask[batch_idx, np.random.randint(0, mask.shape[1])] = 1
            
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
    sc_mask: Optional[Tensor] = None,
    mc_mask: Optional[Tensor] = None,
    calib_mask: Optional[Tensor] = None) -> dict[str, Tensor]:
    """
    Calculates the masked sequence modeling losses.

    Args:
        output_dict: The output dictionary from the model.
        sc_mask: The single-channel model mask.
        mc_mask: The multi-channel model mask.
        calib_mask: The calibration mask.

    Returns:
        A dictionary containing the separate and combined MSM losses.
            - 'loss': The total MSM loss.
            - 'sc': The single-channel MSM loss.
            - 'mc': The multi-channel MSM loss.
            - 'calib': The multi-channel MSM loss.
    """
    # Calculate MSM loss for the single-channel encoder if used
    sc_loss = 0
    if output_dict['sc_targets'] is not None:
        sc_embeds = output_dict['sc_embeddings']
        sc_targets = output_dict['sc_targets']
        
        if sc_mask is not None:
            selection_mask = sc_mask.type(torch.bool).unsqueeze(2).unsqueeze(3)
            masked_sc_embeds = sc_embeds.masked_select(selection_mask)
            masked_sc_targets = sc_targets.masked_select(selection_mask)
            sc_loss = mse_loss(masked_sc_embeds, masked_sc_targets.detach(),
                reduce=True, reduction='mean')
        else:
            sc_loss = mse_loss(sc_embeds, sc_targets.detach(),
                reduce=True, reduction='mean')

    # Calculate MSM loss for the multi-channel encoder if used
    mc_loss = 0
    if output_dict['mc_targets'] is not None:
        mc_embeds = output_dict['mc_embeddings']
        mc_targets = output_dict['mc_targets']
        
        if mc_mask is not None:
            selection_mask = mc_mask.type(torch.bool).unsqueeze(2)
            masked_mc_embeds = mc_embeds.masked_select(selection_mask)
            masked_mc_targets = mc_targets.masked_select(selection_mask)
            mc_loss = mse_loss(masked_mc_embeds, masked_mc_targets.detach(),
                reduce=True, reduction='mean')
        else:
            mc_loss = mse_loss(mc_embeds, mc_targets.detach(),
                reduce=True, reduction='mean')

    # Calculate MSM loss for the calibration encoder if used
    calib_loss = 0
    if output_dict['calib_targets'] is not None:
        calib_embeds = output_dict['calib_embeddings']
        calib_targets = output_dict['calib_targets']
        
        if calib_mask is not None:
            selection_mask = calib_mask.type(torch.bool).unsqueeze(2).unsqueeze(3)
            masked_calib_embeds = calib_embeds.masked_select(selection_mask)
            masked_calib_targets = calib_targets.masked_select(selection_mask)
            calib_loss = mse_loss(masked_calib_embeds, masked_calib_targets.detach(),
                reduce=True, reduction='mean')
        else:
            calib_loss = mse_loss(calib_embeds, calib_targets.detach(),
                reduce=True, reduction='mean')

    # Calculate the total loss
    loss = sc_loss + mc_loss + calib_loss

    return {
        'loss': loss,
        'sc': sc_loss,
        'mc': mc_loss,
        'calib': calib_loss
    }

def calculate_embed_std(
    model: NeuroSignalEncoder,
    data_loader: DataLoader,
    config: dict) -> dict[str, Tensor]:
    """
    Calculates the average standard deviation per embedding dimension
    over several batches of data. This function can be used to help determine
    whether the model is underfitting or overfitting.

    Args:
        model: The model to calculate the embeddings.
        data_loader: Provides data to be used.
        config: Config dict.

    Returns:
        A dictionary containing the average standard deviation for each module.
    """
    model.eval()
    msm_config = config['msm_params']

    data = next(iter(data_loader))
    primary_input = data['primary_input'].to(config['device'])
    calib_input = data['calibration_input'].to(config['device'])

    sc_std = None
    mc_std = None
    calib_std = None
    std_dict = {
        'sc_embed_std': [] if model.sc_encoder is not None else None,
        'mc_embed_std': [] if model.mc_encoder is not None else None,
        'calib_embed_std': [] if model.calibration_model is not None else None
    }

    # Perform with 3 different masks
    for _ in range(3):
        # Create primary masks
        primary_mask_shape = (primary_input.shape[0], model.embed_seq_len)
        sc_mask = generate_mask(primary_mask_shape, msm_config, device=config['device'])
        mc_mask = generate_mask(primary_mask_shape, msm_config, device=config['device'])

        # Create calibration mask if needed
        calib_mask = None
        if model.calibration_model is not None:
            calib_mask_shape = (calib_input.shape[0], model.calib_embed_seq_len)
            calib_mask = generate_mask(calib_mask_shape, msm_config, device=config['device'])
        else:
            calib_input = None


        first_iter = True
        for data in data_loader:
            primary_input = data['primary_input'].to(config['device'])
            calib_input = data['calibration_input'].to(config['device'])
            
            # Run model
            output_dict = model(
                primary_input,
                calibration_input = calib_input,
                sc_sm_mask = sc_mask,
                mc_sm_mask = mc_mask,
                calib_sm_mask = calib_mask)

            if model.sc_encoder is not None:
                sc_embeds = output_dict['sc_embeddings']
                selection_mask = sc_mask.type(torch.bool).unsqueeze(2).unsqueeze(3)
                masked_sc_embeds = sc_embeds.masked_select(selection_mask)
                if not first_iter:
                    sc_size = masked_sc_embeds.numel()
                    sc_diff = torch.sum((masked_sc_embeds - prev_masked_sc_embeds) ** 2)
                    sc_std = torch.sqrt(sc_diff / (sc_size - 1)).item()

            if model.mc_encoder is not None:
                mc_embeds = output_dict['mc_embeddings']
                selection_mask = mc_mask.type(torch.bool).unsqueeze(2)
                masked_mc_embeds = mc_embeds.masked_select(selection_mask)
                if not first_iter:
                    mc_size = masked_mc_embeds.numel()
                    mc_diff = torch.sum((masked_mc_embeds - prev_masked_mc_embeds) ** 2)
                    mc_std = torch.sqrt(mc_diff / (mc_size - 1)).item()

            if model.calibration_model is not None:
                calib_embeds = output_dict['calib_embeddings']
                selection_mask = calib_mask.type(torch.bool).unsqueeze(2).unsqueeze(3)
                masked_calib_embeds = calib_embeds.masked_select(selection_mask)
                if not first_iter:
                    calib_size = masked_calib_embeds.numel()
                    calib_diff = torch.sum((masked_calib_embeds - prev_masked_calib_embeds) ** 2)
                    calib_std = torch.sqrt(calib_diff / (calib_size - 1)).item()

            if not first_iter:
                if not np.isnan(sc_std):
                    std_dict['sc_embed_std'].append(sc_std)
                if not np.isnan(mc_std):
                    std_dict['mc_embed_std'].append(mc_std)
                if not np.isnan(calib_std):
                    std_dict['calib_embed_std'].append(calib_std)

            prev_masked_sc_embeds = masked_sc_embeds
            prev_masked_mc_embeds = masked_mc_embeds
            prev_masked_calib_embeds = masked_calib_embeds

            first_iter = False
    
    return std_dict


def create_loss_map(model_config: dict) -> dict:
    """
    Creates a dictionary mapping loss names to loss functions.

    Args:
        model_config: The configuration dictionary.

    Returns:
        A dictionary mapping loss names to emtpy lists.
    """
    losses = {'loss': []}
    if model_config['single_channel_module']['enabled']:
        losses['sc'] = []
    if model_config['mixed_channel_module']['enabled']:
        losses['mc'] = []
    if model_config['calibration_module']['enabled']:
        losses['calib'] = []
    return losses

def print_loss_map(losses: dict,  prepend: str = '') -> None:
    """
    Prints the loss map.

    Args:
        losses: The loss map.
        prepend: A string to prepend to the name of each loss.
    """
    base_str = ''
    for key, value in losses.items():
        loss_name = key
        if 'loss' not in loss_name:
            loss_name += '_loss'
        loss_name = prepend + loss_name
        base_str += '{}: {:.4f}\t'.format(loss_name, np.mean(value))
    print(base_str)

def create_optimizers(model: NeuroSignalEncoder, config: dict) -> dict:
    """
    Creates the optimizers for the model, leaving out
    optimizers for disabled modules.

    Args:
        model: The model to create the optimizers for.
        config: The configuration dictionary.

    Returns:
        A dictionary containing the optimizers.
    """
    optimizers = {}
    msm_params = config['msm_params']

    if model.sc_encoder is not None:
        sc_lr = msm_params['sc_lr'] \
            if 'sc_lr' in msm_params \
                and msm_params['sc_lr'] is not None \
            else config['learning_rate']
        optimizer = optim.Adam(model.sc_encoder.parameters(), lr=sc_lr)
        optimizers['sc'] = optimizer

    if model.mc_encoder is not None:
        mc_lr = msm_params['mc_lr'] \
            if 'mc_lr' in msm_params \
                and msm_params['mc_lr'] is not None \
            else config['learning_rate']
        optimizer = optim.Adam(model.mc_encoder.parameters(), lr=mc_lr)
        optimizers['mc'] = optimizer

    if model.calibration_model is not None:
        calib_lr = msm_params['calib_lr'] \
            if 'calib_lr' in msm_params \
                and msm_params['calib_lr'] is not None \
            else config['learning_rate']
        optimizer = optim.Adam(model.calibration_model.parameters(), lr=calib_lr)
        optimizers['calib'] = optimizer

    return optimizers

def update_weights(optimizers: dict, losses: dict):
    """
    Updates the weights of the model based on the losses.

    Args:
        optimizers: Dictionary of the optimizers for the model.
        losses: Dictionary of the losses for the model.
    """
    for key in ['mc', 'sc', 'calib']:
        if key in optimizers:
            optimizers[key].zero_grad()
            # TODO: Make sure there is no memory leak happening here
            losses[key].backward(retain_graph=True)
            optimizers[key].step()

def create_schedulers(optimizers: dict, config: dict) -> dict:
    """
    Creates exponential decaying schedulers for the optimizers.

    Args:
        optimizers: The optimizers for the model.
        config: The configuration dictionary.

    Returns:
        A dictionary containing the schedulers.
    """
    schedulers = {}

    # The decay is split into phases based on the number of modules in the model
    # In each phase, a later stage of the model starts to decay
    # Here, we calculate the ideal gamma needed to meet the desired decay
    n_epochs = config['train_epochs']
    # `phase_decay` is the target decay per phase of the scheduler.
    phase_decay = config['msm_params']['phase_decay']
    n_phases = len(optimizers)
    epochs_per_phase = max(1, n_epochs // n_phases)
    gamma = np.power(phase_decay, 1 / epochs_per_phase)

    for key, optimizer in optimizers.items():
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        schedulers[key] = scheduler
    return schedulers

def update_lrs(schedulers: dict, epoch: int, total_epochs: int) -> dict:
    """
    Updates the learning rates in a phased approach.
    The decay is split into phases based on the number of modules in the model.
    In each phase, a later stage of the model starts to decay.
    The phase length is based on the number of modules and total number of epochs.

    Args:
        schedulers: The schedulers for the model.
        epoch: The current epoch.
        total_epochs: The total number of epochs.
    
    Returns:
        A dictionary containing the new learning rates.
    """
    # Make sure the keys are in the same order as the phases
    keys = [key for key in ['calib', 'sc', 'mc'] if key in schedulers]
    epochs_per_phase = max(1, total_epochs // len(keys))

    # Update schedulers based on the current phase
    for i, key in enumerate(keys):
        if epoch >= i * epochs_per_phase:
            schedulers[key].step()

    # Return the latest learning rates
    return {key: schedulers[key].get_last_lr()[0] for key in keys}

def format_loss_map(losses: dict, prepend='') -> dict:
    """
    Formats the loss map for logging.

    Args:
        losses: The loss map.
        prepend: The string to prepend to the loss name.

    Returns:
        A dictionary containing the formatted losses.
    """
    formatted_losses = {}
    for key, value in losses.items():
        loss_name = key
        if 'loss' not in loss_name:
            loss_name += '_loss'
        loss_name = prepend + loss_name
        if isinstance(value, list):
            value = np.mean(value)
        formatted_losses[loss_name] = value
    return formatted_losses

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
    scheduler_enabled = msm_config['scheduler_enabled']

    optimizers = create_optimizers(model, config)
    # Track the learning rates for logging
    learning_rates = {k: v.param_groups[0]['lr'] \
        for k, v in optimizers.items()}
    if scheduler_enabled:
        schedulers = create_schedulers(optimizers, config)

    # Calculate initial validation loss
    val_losses = validate(model, val_loader, config)
    wandb.log(format_loss_map(val_losses, 'val_'))

    for epoch in range(config['train_epochs']):
        model.train()
        batch_losses = create_loss_map(config['model'])
        for batch_idx, data in enumerate(train_loader):
            # Unpack training data
            primary_input = data['primary_input'].to(config['device'])
            calib_input = data['calibration_input'].to(config['device'])

            # Shuffle the primary input
            if msm_config['shuffle_in_batch']:
                primary_input = primary_input[torch.randperm(primary_input.shape[0])]
            
            # Create primary masks
            primary_mask_shape = (primary_input.shape[0], model.embed_seq_len)
            sc_mask = generate_mask(primary_mask_shape, msm_config, device=config['device'])
            mc_mask = generate_mask(primary_mask_shape, msm_config, device=config['device'])

            # Create calibration mask if needed
            calib_mask = None
            if model.calibration_model is not None:
                calib_mask_shape = (calib_input.shape[0], model.calib_embed_seq_len)
                calib_mask = generate_mask(calib_mask_shape, msm_config, device=config['device'])
            else:
                calib_input = None

            # Run model
            output_dict = model(
                primary_input,
                calibration_input = calib_input,
                sc_sm_mask = sc_mask,
                mc_sm_mask = mc_mask,
                calib_sm_mask = calib_mask)

            # Calculate the masked sequence modeling losses
            msm_losses = calculate_msm_losses(output_dict, sc_mask, mc_mask, calib_mask)
            for loss_type in batch_losses.keys():
                batch_losses[loss_type].append(msm_losses[loss_type].item())

            # Log epoch
            wandb.log({'epoch': epoch})
            # Log losses
            wandb.log(format_loss_map(msm_losses))
            # Log learning rates
            wandb.log({k + '_lr': v for k, v in learning_rates.items()})

            # Calcualte losses and update weights
            update_weights(optimizers, msm_losses)

            if (batch_idx + 1) % config['log_interval'] == 0 or \
               (batch_idx + 1) == len(train_loader):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch + 1, batch_idx + 1, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader)))
                print_loss_map(batch_losses)

            # Do validation at the end of each epoch
            # and every `config['val_interval']` batches
            do_validation = val_loader is not None and \
                    (config['val_interval'] is not None and \
                    (batch_idx + 1) % config['val_interval'] == 0) or \
                    (batch_idx + 1) == len(train_loader)
            if do_validation:
                val_losses = validate(model, val_loader, config)
                print('Validation losses:')
                print_loss_map(val_losses)
                wandb.log(format_loss_map(val_losses, 'val_'))
                print()
                model.train()
        
        # Update learning rates
        if scheduler_enabled:
            learning_rates = update_lrs(schedulers, epoch, config['train_epochs'])
    
    # Log the std of the per-dimension embeddings to help identify underfitting
    std_dict = calculate_embed_std(model, val_loader, config)
    std_dict = {k: np.mean(v) for k, v in std_dict.items() if v is not None}
    wandb.log(std_dict)
    print('Per embedding dim stds:', std_dict)
    
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
    val_losses = create_loss_map(config['model'])
    with torch.no_grad():
        for data in val_loader:
            # Unpack training data
            primary_input = data['primary_input'].to(config['device'])
            calib_input = data['calibration_input'].to(config['device'])
            
            # Create primary masks
            primary_mask_shape = (primary_input.shape[0], model.embed_seq_len)
            sc_mask = generate_mask(primary_mask_shape, msm_config, device=config['device'])
            mc_mask = generate_mask(primary_mask_shape, msm_config, device=config['device'])

            # Create calibration mask if needed
            calib_mask = None
            if model.calibration_model is not None:
                calib_mask_shape = (calib_input.shape[0], model.calib_embed_seq_len)
                calib_mask = generate_mask(calib_mask_shape, msm_config, device=config['device'])
            else:
                calib_input = None

            # Run model
            output_dict = model(
                primary_input,
                calibration_input = calib_input,
                sc_sm_mask = sc_mask,
                mc_sm_mask = mc_mask,
                calib_sm_mask = calib_mask)

            # Calculate the masked sequence modeling losses
            msm_losses = calculate_msm_losses(output_dict, sc_mask, mc_mask, calib_mask)
            # val_loss = msm_losses['loss']
            for loss_type in val_losses.keys():
                val_losses[loss_type].append(msm_losses[loss_type].item())

    return val_losses