import argparse
import os
from typing import Optional

import numpy as np
import torch
import wandb

from config_handling import prepare_config, to_wandb_format
from data_loading import prepare_dataloaders, prepare_downsteam_dataloaders
from models import NeuroSignalEncoder, NeuroDecoder
from downstream import train_downstream
from training.msm import train_with_msm
from training.cpc import train_with_cpc

### Create argparser for command line arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/test_config.yaml',
                    help='Path to config file')
parser.add_argument('-g', '--group', type=str, default=None,
                    help='Add the run to the given group for easier tracking.')
parser.add_argument('-p', '--pretrain', dest='pretrain', action='store_true',
                    help='Performs pretraining when set to true.')
parser.add_argument('-np', '--no_pretrain', dest='pretrain', action='store_false',
                    help='Performs pretraining when set to true.')
parser.add_argument('-d', '--downstream', dest='downstream', action='store_true',
                    help='Performs downstream training when set to true.')
parser.add_argument('-nd', '--no_downstream', dest='downstream', action='store_false',
                    help='Performs downstream training when set to true.')
parser.set_defaults(gen_data=False, pretrain=True, downstream=True)


### Main Functions ###


def pretrain(config: dict, group: Optional[str] = None):
    """
    Pretrain the model.
    """
    print('Preparing for pretraining...')

    # Choose random seed if not provided
    if config['seed'] is None:
        config['seed'] = np.random.randint(0, 2**31)
    config['stage'] = 'pretrain'

    # Init wandb for logging
    wandb_config = to_wandb_format(config)
    wandb.init(entity=config['wandb_entity'], group=group,
        project=config['wandb_project'], config=wandb_config)

    # Set and log seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    print('seed:', config['seed'])

    # Prepare the data
    dataloaders = prepare_dataloaders(config)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Create the model
    model_config = config['model']
    model = NeuroSignalEncoder(model_config)
    # Add an LSTM head if CPC is being used
    if config['train_method'].lower() == 'cpc':
        model.add_lstm_head(model_config['lstm_embedding_dim'])
    model = model.to(config['device'])
    wandb.watch(model, log_freq=100)

    # Train the model
    if config['train_method'].lower() == 'msm':
        train_with_msm(model, config, train_loader, val_loader)
    elif config['train_method'].lower() == 'cpc':
        train_with_cpc(model, config, train_loader, val_loader)
    else:
        raise ValueError('Train method "{}" not recognized.'\
            .format(config['train_method']))

    # Save the model
    if model_config['save_path'] is not None:
        base_dir = os.path.dirname(__file__)
        save_path = os.path.join(base_dir, model_config['save_path'])
        torch.save(model.state_dict(), save_path)

    wandb.finish()


def run_downstream(config: dict, group: Optional[str] = None):
    """
    Train the model on a downstream task.
    """
    print('Preparing for downstream training...')

    # Choose random seed if not provided
    if config['seed'] is None:
        config['seed'] = np.random.randint(0, 2**31)
    config['stage'] = 'downstream'

    # Init wandb for logging
    wandb_config = to_wandb_format(config)
    wandb.init(entity=config['wandb_entity'], group=group,
        project=config['wandb_project'], config=wandb_config)

    # Set and log seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    print('seed:', config['seed'])

    # Prepare the data
    dataloaders = prepare_downsteam_dataloaders(config, 1)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # Create the model
    model = NeuroDecoder(config)
    model = model.to(config['device'])
    wandb.watch(model, log_freq=100)

    # Start the training and testing
    train_downstream(model, config, train_loader, val_loader, test_loader)

    # Save the model
    save_path = config['downstream']['model_path']
    if save_path is not None:
        base_dir = os.path.dirname(__file__)
        save_path = os.path.join(base_dir, save_path)
        torch.save(model.state_dict(), save_path)

    wandb.finish()

### Main ###

if __name__ == '__main__':
    args = parser.parse_args()
    # Load the config
    config = prepare_config(args.config)

    # Train the model
    if args.pretrain:
        pretrain(config, args.group)
    # Train the downstream model
    if args.downstream:
        run_downstream(config, args.group)