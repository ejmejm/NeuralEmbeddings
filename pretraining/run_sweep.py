import argparse
import os

import numpy as np
import torch
import wandb

from config_handling import load_config, validate_config
from config_handling import to_wandb_sweep_format, from_wandb_format
from data_loading import prepare_dataloaders
from models import NeuroSignalEncoder
from training.msm import train_with_msm
from training.cpc import train_with_cpc

### Create argparser for command line arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/sweeps/test_sweep_config.yaml')
parser.add_argument('-s', '--sweep_id', type=str, default=None)

### Main ###

def run_sweep_iteration():
    """
    Runs a single sweep iteration.
    """
    with wandb.init() as _:
        # Get sweep iteration config
        config = from_wandb_format(wandb.config)
        validate_config(config)
            
        model_config = config['model_config']

        # Choose random seed if not provided
        if config['seed'] is None:
            config['seed'] = np.random.randint(0, 2**32)

        # Set and log seed
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        wandb.log({'seed': config['seed']})
        print('seed:', config['seed'])

        # Prepare the data
        dataloaders = prepare_dataloaders(config)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']

        # Create the model
        model = NeuroSignalEncoder(model_config)
        if config['train_method'].lower() == 'cpc':
            cpc_config = config['cpc_params']
            model.add_lstm_head(cpc_config['embedding_dim'])
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

if __name__ == '__main__':
    args = parser.parse_args()

    # Get the path of the config file in relation to this main.py file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, args.config)

    # Load the config file
    config = load_config(config_path)

    if args.sweep_id is None:
        # Init wandb for sweep
        wandb_config = to_wandb_sweep_format(config)
        sweep_id = wandb.sweep(wandb_config, project='neural-embeddings')
    else:
        sweep_id = args.sweep_id

    # Run a sweep agent
    # TODO: Need to get right config for the sweep iteration
    count = config['count'] if 'count' in config else None
    wandb.agent(sweep_id, run_sweep_iteration, count=count)