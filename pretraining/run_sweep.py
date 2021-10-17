import argparse
import os

import numpy as np
import torch
import wandb

from config_handling import load_config, to_wandb_format
from config_handling import to_wandb_sweep_format, from_wandb_format
from data_loading import prepare_dataloaders
from models import NeuroSignalEncoder
from training.msm import train_with_msm

### Create argparser for command line arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/sweeps/test_sweep_config.yaml')

### Main ###

def run_sweep_iteration():
    """
    Runs a single sweep iteration.
    """
    with wandb.init() as _:
        # Get sweep iteration config
        config = from_wandb_format(wandb.config)

        # Set seeds
        if config['seed'] is not None:
            np.random.seed(config['seed'])
            torch.manual_seed(config['seed'])

        # Prepare the data
        dataloaders = prepare_dataloaders(config)
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']

        # Create the model
        model = NeuroSignalEncoder(model_config)
        model = model.to(config['device'])
        wandb.watch(model, log_freq=100)

        # Train the model
        if config['train_method'].lower() == 'msm':
            train_with_msm(model, config, train_loader, val_loader)
        else:
            raise ValueError('Unknown train method: "{}"'.format(config['train_method']))

if __name__ == '__main__':
    args = parser.parse_args()

    # Get the path of the config file in relation to this main.py file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, args.config)

    # Load the config file
    config = load_config(config_path)
    model_config = config['model']

    # Init wandb for sweep
    sweep_config = to_wandb_sweep_format(config)
    sweep_id = wandb.sweep(sweep_config, project='neural-embeddings')

    # Run a sweep agent
    # TODO: Need to get right config for the sweep iteration
    count = config['count'] if 'count' in config else None
    wandb.agent(sweep_id, run_sweep_iteration, count=count)