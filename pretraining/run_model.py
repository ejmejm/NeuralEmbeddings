import argparse
import os

import numpy as np
import torch
import wandb

from config_handling import load_config, to_wandb_format, validate_config
from config_handling import merge_configs, DEFAULT_CONFIG
from data_loading import prepare_dataloaders
from models import NeuroSignalEncoder
from training.msm import train_with_msm
from training.cpc import train_with_cpc

### Create argparser for command line arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/test_config.yaml',
                    help='Path to config file')
parser.add_argument('-t', '--train', dest='train', action='store_true',
                    help='Performs training when set to true.')
parser.add_argument('-n', '--no_train', dest='train', action='store_false',
                    help='Performs training when set to true.')
parser.add_argument('-e', '--test', action='store_true',
                    help='Performs testing when set to true.')
parser.set_defaults(gen_data=False, train=True, test=False)


### Main Functions ###


def train(config: dict):
    """
    Train the model.
    """
    # Init wandb for logging
    wandb_config = to_wandb_format(config)
    wandb.init(entity=config['wandb_entity'],
        project=config['wandb_project'], config=wandb_config)

    # Choose random seed if not provided
    if config['seed'] is None:
        config['seed'] = np.random.randint(0, 2**31)
        wandb.config.update({'seed': config['seed']}, True)

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

    # Save the model
    base_dir = os.path.dirname(__file__)
    save_path = os.path.join(base_dir, model_config['save_path'])
    torch.save(model.state_dict(), save_path)

    wandb.finish()

def test(config: dict):
    """
    Test the model.
    """
    raise NotImplementedError()

def prepare_config(config_path: str):
    # Get the path of the config file in relation to this main.py file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, config_path)

    # Load the config file
    config = load_config(config_path)
    config = merge_configs(config, DEFAULT_CONFIG)
    validate_config(config)

    return config

### Main ###

if __name__ == '__main__':
    args = parser.parse_args()
    # Load the config
    config = prepare_config(args.config)

    # Train the model
    if args.train:
        train(config)
    # Test the model
    if args.test:
        test(config)