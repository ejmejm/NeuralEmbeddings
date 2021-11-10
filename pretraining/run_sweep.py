import argparse
import os

import numpy as np
import torch
import wandb

from config_handling import prepare_config, validate_config
from config_handling import to_wandb_sweep_format, from_wandb_format
from data_loading import prepare_dataloaders
from models import NeuroSignalEncoder
from training.msm import train_with_msm
from training.cpc import train_with_cpc
from prepare_data import prepare_data

### Create argparser for command line arguments ###

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/sweeps/test_sweep_config.yaml')
parser.add_argument('-p', '--preprocess', action='store_true',
                    help='Whether to re-preprocess the data each run.')
parser.add_argument('-np', '--no_preprocess', dest='preprocess', action='store_false',
                    help='Whether to re-preprocess the data each run.')
parser.add_argument('-t', '--train', dest='train', action='store_true',
                    help='Performs training when set to true.')
parser.add_argument('-nt', '--no_train', dest='train', action='store_false',
                    help='Performs training when set to true.')
parser.add_argument('-d', '--device', dest='device', default=None,
                    help='Device to use for training.')
parser.add_argument('-s', '--sweep_id', type=str, default=None)

parser.set_defaults(preprocess=True, train=True)

### Main ###

def run_sweep_iteration(preprocess=True, device=None):
    """
    Runs a single sweep iteration.
    """
    # Config items to be updated after the run
    config_updates = {}
    with wandb.init() as _:
        run_id = wandb.run.id
        # Get sweep iteration config
        config = from_wandb_format(wandb.config)

        try:
            validate_config(config)
        except ValueError as e:
            print('This sweep config is invalid for reason:', e)
            print('Moving onto next run...')
            # wandb.run.delete()
            return

        # Update the config device if necessary
        if device is not None:
            config['device'] = device
            config_updates['device'] = device

        # Re-preprocess data if required
        if preprocess:
            prepare_data(config)

        # Choose random seed if not provided
        if config['seed'] is None:
            config['seed'] = np.random.randint(0, 2**31)
            config_updates['seed'] = config['seed']

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
                
    # Update the config with any changed values
    if len(config_updates) > 0:
        api = wandb.Api()
        run = api.run('{}/{}/{}'.format(
            config['wandb_entity'], config['wandb_project'], run_id))
        run.config.update(config_updates)
        run.update()

if __name__ == '__main__':
    args = parser.parse_args()

    # Get the path of the config file in relation to this main.py file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, args.config)

    # Load the config file
    config = prepare_config(config_path, validate=False)
    # Stops invalid runs from ending the sweep
    os.environ['WANDB_AGENT_DISABLE_FLAPPING'] = 'true'

    if args.sweep_id is None:
        # Init wandb for sweep
        wandb_config = to_wandb_sweep_format(config)
        sweep_id = wandb.sweep(wandb_config, entity=config['wandb_entity'],
            project=config['wandb_project'])
    else:
        sweep_id = args.sweep_id
    print(f'Sweep ID: {sweep_id}')

    if args.train:
        # Run a sweep agent
        # TODO: Need to get right config for the sweep iteration
        count = config['count'] if 'count' in config else None
        run_func = lambda: run_sweep_iteration(
            args.preprocess, args.device)
        wandb.agent(sweep_id, run_func, config['wandb_entity'],
            config['wandb_project'], count=count)