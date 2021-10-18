import copy
import os
from pprint import pprint
import warnings

import yaml

def load_config(path: str) -> dict:
    """
    Load a config from a yaml file.

    Args:
        path: The path to the config file.
    
    Returns:
        The loaded config dictionary.
    """
    with open(path, 'r') as f:
        return yaml.load(f)

def to_wandb_sweep_format(config: dict) -> dict:
    """
    Converts the config to the format that wandb expects for sweeps.

    Args:
        config: The config to convert.

    Returns:
        The converted config.
    """
    if 'sweep' not in config:
        raise ValueError('The config does not contain sweep params.')

    wandb_config = {}
    parse_config = copy.copy(config)

    wandb_config.update(parse_config['sweep'])
    del parse_config['sweep']
    params_config = to_wandb_format(parse_config)
    wandb_config['parameters'] = params_config

    return wandb_config

def to_wandb_format(config: dict) -> dict:
    """
    Converts the config to the format that wandb expects.

    Args:
        config: The config to convert.

    Returns:
        The converted config.
    """
    wandb_config = {}
    for key, value in config.items():
        # Handle non-string keys
        if not isinstance(key, str):
            warnings.warn(f'The key {key} is not a string, auto-converting could cause issues.')
            key = str(key)

        # Recursive processing of the config
        if isinstance(value, dict):
            if not ('value' in value \
                    or 'values' in value \
                    or 'distribution' in value):
                sub_dict = to_wandb_format(value)
                sub_dict = {key + '.' + sub_key: sub_value for \
                    sub_key, sub_value in sub_dict.items()}
                wandb_config.update(sub_dict)
            else:
                wandb_config[key] = value
        elif isinstance(value, list):
            wandb_config[key] = {'values': value}
        else:
            wandb_config[key] = {'value': value}
    return wandb_config

def from_wandb_format(config: dict) -> dict:
    """
    Create a usable config from the wandb config format.

    Args:
        config: The config to convert.

    Returns:
        The converted config.
    """
    wandb_config = {}
    for key, value in config.items():
        if '.' in key:
            sub_keys = key.split('.')
            curr_dict = wandb_config
            for sub_key in sub_keys[:-1]:
                if sub_key not in curr_dict:
                    curr_dict[sub_key] = {}
                curr_dict = curr_dict[sub_key]
            curr_dict[sub_keys[-1]] = value
        else:
            wandb_config[key] = value
    return wandb_config

if __name__ == '__main__':
    # Get absolute path of the this directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(curr_dir, 'configs/sweeps/test_sweep_config.yaml'))
    wandb_config = to_wandb_format(config)
    wandb_sweep_config = to_wandb_sweep_format(config)
    pprint(wandb_config)
    pprint(wandb_sweep_config)