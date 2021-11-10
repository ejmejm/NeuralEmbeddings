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
        return yaml.load(f, Loader=yaml.FullLoader)

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
            if isinstance(value, dict) and 'value' in value:
                curr_dict[sub_keys[-1]] = value['value']
            else:
                curr_dict[sub_keys[-1]] = value
        elif isinstance(value, dict) and 'value' in value:
            wandb_config[key] = value['value']
        else:
            wandb_config[key] = value
    return wandb_config

def merge_configs(config: dict, default_config: dict,
                  wandb_format: bool = False) -> dict:
    """
    Merges two configs, where the first takes precedence.

    Args:
        config: The config to merge over the default config.
        default_config: The default config.
        wandb_format: Whether the configs are in wandb format or not.
    """
    new_config = copy.deepcopy(default_config)
    if wandb_format:
        new_config.update(config)
    else:
        config = to_wandb_format(config)
        new_config = to_wandb_format(new_config)
        new_config.update(config)
        new_config = from_wandb_format(new_config)
    return new_config

def validate_config(config: dict):
    """
    Validates whether the config is formatted correctly or not.
    Does not check everything, but specifically cases where two
    values are not compatible.

    Args:
        config: The config in nested dict format.
    
    Throws:
        ValueError: If the config is not formatted correctly.
    """
    if config['train_method'].lower() == 'msm' and \
       config['msm_params'] is None:
        raise ValueError('MSM params must be specified for MSM training.')
    if config['train_method'].lower() == 'cpc' and \
       config['cpc_params'] is None:
        raise ValueError('CPC params must be specified for CPC training.')

    if config['train_method'].lower() == 'msm' and \
       config['msm_params']['min_mask_len'] > \
       config['msm_params']['max_mask_len']:
        raise ValueError('The minimum mask length must be less than or ' + \
            'equal to the maximum mask length.')

    if config['use_standardization'] and \
       config['use_normalization']:
        raise ValueError('Standardization and normalization cannot be ' + \
            'used at the same time.')

    model_config = config['model']
    if not model_config['single_channel_module']['enabled'] and \
       not model_config['mixed_channel_module']['enabled']:
        raise ValueError('At least one primary encoder module must be enabled.')

    if not model_config['single_channel_module']['enabled'] and \
       model_config['calibration_module']['enabled']:
        raise ValueError('Calibration module cannot be enabled without ' + \
            'a single channel module.')

def prepare_config(config_path: str, validate=True):
    # Get the path of the config file in relation to this main.py file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, config_path)

    # Load the config file
    config = load_config(config_path)
    config = merge_configs(config, DEFAULT_CONFIG)
    if validate:
        validate_config(config)

    return config


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__),
    'configs/default_config.yaml')
DEFAULT_CONFIG = load_config(DEFAULT_CONFIG_PATH)


if __name__ == '__main__':
    # Get absolute path of the this directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(curr_dir, 'configs/sweeps/test_sweep_config.yaml'))
    wandb_config = to_wandb_format(config)
    wandb_sweep_config = to_wandb_sweep_format(config)
    pprint(wandb_config)
    pprint(wandb_sweep_config)