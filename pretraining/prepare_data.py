import argparse
import os

from config_handling import prepare_config
from data_loading import load_raw_data, preprocess_and_save_data
from data_loading import preprocess_and_save_epoched_data, fit_partial_preprocessors


parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config', type=str, default='configs/test_config.yaml',
                    help='Path to config file')
parser.add_argument('-f', '--fit', dest='fit', action='store_true',
                    help='Fits the preprocessors when true.')
parser.add_argument('-nf', '--no_fit', dest='fit', action='store_false',
                    help='Fits the preprocessors when true.')
parser.add_argument('-p', '--pretrain', dest='pretrain', action='store_true',
                    help='Prepares pretraining data when true.')
parser.add_argument('-np', '--no_pretrain', dest='pretrain', action='store_false',
                    help='Prepares pretraining data when true.')
parser.add_argument('-d', '--downstream', dest='downstream', action='store_true',
                    help='Prepares downstream data when true.')
parser.add_argument('-nd', '--no_downstream', dest='downstream', action='store_false',
                    help='Prepares downstream data when true.')
parser.set_defaults(fit=False, pretrain=False, downstream=False)

def prepare_data(config, fit=True, transform=True):
    # Load and process the train data
    base_dir = os.path.dirname(__file__)
    file_type = config['data_file_type']
    train_dir = os.path.join(base_dir, config['train_dir'])

    print('Loading train data...')
    train_data = load_raw_data(train_dir, file_type)
    if transform:
        metadata_fn = config['train_val_info']
        output_dir = os.path.join(base_dir, config['train_val_preprocessed'])
        preprocess_and_save_data(train_data, config,
            output_dir, metadata_fn, fit_preprocessors=fit)
    elif fit:
        flat_train_data = [raw_data for database in train_data for raw_data in database]
        fit_partial_preprocessors(flat_train_data, config)

    if transform:
        # Load and process the test data
        metadata_fn = config['test_info']
        test_dir = os.path.join(base_dir, config['test_dir'])
        output_dir = os.path.join(base_dir, config['test_preprocessed'])

        print('Loading test data...')
        test_data = load_raw_data(test_dir, file_type)
        preprocess_and_save_data(test_data, config,
            output_dir, metadata_fn, fit_preprocessors=False)


def prepare_downstream_data(config):
    # Load and process the train data
    base_dir = os.path.dirname(__file__)
    file_type = config['data_file_type']
    label_fn = config['label_info']
    downstream_dir = os.path.join(base_dir, config['downstream_dir'])
    output_dir = os.path.join(base_dir, config['downstream_preprocessed'])

    # print('Fixing train split issues...')
    # correct_all_data(train_dir, file_type)

    print('Loading train data...')
    downstream_data = load_raw_data(downstream_dir, file_type)

    # Need to pass in a list of events 
    # (for colors dataset we have are looking for events with id = 1, 2, ..., 10)
    # TODO: This shouldn't be hardcoded
    preprocess_and_save_epoched_data(downstream_data, config, output_dir, label_fn, include_events_list=list(range(1, 9)))


if __name__ == '__main__':
    args = parser.parse_args()

    # Load the config file
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, args.config)
    config = prepare_config(config_path)

    # Preprocess the pretraining data
    if args.fit or args.pretrain:
        print('Starting pretraining data preprocessing...')
        prepare_data(config, args.fit, args.pretrain)

    # Preprocess the downstream data
    if args.downstream:
        print('Starting downstream data preprocessing...')
        prepare_downstream_data(config)