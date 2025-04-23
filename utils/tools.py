import argparse
import os
import sys
import time
import yaml
import numpy as np

_DEFAULT_CONFIG_PARSER = argparse.ArgumentParser(description="Experiment Config", add_help=False)
_DEFAULT_CONFIG_PARSER.add_argument("--cfg", required=True, type=str, metavar="FILE", help="YAML config file specifying default arguments")


def parse_args_and_yaml(given_parser=None):
    """
    use yaml file and argparse to specify the experiment config.
    some code from https://www.cnblogs.com/zxyfrank/p/15414605.html
    """
    if given_parser is None:
        given_parser = _DEFAULT_CONFIG_PARSER
    given_configs, remaining = given_parser.parse_known_args()
    if given_configs.cfg:
        with open(given_configs.cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            given_parser.set_defaults(**cfg)

    args = given_parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later.
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text


def set_seed(seed: int):
    """Set random seed, remeber to add worker_init_fn=_init_fn when creating the dataloader"""
    import torch
    import numpy as np
    import random
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('Set random seed as: {}'.format(seed))
    
def make_exp_folder(exp_name):
    """
    Make a folder for the experiment, and return the path,
    if the folder already exists, ask the user to confirm whether to overwrite it.
    """
    exp_folder = os.path.join('checkpoints', exp_name)
    if os.path.exists(exp_folder):
        print('The folder {} already exists, do you want to overwrite it?'.format(exp_folder))
        while True:
            choice = input('Please input y/n: ')
            if choice == 'y':
                break
            elif choice == 'n':
                print('Please change the exp_name in the config file')
                sys.exit(0)
            else:
                print('Invalid input, please input y/n')
    else:
        os.makedirs(exp_folder)
    return exp_folder

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        # overwrite the log file if it exists
        self.log = open(filename, "w")
    
    def write(self, message):
        # add timestamp before the message
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass