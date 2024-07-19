import os
import warnings
import argparse

def positive_float(value):
    f_value = float(value)
    if f_value < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float value")
    return f_value

def get_args():
    parser = argparse.ArgumentParser(description = 'Run elderly fall detection script.')
    parser.add_argument('--max_acc_drop', type = positive_float, default = None, 
                        help = 'Maximum accuracy drop (default: None)')
    parser.add_argument('--device', type = str, choices = ['cpu', 'cuda'], default = 'cpu', 
                        help = 'Device to run the model on (default: cpu)')
    return parser.parse_args()
    
def initialize_environment(file_name):
    # Suppress specific warnings
    warnings.filterwarnings("ignore", 
        message="Named tensors and all their associated APIs are an experimental feature and subject to change.")

    # Get the file name without extension
    file_name = os.path.basename(file_name)
    name = file_name.split(".")[0]

    return name
