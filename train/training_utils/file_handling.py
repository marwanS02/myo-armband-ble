from datetime import datetime
import json
import torch.nn as nn
import torch.optim as optim
import os

def create_folders(model):
    model_class_name = model.__class__.__name__
    model_class_dir = f"models/{model_class_name}"
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_session_dir = f"{model_class_dir}/{now}"
    logs_dir = f"training_logs/training_details"
    plots_dir = f"training_logs/plots"
    
    
    os.makedirs('training_logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(model_class_dir, exist_ok=True)
    os.makedirs(train_session_dir, exist_ok=True)
    os.makedirs(logs_dir,exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return model_class_dir, train_session_dir, logs_dir, plots_dir

def save_losses(train_losses, val_losses, filename):
    losses = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(filename, 'w') as f:
        json.dump(losses, f)
    print(f"Losses saved to {filename}")


import json
import torch.nn as nn
import torch.optim as optim

def convert_to_serializable(hparam_dict):
    """Convert objects in the hyperparameter dictionary to serializable formats."""
    serializable_dict = {}
    for key, value in hparam_dict.items():
        if isinstance(value, type):  # Handle classes like optim.Adam
            serializable_dict[key] = value.__name__
        elif callable(value):  # Handle activation functions
            serializable_dict[key] = value.__class__.__name__
        elif isinstance(value, dict):  # Recursively handle nested dictionaries
            serializable_dict[key] = convert_to_serializable(value)
        else:
            serializable_dict[key] = value
    return serializable_dict

def save_hyperparameters(hparam_dict, filename):
    """Save the hyperparameters to a JSON file."""
    serializable_hparams = convert_to_serializable(hparam_dict)
    flattened_dict = flatten_dict(serializable_hparams)
    with open(filename, 'w') as f:
        json.dump(flattened_dict, f, indent=4)
    print(f"Hyperparameters saved to {filename}")


def reconstruct_hparams(hparam_dict):
    """Reconstruct the original hyperparameter objects from the loaded dictionary."""
    reconstructed_dict = {}
    for key, value in hparam_dict.items():
        if isinstance(value, str):
            # Handle optimizers and criteria
            if hasattr(optim, value):
                reconstructed_dict[key] = getattr(optim, value)
            elif hasattr(nn, value):
                reconstructed_dict[key] = getattr(nn, value)
            else:
                reconstructed_dict[key] = value
        elif isinstance(value, dict):  # Recursively handle nested dictionaries
            reconstructed_dict[key] = reconstruct_hparams(value)
        else:
            reconstructed_dict[key] = value
    return reconstructed_dict

def load_hyperparameters(filename):
    """Load the hyperparameters from a JSON file and reconstruct the original objects."""
    with open(filename, 'r') as f:
        loaded_hparams = json.load(f)
    return reconstruct_hparams(loaded_hparams)

def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key for nested items (used in recursion).
        sep (str): The separator between keys (e.g., '_' or '.').

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)