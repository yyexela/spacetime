import os
import sys
import yaml
import argparse
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization

file_dir = Path(__file__).parent
results_file = file_dir / 'results.yaml'

# Update python PATH so that we can load run.py from CTF_NaiveBaselines directly
sys.path.insert(0, str(file_dir))

from run_opt import main as run_opt_main

def sum_results(results):
    """
    Sums all metric values from a results dictionary containing evaluation metrics.
    
    Iterates through all pairs in the results dictionary and sums all metric values
    found in each pair's 'metrics' dictionary. This is used to aggregate
    evaluation metrics from a batch_results.yaml file.

    Args:
        results (dict): A dictionary containing evaluation results.
    
    Returns:
        float: The sum of all metric values across all pairs in the results dictionary.
    """
    total = 0
    for pair_dict in results['pairs']:
        metric_dict = pair_dict['metrics']
        for metric in metric_dict.keys():
            total += metric_dict[metric]
    return total

def create_search_space(tuning_config):
    """
    Create a Ray Tune search space dictionary from the tuning config flie.

    Args:
        tuning_config (dict):
            Dictionary containing the parameter specification with the following keys:
            - 'type': str, either 'float' or 'int' indicating the parameter type
            - 'lower_bound': float/int, the minimum value for the parameter
            - 'upper_bound': float/int, the maximum value for the parameter
            - 'log': bool, whether to sample in log space

    Returns:
        dict: Ray Tune expected search_space dictionary

    Raises:
        Exception:
            If any of the required keys ('type', 'lower_bound', 'upper_bound', 'log')
            are missing from tuning_config for a parameter.
            If the parameter type is neither 'float' nor 'int'.
    """
    search_space = {}
    for name in tuning_config['hyperparameters'].keys():
        param_dict = tuning_config['hyperparameters'][name]
        if 'type' not in param_dict:
            raise Exception(f"\'type\' not in {param_dict} keys")

        if param_dict['type'] == "uniform":
            search_space[name] = tune.uniform(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "quniform":
            search_space[name] = tune.quniform(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "loguniform":
            search_space[name] = tune.loguniform(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "qloguniform":
            search_space[name] = tune.qloguniform(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "randn":
            search_space[name] = tune.randn(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "qrandn":
            search_space[name] = tune.qrandn(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "randint":
            search_space[name] = tune.randint(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "qrandint":
            search_space[name] = tune.qrandint(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "lograndint":
            search_space[name] = tune.lograndint(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "qlograndint":
            search_space[name] = tune.qlograndint(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "choice":
            search_space[name] = tune.choice(param_dict['choices'])
        elif param_dict['type'] == "grid":
            search_space[name] = tune.grid(param_dict['grid'])
        else:
            raise Exception(f"Parameter type {param_dict['type']} not supported.")

    return search_space

def generate_config(config, template, name):
    """
    Generates a configuration file with suggested hyperparameter values.

    This function suggests a value for the constant parameter using Raytun's config,
    updates the configuration template with this value, and saves the resulting
    configuration to a YAML file.

    Args:
        config (dict): Dictionary containing selected hyperparameters.
        template (dict): Configuration template dictionary that will be populated with
            the suggested values.
        name (str): Name to use for the output configuration file (without extension).

    Returns:
        dict: updated config file

    Side Effects:
        - Writes a new YAML configuration file to ./config/{name}.yaml
        - Modifies the input template dictionary by adding the suggested constant value
    """
    # Fill out dictionary
    template['model']['lag'] = config['lag']
    template['model']['horizon'] = config['horizon']
    template['model']['n_kernels'] = config['n_kernels']
    template['model']['n_blocks'] = config['n_blocks']
    template['model']['weight_decay'] = config['weight_decay']
    template['model']['lr'] = config['lr']
    template['model']['dropout'] = config['dropout']
    # Save config
    config_path = file_dir / 'ctf_config' / f'{name}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(template, f)
    return config_path

def main(config_path: str, save_config: bool = True) -> None:
    """
    Main function to generate configuration files and run the naive baseline
    model on specified sub-datasets for hyperparameter optimization.

    Loads configuration, generates specific config files, runs model on training
    and validation set, and performs hyperparameter optimization.

    Args:
        config_path (str): Path to the configuration file.
        save_config (str): Save the final configuration file. (only False in unit tests)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        hp_config = yaml.safe_load(f)

    # Blank dictionary for runnable yaml file
    yaml_dict = {
        "dataset":{
            "name": "PDE_KS",
            "pair_id": hp_config['dataset']['pair_id']
        },
        "model":{
            "name": "spacetime",
            "lag": 10,
            "horizon": 10,
            "embedding_config": "embedding/repeat",
            "encoder_config": "encoder/default_no_skip",
            "decoder_config": "decoder/default",
            "output_config": "output/default",
            "n_blocks": 1,
            "mlp_n_layers": 1,
            "kernel_dim": 64,
            "norm_order": 1,
            "batch_size": 5,
            "dropout": 0.25,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 2,
            "early_stopping_epochs": 2,
            "data_transform": "mean",
            "loss": "informer_rmse",
            "val_metric": "informer_rmse",
            "criterion_weights": [1, 1, 1],
            "seed": 0,
            "no_wandb": True
        }
    }

    # Generate parameter dictionary for Ray Tune
    param_dict = create_search_space(hp_config)

    # Define objective for Ray Tune
    def objective(config):
        # Create config file
        config_path = generate_config(config, yaml_dict, 'hp_config')
        # Run model
        print("running")
        print(f"python {file_dir / 'run.py'} {config_path}")
        ret = os.system(f"python {file_dir / 'run_opt.py'} {config_path}")
        if ret != 0:
            raise Exception(f"Output: {ret}")
        # Extract results
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)
        score = sum_results(results)
        # Return score
        return {"score": score}

    # Create Ray Tune object
    tuner = tune.Tuner(objective,
                        param_space=param_dict,
                        tune_config=tune.TuneConfig(
                            #search_alg=OptunaSearch(), # Throws errors (seg faults) but still runs
                            num_samples=hp_config['model']['n_trials'],
                            max_concurrent_trials=1, # Don't parallelize, for debugging purposes
                            metric="score",
                            mode="max",
                        ),
                       )
    
    # Run optimization
    results = tuner.fit()

    # Remove last results file and hp_config.yaml (no loose files)
    results_file.unlink(missing_ok=True)
    (file_dir / 'config' / 'hp_config.yaml').unlink(missing_ok=True)

    # Obtain best hyperparameter value
    best_score = results.get_best_result(metric="score", mode="max").metrics['score']
    best_params = results.get_best_result(metric="score", mode="max")
    print(f"Best score: {best_score} (params: {best_params})")

    # Save final configuration yaml from hyperparameter optimization
    if not save_config: # Only False when unit testing
        print("Not saving final config file.")
    else:
        config_path = file_dir / 'config' / f'config_{hp_config["dataset"]["name"]}_constant_batch_{hp_config["dataset"]["pair_id"]}.yaml'
        yaml_dict['model']['constant_value'] = best_params 
        print("Final config file saved to:", config_path)
        with open(config_path, 'w') as f:
            yaml.dump(yaml_dict, f)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the hyperparameter configuration file.")
    parser.add_argument('save_config', action='store_true', help="Save the final hyperparameter configuration file. Only used when unit testing.")
    args = parser.parse_args()
    main(args.config, args.save_config)