import os
import yaml
import time
import torch
import argparse
import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots

# file dir
file_dir = Path(__file__).parent

def main(config_path: str) -> None:
    """
    Main function to run the spacetime model with specified config file.

    Loads configuration and prepares to call the model.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load prepare command to execute
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['name']}"

    # Generate a unique batch_id for this run
    # Define the name of the output folder for your batch
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'sub_datasets': []
    }

    # Initialize Visualization object
    viz = Visualization()

    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    # Prepare command
    cmd = \
    """\
    python\
    {spacetime_main_path}\
    --dataset {dataset}\
    --train_ids {train_ids}\
    --reconstruct_ids {reconstruct_ids}\
    --forecast_ids {forecast_ids}\
    --forecast_lengths {forecast_lengths}\
    --lag {lag}\
    --horizon {horizon}\
    --embedding_config {embedding_config}\
    --encoder_config {encoder_config}\
    --decoder_config {decoder_config}\
    --output_config {output_config}\
    --n_blocks {n_blocks}\
    --kernel_dim {kernel_dim}\
    --norm_order {norm_order}\
    --batch_size {batch_size}\
    --dropout {dropout}\
    --lr {lr}\
    --weight_decay {weight_decay}\
    --max_epochs {max_epochs}\
    --early_stopping_epochs {early_stopping_epochs}\
    --data_transform {data_transform}\
    --loss {loss}\
    --val_metric {val_metric}\
    --criterion_weights {criterion_weights}\
    --seed {seed}\
    {no_wandb}\
    """

    cmd_formatted = cmd.format(
        spacetime_main_path = file_dir / "main.py",
        dataset=config['dataset']['name'],
        train_ids = ' '.join([str(i) for i in config['dataset']['train_ids']]),
        reconstruct_ids = ' '.join([str(i) for i in config['dataset']['reconstruct_ids']]),
        forecast_ids = ' '.join([str(i) for i in config['dataset']['forecast_ids']]),
        forecast_lengths = ' '.join([str(i) for i in config['dataset']['forecast_lengths']]),
        lag=config['model']['lag'],
        horizon=config['model']['horizon'],
        embedding_config=config['model']['embedding_config'],
        encoder_config=config['model']['encoder_config'],
        decoder_config=config['model']['decoder_config'],
        output_config=config['model']['output_config'],
        n_blocks=config['model']['n_blocks'],
        kernel_dim=config['model']['kernel_dim'],
        norm_order=config['model']['norm_order'],
        batch_size=config['model']['batch_size'],
        dropout=config['model']['dropout'],
        lr=config['model']['lr'],
        weight_decay=config['model']['weight_decay'],
        max_epochs=config['model']['max_epochs'],
        early_stopping_epochs=config['model']['early_stopping_epochs'],
        data_transform=config['model']['data_transform'],
        loss=config['model']['loss'],
        val_metric=config['model']['val_metric'],
        criterion_weights=f"{config['model']['criterion_weights'][0]} {config['model']['criterion_weights'][1]} {config['model']['criterion_weights'][2]}",
        seed=config['model']['seed'],
        no_wandb="--no_wandb" if config['model']['no_wandb'] else "",
    )

    # Execute command
    print("---------------")
    print("Python running:")
    print(cmd_formatted)
    print("---------------")

    out = os.system(cmd_formatted)
    time.sleep(1) # to allow for ctrl+c

    print("---------------")
    print(f"Returned: {out}")
    print("---------------")

    # Load predictions
    pred_data = torch.load(file_dir / 'tmp_pred' / 'results.torch', weights_only=False)

    if 0: # Waiting on Philippe
        # Load test data
        _, test_data = load_dataset(dataset_name, pair_id)

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, test_data, pred_data)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        results_for_yaml = {key: float(value) for key, value in results.items()}
        batch_results['sub_datasets'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_run(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type)

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)