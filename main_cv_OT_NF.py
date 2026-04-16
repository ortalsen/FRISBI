import os
import yaml
import wandb
from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from argparse import ArgumentParser
from data import *
from pipelines.NFOT import *

# Load configuration from YAML file
def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

# Evaluate fold-trained models on the test set
def evaluate_test_set(models, test_data_loader):

    accumulated_metrics = {}
    metrics_per_fold = []

    xs_test_loader,xo_calib_test_loader, train_xo_loader, val_xo_loader = test_data_loader

    for fold, models in enumerate(models):
        metrics_rope = models[0].test_pipeline((xs_test_loader,xo_calib_test_loader, train_xo_loader, val_xo_loader))
        metrics_nfrope = models[1].test_pipeline(xo_calib_test_loader)
        metrics_wass = models[2].test_pipeline(xo_calib_test_loader)

        metrics_rope = {f"OT_align_{key}": value for key, value in metrics_rope.items()}
        metrics_nfrope = {f"NF_align_{key}": value for key, value in metrics_nfrope.items()}
        metrics_wass = {f"WassOT_NF_align_{key}": value for key, value in metrics_wass.items()}

        metrics = {**metrics_rope, **metrics_nfrope, **metrics_wass}

        for key, value in metrics.items():
            accumulated_metrics.setdefault(key, []).append(value)

    # Compute average and standard deviation for each metric
    for key, values in accumulated_metrics.items():
        mean_val = sum(values) / len(values)
        std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
        print(f"{key}: Mean={mean_val:.4f}, Std={std_val:.4f}")
        wandb.log({f"mean_test_{key}": mean_val, f"std_test_{key}": std_val})

# Function to run a single baseline sweep
def run_sweep(config_path, num_samples, gpu_id, project_name, label_noise, debug):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Running experiment with #{num_samples} samples and {label_noise} noise on device: {device}")

    # Load and adjust config
    global_config = load_config(config_path)
    global_config['real_data']['calib_train']['num_samples'] = num_samples

    # Prepare dataset
    dataset = {}
    real_data = globals()[global_config["real_data"]["name"]]
    sim_data = globals()[global_config["sim_data"]["name"]]
    dataset['train'] = real_data(**global_config['real_data']['params'], **global_config['real_data']['calib_train'], is_label=True, is_noisy_label=label_noise)
    dataset['test']  = real_data(**global_config['real_data']['params'], **global_config['real_data']['calib_test'],  is_label=True, is_noisy_label=label_noise)
    xo_train_data = real_data(**global_config['real_data']['params'], **global_config['real_data']['train'], is_label=True)
    xo_val_data = real_data(**global_config['real_data']['params'], **global_config['real_data']['val'], is_label=True)
    data_xs_train = sim_data(**global_config['sim_data']['params'], **global_config['sim_data']['train'])
    data_xs_val = sim_data(**global_config['sim_data']['params'], **global_config['sim_data']['val'])
    data_xs_test = sim_data(**global_config['sim_data']['params'], **global_config['sim_data']['test'])


    # Build sweep config
    sweep_config = {
        'method': 'random',
        'metric': {'name': f"avg_val_loss_Wass_OT_align_finetune", 'goal': 'minimize'},
        'parameters': {k: {'values': v} for k, v in global_config['sweep_hyperparams'].items()}
    }

    if debug:
        global_config['training']['epochs'] = 1

    # Internal CV training function
    def sweep_cv(config=None):
        with wandb.init(config=config, group='Wass_OT_finetune') as run:
            config = wandb.config
            kf = KFold(n_splits=global_config['training']['num_folds'], shuffle=True, random_state=42)
            fold_models = []
            losses = []

            # update training params
            global_config['training'].update(config)
            num_workers = global_config['training']['num_workers']

            # fixed dataset loaders
            xs_train_loader = DataLoader(data_xs_train, batch_size=global_config['training']['batch_size'], shuffle=True, num_workers=num_workers)
            xs_val_loader = DataLoader(data_xs_val, batch_size=global_config['training']['batch_size'], shuffle=False, num_workers=num_workers)
            xs_test_loader = DataLoader(data_xs_test, batch_size=global_config['training']['batch_size'], shuffle=False, num_workers=num_workers)

            # create dataloaders for unpaired xo 
            train_xo_loader = DataLoader(xo_train_data, batch_size=global_config['training']['batch_size'], shuffle=True, num_workers=num_workers)
            val_xo_loader = DataLoader(xo_val_data, batch_size=global_config['training']['batch_size'], shuffle=False, num_workers=num_workers)

            train_xo_loader_small_batch = DataLoader(xo_train_data, batch_size=32, shuffle=True, num_workers=num_workers)
            val_xo_loader_small_batch = DataLoader(xo_val_data, batch_size=32, shuffle=False, num_workers=num_workers)

            
            for fold, (train_idx, val_idx) in enumerate(kf.split(dataset['train'])):
                print(f"CV experiment: Fold {fold+1}")
                
                # data loaders
                train_ds = Subset(dataset['train'], train_idx)
                val_ds   = Subset(dataset['train'], val_idx)
                xo_calib_train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
                xo_calib_val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=num_workers)

                train_loaders = (xs_train_loader, xs_val_loader, train_xo_loader_small_batch, val_xo_loader_small_batch, xo_calib_train_loader, xo_calib_val_loader)
                # train
                rope_pipeline = RoPE(global_config, run, device = device, fold=fold)
                rope_pipeline.train_pipeline(train_loaders)
                pipeline = NFRoPE(global_config, run, device = device, fold=fold)
                pipeline.train_pipeline(train_loaders + (xs_test_loader,))
                wass_pipeline = WassRoPE(global_config, run, device = device, fold=fold)
                val_loss = wass_pipeline.train_pipeline(((train_xo_loader, train_xo_loader_small_batch), (val_xo_loader, val_xo_loader_small_batch), xs_test_loader, xo_calib_train_loader, xo_calib_val_loader)) 

                losses.append(val_loss)
                wandb.log({f"fold_{fold+1}_val_loss": val_loss, 'fold': fold+1})
                fold_models.append((rope_pipeline, pipeline, wass_pipeline))
                if kf.n_splits==2:
                    break

            # log CV results
            avg_loss = sum(losses) / len(losses)
            std_loss = (sum((l - avg_loss)**2 for l in losses) / len(losses))**0.5
            wandb.log({f"avg_val_loss_Wass_OT_align_finetune": avg_loss, f"std_val_loss_Wass_OT_align_finetune": std_loss})

            # evaluate on test set
            xo_calib_test_loader = DataLoader(dataset['test'], batch_size=global_config['training']['batch_size'], shuffle=False, num_workers=num_workers)
            test_loaders = (xs_test_loader,xo_calib_test_loader, train_xo_loader, val_xo_loader)
            evaluate_test_set(fold_models, test_loaders)

    # launch sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, sweep_cv, count = 1)

if __name__ == '__main__':
    parser = ArgumentParser(description='Run a full experiment sweep')
    parser.add_argument('--config',       type=str, required=True, help='Path to config YAML')
    parser.add_argument('--num_samples',  type=int, required=True, help='Number of training samples')
    parser.add_argument('--project_name', type=str, required=True, help='WandB project name')
    parser.add_argument('--label_noise', type=float, default=0.0, help='Label noise percentage')
    parser.add_argument('--debug', action='store_true', help='Debug mode, single epoch, no wandb')
    args = parser.parse_args()

    # Always use GPU 0 or CPU fallback
    gpu_id = 0
    run_sweep(args.config, args.num_samples, gpu_id, args.project_name, args.label_noise, args.debug)
