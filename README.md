# Inductive Domain Transfer in Misspecified SBI

This repository provides the implementation of the NeurIPS25' paper: Inductive Domain Transfer in Misspecified Simulation-Based Inference.

The code enables reproduction of all experiments **except** the one involving arterial pressure waves (Figure 5), which requires credentials and a signed agreement to access the [MIMIC](https://physionet.org/content/mimiciii/1.4/) dataset.

---

### Code Overview

The main script is `main_cv_OT_NF.py`.
It performs **k-fold cross-validation** by splitting the calibration set into `k` folds of train and validation subsets.
A test set is held out throughout all experiments and baselines. Final test metrics are reported as the **mean and standard deviation** across models trained on each fold.

The checkpoint selected for each trained model is the one achieving the **lowest validation loss**.
This script trains both the proposed pipeline and the baselines, using the implementation in `pipelines/NFOT.py`.

---

### Hyperparameters and Configurations

You can find configuration files for each experiment under the `configs` directory. These define:

* Dataset details
* Backbone architecture
* Training hyperparameters

To reproduce the results reported in the manuscript, use the provided configuration files as-is.  
If you wish to perform hyperparameter sweeps, you can populate the `sweep_hyperparams` field with multiple values.  
Note that this work focuses on evaluating the **pipeline**, rather than on extensive hyperparameter tuning. Only minimal tuning was performed—primarily for batch size and learning rate.  

Additionally, we did **not** tune the weight of the supervised loss term, denoted as `lambda` in the joint training objective (Equation 4). This weight was fixed at `lambda = 1` across all experiments and does **not** appear as a configurable parameter in the provided files.


For convenience, we provide pre-trained checkpoints for the NPE and NSE models, which are fixed across all folds (folds are defined with respect to the calibration set).
These checkpoints are already referenced in the config files and will be automatically loaded during training.
The number of folds is also specified in the config files (default: `5`).


---

### Running an Experiment

To run the wind tunnel benchmark experiment (other benchmarks work similarly, just change the config path), use:

```bash
python main_cv_OT_NF.py --config ./configs/CV_WindTunnel_NFOT.yaml --num_samples 50 --label_noise 0 --project_name <WANDB_PROJECT_NAME>
```

* `--num_samples`: size of the calibration set
* `--label_noise`: rate of label noise to be added

You may also add the `--debug` flag to run only one epoch per model for debugging purposes.

All experiments were run on machines equipped with an NVIDIA A100 80GB GPU, with data loading parallelized across 4 CPU cores. Training the normalizing flows—especially due to the need to sample, store those samples, and retain their gradients in memory—requires a high-memory GPU such as the A100 80GB.
 

---

### Analysis

All training, validation, and test metrics are logged to **Weights & Biases (wandb)**.
You can inspect and analyze them manually via the wandb dashboard.

To reproduce plots similar to those in the manuscript, follow the instructions in `analysis.ipynb`.
You will need to provide the wandb remote paths for the relevant runs as detailed in the notebook.

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{senoufinductive,
  title={Inductive Domain Transfer In Misspecified Simulation-Based Inference},
  author={Senouf, Ortal and Wehenkel, Antoine and Vincent-Cuaz, C{\'e}dric and Abbe, Emmanuel and Frossard, Pascal},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
