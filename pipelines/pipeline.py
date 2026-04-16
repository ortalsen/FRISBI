from torch.nn import Module
from abc import ABC, abstractmethod
import torch
import os

class BaseStage(Module):
    def __init__(self):
        super(BaseStage, self).__init__()
        # Any common initialization code can go here

    @abstractmethod
    def train_step(self, inputs, labels, pretrained_encoder=None):
        """Method for the training step, to be implemented by subclasses."""
        pass

    @abstractmethod
    def eval_step(self, inputs, labels, pretrained_encoder=None):
        """Method for the evaluation step, to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def data_preprocess(self, data):
        """Method for preprocessing data, to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def metrics_calculate(self, data):
        """Method for calculating metrics, to be implemented by subclasses."""
        pass

    def save_checkpoint(self, path):
        fold_suffix = '' if self.fold is None else f"_{self.fold}"
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        if not os.path.exists(os.path.join('./checkpoints',path)):
            os.makedirs(os.path.join('./checkpoints',path))
        if not os.path.exists(os.path.join('./checkpoints',path, self.__class__.__name__)):
            os.makedirs(os.path.join('./checkpoints',path, self.__class__.__name__))
        file_path = os.path.join('./checkpoints', path, self.__class__.__name__, f'best_model{fold_suffix}.pth') 
        if hasattr(self, 'normalisers'):
            torch.save({'model_state_dict': self.state_dict(), 'normalisers': self.normalisers}, file_path)
        else:
            torch.save(self.state_dict(), file_path)


    def load_checkpoint(self, path):
        fold_suffix = '' if self.fold is None else f"_{self.fold}"
        file_path = os.path.join('./checkpoints', path, self.__class__.__name__, f'best_model{fold_suffix}.pth')
        
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            
            # Check if 'normalisers' exists in the checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                if 'normalisers' in checkpoint:
                    self.normalisers = checkpoint['normalisers']
                print(f"Loaded model and normalisers from {file_path}")
            else:
                # If checkpoint is a plain state_dict (for backward compatibility)
                self.load_state_dict(checkpoint)
                print(f"Loaded model state_dict from {file_path}")
            
            return True
        else:
            print(f"Checkpoint not found at {file_path}")
            return False
 
    
    def scale_params(self, params, name, inverse=False):
        '''Scale the params with the known prior distribution'''
        scalers_type = self.params_infer[name]['scalers']['type']
        scalers = self.params_infer[name]['scalers']['params']
        if scalers_type == 'uniform':
            scaler_max = scalers['max']
            scaler_min = scalers['min']
            if inverse:
                # Transform back to the original uniform distribution
                params = ((params + 1) / 2) * (scaler_max - scaler_min) + scaler_min
            else:
                # Scale the values to [-1, 1]
                params = 2 * ((params - scaler_min) / (scaler_max - scaler_min)) - 1
        elif scalers_type == 'normal':
            scaler_mean = scalers['mean']
            scaler_std = scalers['std']
            if inverse:
                params = params * scaler_std + scaler_mean
            else:
                params = (params - scaler_mean) / scaler_std
        return params



class BasePipeline(ABC):
    '''This class is a wrapper around a list of stages, each of which is a neural network module.
    The train_pipeline method trains all the stages in the pipeline sequentially, 
    the method train_stage trains a single stage, and the method test_pipeline tests only the inference of the pipeline.'''

    @abstractmethod
    def train_stage(self, stage, train_dataloader, val_dataloader, early_stopping=True, patience=5):
        '''
        Train a single stage in the pipeline.
        
        Args:
            stage: The stage to be trained (stage object).
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            early_stopping: Whether to use early stopping.
            patience: Number of epochs with no improvement after which training will be stopped.
        '''
        pass
    
    @abstractmethod
    def train_pipeline(self, dataloaders, early_stopping=(True, True, True), patience=(5, 5, 5)):
        '''
        Train all stages in the pipeline sequentially.

        Args:
            dataloaders: A tuple or a dictionary of relevant dataloaders for the pipeline.
            early_stopping: A tuple of booleans indicating whether early stopping should be used for each stage.
            patience: A tuple of integers indicating the patience for each stage.
        '''
        pass

    @abstractmethod
    def test_pipeline(self, dataloaders):
        '''
        Test the entire pipeline.

        Args:
            dataloaders: A tuple or a dictionary of relevant dataloaders for testing.
        '''
        pass

    def mount_to_device(self, tensors):
        if isinstance(tensors, torch.Tensor):
            return tensors.to(self.device)
        elif isinstance(tensors, list):
            return [self.mount_to_device(t) for t in tensors]
        elif isinstance(tensors, tuple):
            return tuple(self.mount_to_device(t) for t in tensors)
        elif isinstance(tensors, dict):
            return {k: self.mount_to_device(v) for k, v in tensors.items()}
        else:
            return tensors

    def wandb_log_metrics(self, metrics, phase):
        for key, value in metrics.items():
            phase_key = f"{phase}_{key}"
            self.wandb_logger.log({phase_key:value})