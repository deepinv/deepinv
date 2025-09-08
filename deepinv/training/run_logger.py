from abc import ABC, abstractmethod
import json
import csv
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import warnings
from typing import Any, Optional, Union

class RunLogger(ABC):
    """
    Abstract base class for logging training runs.
    
    Defines the interface for logging metrics, losses, images, and other
    training artifacts during model training and evaluation.
    """
    
    def __init__(self, run_name: Optional[str] = None, config: Optional[dict[str, Any]] = None):
        """
        Initialize the logger.
        
        :param run_name: Optional name for the training run
        :param config: Configuration dictionary for the logger
        """
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.current_epoch = 0
        self.current_step = 0
        
    @abstractmethod
    def start_run(self, hyperparams: Optional[dict[str, Any]] = None):
        """
        Start a new training run.
        
        :param hyperparams: Dictionary of hyperparameters to log
        """
        pass

     @abstractmethod
    def log_losses(self, losses: dict[str, float], step: Optional[int] = None,
                  epoch: Optional[int] = None, phase: str = 'train'):
        """
        Log loss values for the current step/epoch.
        
        :param losses: Dictionary of loss_name -> value
        :param step: Current training step  
        :param epoch: Current epoch
        :param phase: Training phase ('train', 'eval', 'test')
        """
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None, 
                   epoch: Optional[int] = None, phase: str = 'train'):
        """
        Log metrics for the current step/epoch.
        
        :param metrics: Dictionary of metric_name -> value
        :param step: Current training step
        :param epoch: Current epoch
        :param phase: Training phase ('train', 'eval', 'test')
        """
        pass
    
    @abstractmethod
    def log_images(self, images: dict[str, Union[torch.Tensor, np.ndarray]], 
                  step: Optional[int] = None, epoch: Optional[int] = None, 
                  phase: str = 'train'):
        """
        Log images for visualization.
        
        :param images: Dictionary of image_name -> tensor/array
        :param step: Current training step
        :param epoch: Current epoch  
        :param phase: Training phase ('train', 'eval', 'test')
        """
        pass
    
    @abstractmethod
    def log_model_checkpoint(self, checkpoint_path: str, metrics: Optional[dict[str, float]] = None,
                           epoch: Optional[int] = None):
        """
        Log model checkpoint information.
        
        :param checkpoint_path: Path to the saved checkpoint
        :param metrics: Optional metrics associated with this checkpoint
        :param epoch: Epoch when checkpoint was saved
        """
        pass
    
    @abstractmethod
    def finish_run(self):
        """
        Finalize and close the training run.
        """
        pass
    
    def set_step(self, step: int):
        """Set the current step."""
        self.current_step = step
    
    def set_epoch(self, epoch: int):
        """Set the current epoch."""
        self.current_epoch = epoch