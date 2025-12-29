import os
from typing import Optional
import json
from dataclasses import asdict

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
from transformers.trainer_callback import TrainerCallback
import torch.distributed as dist
from .modeling import EncoderModel

import logging
logger = logging.getLogger(__name__)


class EpochCheckpointCallback(TrainerCallback):
    """
    Callback to save checkpoints every N epochs.
    """
    def __init__(self, save_epochs: int):
        self.save_epochs = save_epochs
        self.last_saved_epoch = -1

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Event called at the end of an epoch.
        """
        current_epoch = int(state.epoch)
        
        # Check if we should save based on save_epochs interval
        if self.save_epochs > 0 and current_epoch > 0:
            if current_epoch % self.save_epochs == 0 and current_epoch != self.last_saved_epoch:
                control.should_save = True
                self.last_saved_epoch = current_epoch
                logger.info(f"Triggering checkpoint save at epoch {current_epoch}")
        
        return control


class TevatronTrainer(Trainer):
    def __init__(self, *args, model_args=None, data_args=None, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self._pending_tracking_metrics = None  # Store tracking metrics to merge with loss logs
        self.model_args = model_args  # Store model args for saving in checkpoints
        self.data_args = data_args  # Store data args for saving in checkpoints
        
        # Add epoch checkpoint callback if save_epochs is specified
        if hasattr(self.args, 'save_epochs') and self.args.save_epochs is not None:
            epoch_callback = EpochCheckpointCallback(save_epochs=self.args.save_epochs)
            self.add_callback(epoch_callback)
            logger.info(f"Added epoch checkpoint callback: saving every {self.args.save_epochs} epochs")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()` or custom save method
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            # Call model's save method if available, otherwise use standard approach
            if hasattr(self.model, 'save') and callable(self.model.save):
                # Use model's custom save method (handles encoder + any additional components)
                self.model.save(output_dir)
            else:
                # Standard save for models without custom save method
                if state_dict is None:
                    state_dict = self.model.state_dict()
                prefix = 'encoder.'
                assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
                state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
                self.model.encoder.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

        # Use processing_class for compatibility with newer transformers versions
        tokenizer = self.processing_class
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        
        # Save config arguments if available
        if self.model_args is not None:
            with open(os.path.join(output_dir, "model_args.json"), 'w') as f:
                json.dump(asdict(self.model_args), f, indent=4)
        if self.data_args is not None:
            with open(os.path.join(output_dir, "data_args.json"), 'w') as f:
                json.dump(asdict(self.data_args), f, indent=4)
        if self.args is not None:
            with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
                json.dump(asdict(self.args), f, indent=4)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage = inputs
        output = model(query=query, passage=passage)
        
        # Store tracking metrics to be merged with loss logs later
        # Handle both wrapped (DDP/DeepSpeed) and unwrapped models
        unwrapped_model = self.accelerator.unwrap_model(model) if hasattr(self, 'accelerator') else model
        if hasattr(unwrapped_model, '_tracking_metrics') and unwrapped_model._tracking_metrics:
            self._pending_tracking_metrics = unwrapped_model._tracking_metrics
        
        return output.loss
    
    def log(self, logs, *args, **kwargs) -> None:
        """
        Override log method to merge tracking metrics with loss/grad_norm logs.
        This ensures all metrics appear on a single line.
        """
        # If we have pending tracking metrics and this log contains training metrics, merge them
        if self._pending_tracking_metrics and ('loss' in logs or 'grad_norm' in logs):
            logs.update(self._pending_tracking_metrics)
            self._pending_tracking_metrics = None  # Clear after merging
        
        # Call parent's log method
        super().log(logs, *args, **kwargs)

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor


class DistilTevatronTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage, reranker_labels = inputs
        scores = model(query=query, passage=passage).scores
        
        if model.is_ddp:
            # reranker_scores are gathered across all processes
            reranker_labels = model._dist_gather_tensor(reranker_labels)
        
        # Derive student_scores [batch, num_labels]
        batch_size, total_passages = scores.size()
        num_labels = reranker_labels.size(1)
        start_idxs = torch.arange(0, batch_size * num_labels, num_labels, device=scores.device)
        idx_matrix = start_idxs.view(-1, 1) + torch.arange(num_labels, device=scores.device)
        student_scores = scores.gather(1, idx_matrix)

        # Temperature‐scaled soft distributions
        T = self.args.distil_temperature
        student_log   = torch.log_softmax(student_scores.float() / T, dim=1)
        teacher_probs = torch.softmax(reranker_labels.float()    / T, dim=1)

        # KL Divergence loss (shapes now [batch, num_labels])
        loss = torch.nn.functional.kl_div(
            student_log,
            teacher_probs,
            reduction="batchmean"
        ) * self._dist_loss_scale_factor

        return loss

    def training_step(self, *args):
        return super(DistilTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor
