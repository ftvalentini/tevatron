import logging
import os
import sys
import json
from dataclasses import asdict

import torch
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.collator import TrainCollator
from tevatron.retriever.modeling import DenseModel, DenseModelWithPriors
from tevatron.retriever.trainer import TevatronTrainer as Trainer
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.padding_side == 'right':
        tokenizer.padding_side = 'right'
    else:
        tokenizer.padding_side = 'left'
    
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Select model class based on whether to use document priors
    model_cls = DenseModelWithPriors if model_args.use_document_priors else DenseModel
    
    model = model_cls.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, tokenizer)

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer, # We need this there so that tokenizer is saved in every checkpoint
    )
    train_dataset.set_trainer(trainer)
    
    # Log all arguments to wandb
    trainer.add_callback(ConfigCallback(model_args, data_args))

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Save all arguments (useful for inference e.g. passage_max_len, pooling, etc.)
        save_configs(training_args.output_dir, model_args, data_args, training_args)


class ConfigCallback(TrainerCallback):
    """A callback that logs the model and data arguments to wandb at the start of training.
    """
    def __init__(self, model_args, data_args):
        self.model_args = model_args
        self.data_args = data_args

    def on_train_begin(self, args, state, control, **kwargs):
        """
        This runs at the beginning of training, after wandb is initialized.
        """
        # We only import wandb inside the method to avoid top-level dependency
        try:
            import wandb
            if wandb.run is not None:
                # Nest custom dataclasses under their own keys in wandb config
                wandb.config.update({
                    "model_args": asdict(self.model_args),
                    "data_args": asdict(self.data_args)
                }, allow_val_change=True)
        except ImportError:
            pass


def save_configs(output_dir, model_args, data_args, training_args):
    """Save the dataclass arguments to JSON files in the output directory."""
    with open(os.path.join(output_dir, "model_args.json"), 'w') as f:
        json.dump(asdict(model_args), f, indent=4)
    with open(os.path.join(output_dir, "data_args.json"), 'w') as f:
        json.dump(asdict(data_args), f, indent=4)
    with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
        json.dump(asdict(training_args), f, indent=4)

if __name__ == "__main__":
    main()
