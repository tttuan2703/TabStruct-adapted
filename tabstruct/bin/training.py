import os
import numpy as np
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint
from collections import defaultdict
import torch
from typing import Optional

from transformers import Seq2SeqTrainer, EarlyStoppingCallback

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabstruct.metrics.base_metric import get_denotation_accuracy, postprocess_text



class CustomSeq2SeqTrainer(Seq2SeqTrainer):

    def _save(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model using `save_pretrained()` with safe_serialization=False.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        
        self.model.save_pretrained(output_dir, safe_serialization=False)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False):

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        if return_outputs:
            return loss, outputs
        else:
            return loss
    


def setup_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics=None):

    # Set up early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=15,     
        early_stopping_threshold=0.0   # Threshold for improvement
    )

    #model.model_body[0].max_seq_length = 512
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[early_stopping_callback] if training_args.do_train and training_args.load_best_model_at_end  else None,
    )
    return trainer

def run_training(trainer, data_args, training_args, logger):

    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
    else:
        last_checkpoint = None

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    logger.info(f"checkpoint : {checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics


    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

def run_evaluation(trainer, data_args, eval_dataset):
    metrics = trainer.evaluate(max_new_tokens=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval", min_length=1)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def run_prediction(trainer, data_args, predict_dataset):
    predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_new_tokens=data_args.val_max_target_length, num_beams=data_args.num_beams)
    metrics = predict_results.metrics

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    return metrics



def count_until_second_zero(input_list):
    zero_count = 0  
    for index, element in enumerate(input_list):
        if element == 0:
            zero_count += 1
            if zero_count == 2:
                return index + 1 
    return -1