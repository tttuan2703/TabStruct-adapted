
import os

from tabstruct.utils.logger import setup_logger
from tabstruct.utils.show import show_example
from tabstruct.models.model_setup import load_config, load_tokenizer, load_model
from tabstruct.utils.sanity_checks import check_parameters
from tabstruct.data_modules.data_loader import load_datasets
from tabstruct.data_modules.preprocessing import preprocess_datasets
from tabstruct.metrics.base_metric import compute_metrics
from tabstruct.bin.training import setup_trainer, run_training, run_evaluation, run_prediction

from transformers import DataCollatorForSeq2Seq

from functools import partial



def main_train(model_args, data_args, training_args):



    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    logger = setup_logger()

    check_parameters(model_args, data_args, training_args, logger)
    
    datasets = load_datasets(data_args, model_args, logger)

    config = load_config(data_args, model_args, logger)
    tokenizer = load_tokenizer(model_args, logger)
    model = load_model(model_args, config, logger)
    logger.info(datasets)

    show_example(datasets, "train", logger)

    train_dataset, eval_dataset, predict_dataset = preprocess_datasets(datasets, tokenizer, data_args, model_args, training_args, logger)
  
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True if data_args.pad_to_max_length else "longest",
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,)

    

    compute_metrics_ = partial(compute_metrics, tokenizer=tokenizer, data_args=data_args)
    trainer = setup_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics_)

    if training_args.do_train:
        run_training(trainer, data_args, training_args, logger)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        run_evaluation(trainer, data_args, eval_dataset)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        run_prediction(trainer, data_args, predict_dataset)

    return trainer.state.log_history

