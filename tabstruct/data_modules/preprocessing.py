
from functools import partial
import os
from tabstruct.data_modules.T1_preprocessing import  preprocess_tableqa_function_T1
from tabstruct.data_modules.T0_preprocessing import preprocess_tableqa_function_T0
from tabstruct.data_modules.T2_preprocessing import preprocess_tableqa_function_T2
from tabstruct.processor import setup_table_processor


preprocess_func = {"T0":preprocess_tableqa_function_T0,
                   "T1":preprocess_tableqa_function_T1,
                   "T2":preprocess_tableqa_function_T2 }

def preprocess_datasets(datasets, tokenizer, data_args, model_args, training_args, logger):

    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    padding = "max_length" if data_args.pad_to_max_length else False
    TABLE_PROCESSOR = setup_table_processor(model_args, data_args)

    output = {}
    do_ = [training_args.do_train, training_args.do_eval, training_args.do_predict]
    modes = ['train', "validation", "test"]

    is_training = model_args.task == "train"

    for _, (do, mode) in enumerate(zip(do_, modes)):
        if not do:
            output[mode] = None
            continue

        preprocess_tableqa_function = partial(preprocess_func[model_args.input_token_structure],
                                                tokenizer = tokenizer,
                                                data_args=data_args,
                                                padding=padding,
                                                table_processor=TABLE_PROCESSOR,
                                                is_training=is_training,
                                                logger=logger)
            
        output[mode] = datasets[mode].map(preprocess_tableqa_function,
                                        batched=True, 
                                       #batch_size=64, num_proc=os.cpu_count() // 2,
                                        num_proc=data_args.preprocessing_num_workers,
                                        remove_columns=column_names,
                                        load_from_cache_file=not data_args.overwrite_cache)
        
        logger.info(f"Dataset {mode}")
        logger.info(output[mode])


    return output.get('train'), output.get('validation'), output.get('test')