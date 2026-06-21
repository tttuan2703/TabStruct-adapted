# Imports standards
import os

# Imports Transformers
from transformers import (
    AutoConfig,
    BartTokenizer,
    TapasConfig,
    TapasModel,
    BartForConditionalGeneration
)

# Imports spécifiques au projet
from tabstruct.models.modeling_tab_struct import TabStructForConditionalGeneration

def load_config(data_args, model_args, logger ):
    logger.info(f"loadconfig..")

    config = AutoConfig.from_pretrained(
        model_args.config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        early_stopping = False,
        no_repeat_ngram_size = 0,
    )

    model_args.ignore_mismatched_sizes = False

    tapasconfig =  TapasConfig()
    config.type_vocab_sizes = tapasconfig.type_vocab_sizes 

    # config the model structure
    config.tabular_structure_embedding = model_args.tabular_structure_embedding
    config.encoding_structure_bias = model_args.encoding_structure_bias
    config.positional_embedding = model_args.positional_embedding
    config.mask_number = int(model_args.mask_sparsity_level[-1])
    config.input_token_structure = model_args.input_token_structure


    config._attn_implementation = "sdpa" # To use StructAttention
    logger.info(f"attention_type sdpa")

    logger.info(f"config : {config}")
    logger.info(f"mask_number : {config.mask_number}")
    return config


def load_tokenizer(model_args, logger = None):

    tokenizer = BartTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            add_prefix_space=True,)

    if logger is not None:
        logger.info(f"tokenizer : {tokenizer}")     

    return tokenizer


def load_model(model_args, config, logger):
    logger.info('\n' * 5)


    if model_args.model_name_or_path is None:
        model = TabStructForConditionalGeneration(config=config)
        logger.info('Loaded TabStruct from scratch')
        
    else: # init model using bart weight
        model = TabStructForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, 
            config=config, 
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes)

        logger.info(f'Loaded TabStruct from checkpoint: {model_args.model_name_or_path}')
        if model_args.task == "train" and not "checkpoint" in model_args.model_name_or_path.split('/')[-1]:
            # load extra weights only if model_name_or_path is a pre-trained model 
            model = load_extra_weights(model, model_args, config, logger)

    logger.info('\n' * 5)
    logger.info(model)
    return model


def load_extra_weights(model, model_args, config, logger):
    # Load BART weights
    model2 = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, 
        config=config, 
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,)
    
    if model_args.positional_embedding in ["TPE", "CPE"]:
        model.model.encoder.embed_positions.embed_positions.weight = model2.model.encoder.embed_positions.weight
        logger.info(f'Loaded extra weights for {model_args.positional_embedding} (Table Positional Embedding)')

    del model2

    # Load Tapas segment embeddings
    model2 = TapasModel.from_pretrained(model_args.tapas_path)

    model.model.encoder.embed_positions.segment_embedding.weight = model2.embeddings.token_type_embeddings_0.weight
    logger.info('Loaded extra weights for Segment Embedding')

    if model_args.tabular_structure_embedding in ["RCE", "RRCE"]:
        model.model.encoder.embed_positions.tabular_structure_embedding.token_type_embeddings_1.weight = model2.embeddings.token_type_embeddings_1.weight
        model.model.encoder.embed_positions.tabular_structure_embedding.token_type_embeddings_2.weight = model2.embeddings.token_type_embeddings_2.weight
        logger.info('Loaded extra weights for token_type_embeddings_1 and 2')

    del model2

    return model




