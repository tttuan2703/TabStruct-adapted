
import os 
import json

def find_checkpoint(base_path, encoding_type, logger=None):
    """Find checkpoint in the specified task and encoding type directories."""
    models_path = os.path.join(base_path, 'models', encoding_type, "synthetic")
    if logger:
        logger.info(f"Searching for checkpoints in: {models_path}")
    try:
        paths = os.listdir(models_path)
        checkpoint = [path for path in paths if path.startswith('checkpoint')]
        if len(checkpoint) == 1:
            return os.path.join(models_path, checkpoint[0])
        else:
            message = f"Expected one checkpoint, but found: {checkpoint}"
            if logger:
                logger.warning(message)
            raise ValueError(message)
    except FileNotFoundError:
        message = f"No directory found for: {models_path}"
        if logger:
            logger.error(message)
        raise ValueError(message)
    
