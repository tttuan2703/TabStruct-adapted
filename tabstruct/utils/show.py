import pandas as pd
import random
import re

def show_example(datasets, mode, logger):
    logger.info("\n\n\n*** Random train example ***\n\n\n")
    rnd = random.randint(0, len(datasets[mode])) -1
    logger.info(f"random example {mode} {rnd}")
    example = datasets[mode][rnd]
    question = example["question"]
    df = pd.DataFrame(example["table"]["rows"], columns=example["table"]["header"])
    answers = example["answers"]
    logger.info(f"\nQuestion: {question}")
    logger.info(f"\nAnswers: {answers}")
    logger.info(f"\n{df}\n\n\n")


def decode_example(example, tokenizer):
    seq = tokenizer.decode(example['input_ids'], skip_special_tokens=True).strip()
    query, table = seq.split(' col : ')
    header, table = table.split(' row 1 : ')

    rows = [i.strip().split(' | ') for i in re.split(r'row \d+ :', table)]
    header = header.split(' | ')
    
    df = pd.DataFrame(rows, columns=header)
    labels = [i for i in example["labels"] if i not in [0,2,-100]]
    labels = tokenizer.decode(labels).strip()
        
    return df, query, labels

