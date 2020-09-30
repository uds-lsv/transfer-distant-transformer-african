import os
import torch
import logging

import numpy as np
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class Instance:

    def __init__(self, text, label):
        self.text = text
        self.label = label


def read_instances_from_file(data_dir, filename, delimiter="\t"):
    filepath = os.path.join(data_dir, filename)
    instances = []

    with open(filepath, "r") as input_file:
        next(input_file)  # skip header line
        for line in input_file:
            line = line.strip().split(delimiter)
            if len(line) == 0:
                continue
            elif len(line) == 4: # Yoruba data
                text, label = line[2:4]
            elif len(line) == 5: # Yoruba noisy data
                text, label = line[3:5]
            elif len(line) == 2: # Hausa data
                text, label = line
            elif len(line) == 3: # Hausa noisy data
                text, label = line[1:3]
            else:
                raise Exception(f"Line {line} does not have the expected number of columns.")

            instances.append(Instance(text, label))

    return instances

def create_label_map(label_list):
    return {label: i for i, label in enumerate(label_list)}

def convert_instances_to_features_and_labels(instances, tokenizer, exp_config):
    label_map = {label: i for i, label in enumerate(exp_config["labels"])}

    token_ids = []
    token_type_ids = []
    attention_masks = []
    labels = []
    for instance_idx, instance in enumerate(instances):
        tokenization_result = tokenizer.encode_plus(text=instance.text,
                                                    max_length=exp_config["max_seq_length"], pad_to_max_length=True,
                                                    return_overflowing_tokens=True)
        token_ids.append(tokenization_result["input_ids"])
        if exp_config["model_with_token_types"]: # XLM and RoBERTa don"t use token_type/segment_ids
            token_type_ids.append(tokenization_result["token_type_ids"])
        attention_masks.append(tokenization_result["attention_mask"])
        labels.append(label_map[instance.label])

        if "num_truncated_tokens" in tokenization_result:
            logger.info(f"Removed {tokenization_result['num_truncated_tokens']} tokens from {instance.text} as they "
                         f"were longer than max_seq_length {exp_config['max_seq_length']}.")

        if instance_idx < 3:
            logger.info("Tokenization example")
            logger.info(f"  text: {instance.text}")
            logger.info(f"  tokens (by input): {tokenizer.tokenize(instance.text)}")
            logger.info(f"  token_ids: {tokenization_result['input_ids']}")
            if exp_config["model_with_token_types"]:
                logger.info(f"  token_type_ids: {tokenization_result['token_type_ids']}")
            logger.info(f"  attention mask: {tokenization_result['attention_mask']}")

    # Convert to Tensors and build dataset
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    if exp_config["model_with_token_types"]:
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        return token_ids, token_type_ids, attention_masks, labels
    else:
        return token_ids, attention_masks, labels


def load_dataset(filename, tokenizer, exp_config, subset_indices=None):
    instances = read_instances_from_file(exp_config["data_dir"], filename)
    logging.info(f"Dataset {filename} has {len(instances)} instances.")
    if subset_indices is not None:
        instances = np.array(instances)[subset_indices]
        logging.info(f"Loading subset of dataset with indices {subset_indices}.")
    else:
        logging.info("Loading full dataset.")

    features_and_labels = convert_instances_to_features_and_labels(instances, tokenizer, exp_config)
    assert (exp_config["model_with_token_types"] and len(features_and_labels) == 4) or \
           (not exp_config["model_with_token_types"] and len(features_and_labels) == 3)

    dataset = TensorDataset(*features_and_labels)
    return dataset


def batch_to_inputs(batch, exp_config):
    """
    Convert the output of the DataLoader to the inputs expected by the model.
    :return:
    """
    batch = tuple(t.to(exp_config["device"]) for t in batch)
    if exp_config["model_with_token_types"]:
        inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2]}
    else: # XLM and RoBERTa don't use segment_ids/token_types
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
    labels = batch[-1]
    return inputs, labels

