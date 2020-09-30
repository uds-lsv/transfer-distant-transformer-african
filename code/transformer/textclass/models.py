import torch
import numpy as np
import itertools

from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW, get_constant_schedule_with_warmup

from nn_models.cmmodel import ConfusionMatrixModel


def create_base_model(exp_config):
    labels = exp_config["labels"]

    config = AutoConfig.from_pretrained(
        exp_config["model_name"],
        num_labels=len(labels),
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )

    model = AutoModelForSequenceClassification.from_pretrained(exp_config["model_name"], config=config)
    model.to(exp_config["device"])
    return model

def compute_loss(logits, labels, is_log_prob=False):
    """
    Code adapted from the transformers.BertForTokenClassification code
    """
    if is_log_prob:
        loss_fct = torch.nn.NLLLoss()
    else:
        loss_fct = torch.nn.CrossEntropyLoss()

    return loss_fct(logits, labels)


def create_cm_model(base_model, noise_matrix, exp_config):
    channel_weights = np.log(noise_matrix + 1e-7)
    return ConfusionMatrixModel(base_model, channel_weights, exp_config)

def create_optimizer(models, exp_config):
    # Prepare optimizer and schedule (linear warmup and decay)
    if isinstance(models, torch.nn.Module): # if it is just one model, put it into a list
        models = [models]
    parameters = itertools.chain(*[model.named_parameters() for model in models])

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": exp_config["weight_decay"],
        },
        {
            "params": [p for n, p in parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=exp_config["learning_rate"], eps=exp_config["adam_epsilon"])

    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=exp_config["warmup_steps"]
    )

    return optimizer, scheduler