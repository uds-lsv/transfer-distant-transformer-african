import logging
import os
import sys

import torch

from utils import set_seed
from trainers.basetrainer import BaseTrainer
from trainers.cmtrainer import ConfusionMatrixTrainer
from trainers.evaltrainer import EvalTrainer

def run_experiment(config):
    set_seed(config["seed"])

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    log_filehandler = logging.FileHandler(os.path.join(config["output_dir"], "output.log"), 'w')
    logging.getLogger().addHandler(log_filehandler)

    if config["trainer"] == "base":
        trainer = BaseTrainer(config)
    elif config["trainer"] == "cmmodel":
        trainer = ConfusionMatrixTrainer(config)
    elif config["trainer"] == "eval":
        trainer = EvalTrainer(config)
    else:
        raise Exception("Unknown trainer")

    trainer.train(config)

    del trainer
    torch.cuda.empty_cache()

    #CHANGE_ME storing the trained models requires a lot of disc space
    # Remove trained model to save space
    #import shutil
    #model_dir = os.path.join(config["output_dir"], "checkpoint-best-on-dev")
    #shutil.rmtree(model_dir)

    logging.getLogger().removeHandler(log_filehandler)

def main_yoruba_base():

    config = {
        "model_name": "bert-base-multilingual-cased",
        "trainer": "base",
        "data_dir": "../../data/yoruba_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, -1], [10, 14, 36, 57, 92, 116]):  # -1 is 816
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_ner_base_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_base_fulldev():

    config = {
        "model_name": "bert-base-multilingual-cased",
        "trainer": "base",
        "data_dir": "../../data/yoruba_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size in [10, 100, 250, 400, 650, -1]:  # -1 is 816
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_ner_base_full-dev_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)


def main_yoruba_nodate_base():

    config = {
        "model_name": "bert-base-multilingual-cased",
        "trainer": "base",
        "data_dir": "../../data/yoruba_ner_no-date/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, -1], [10, 14, 36, 57, 92, 116]):  # -1 is 816
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_ner_no-date_base_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)


def main_yoruba_base_distilled():

    config = {
        "model_name": "distilbert-base-multilingual-cased",
        "trainer": "base",
        "data_dir": "../../data/yoruba_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, -1], [10, 14, 36, 57, 92, 116]):  # -1 is 816
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_ner_base-distilled_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_base_and_noise():

    config = {
        "model_name": "bert-base-multilingual-cased",
        "trainer": "base",
        "data_dir": "../../data/yoruba_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": True,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, -1], [10, 14, 36, 57, 92, 116]):  # -1 is 816
        if clean_subset_size == -1:
            noisy_subsample_size = 933
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_ner_base+noise_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_cmmodel_smoothed():

    config = {
        "model_name": "bert-base-multilingual-cased",
        "trainer": "cmmodel",
        "data_dir": "../../data/yoruba_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False,
        "use_true_cm": False,
        "use_io_estimation": False,
        "matrix_smoothing_value": 0.8
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, -1], [10, 14, 36, 57, 92, 116]):  # -1 is 816
        if clean_subset_size == -1:
            noisy_subsample_size = 933
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_ner_cm-smoothed_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_cmmodel():

    config = {
        "model_name": "bert-base-multilingual-cased",
        "trainer": "cmmodel",
        "data_dir": "../../data/yoruba_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False,
        "use_true_cm": False,
        "use_io_estimation": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, -1], [10, 14, 36, 57, 92, 116]):  # -1 is 816
        if clean_subset_size == -1:
            noisy_subsample_size = 933
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_ner_cm_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_nodate_base_fewshot():

    config = {
        "model_name": "../../logs/mbert_conll_en/checkpoint-best-on-dev", #CHANGE_ME # Needs to be pretrained first with main_conll_en_mbert()!
        "trainer": "base",
        "data_dir": "../../data/yoruba_ner_no-date/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, -1], [10, 14, 36, 57, 92, 116]):  # -1 is 816
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_ner_no-date_base-fewshot_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_nodate_base_zeroshot():

    config = {
        "trained_model_directory": "../../logs/mbert_conll_en/", #CHANGE_ME # Needs to be pretrained first with main_conll_en_mbert()!
        "trainer": "eval",
        "seed": 12345,
        "eval_data_dir": "../../data/yoruba_ner_no-date/", #CHANGE_ME
        "eval_file": "test.tsv",
        "output_dir": f"../../logs/yoruba_ner_no-date_base-zeroshot" #CHANGE_ME
    }
    run_experiment(config)

def main_hausa_base():

    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "base",
        "data_dir": "../../data/hausa_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, 800, -1], [10, 14, 36, 57, 93, 114, 145]):  # -1 is 1014
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_ner_base_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_hausa_base_fulldev():

    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "base",
        "data_dir": "../../data/hausa_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size in [10, 100, 250, 400, 650, 800, -1]:  # -1 is 1014
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_ner_base_full-dev_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)


def main_hausa_nodate_base():

    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "base",
        "data_dir": "../../data/hausa_ner_no-date/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, 800, -1], [10, 14, 36, 57, 93, 114, 145]):  # -1 is 1014
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_ner_no-date_base_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_hausa_base_and_noise():

    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "base",
        "data_dir": "../../data/hausa_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": True,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, 800, -1], [10, 14, 36, 57, 93, 114, 145]):  # -1 is 1014
        if clean_subset_size == -1:
            noisy_subsample_size = 968
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_ner_base+noise_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_hausa_cmmodel_smoothed():

    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "cmmodel",
        "data_dir": "../../data/hausa_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False,
        "use_true_cm": False,
        "use_io_estimation": False,
        "matrix_smoothing_value": 0.8
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, 800, -1], [10, 14, 36, 57, 93, 114, 145]):  # -1 is 1014
        if clean_subset_size == -1:
            noisy_subsample_size = 968
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_ner_cm-smoothed_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_hausa_cmmodel():

    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "cmmodel",
        "data_dir": "../../data/hausa_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False,
        "use_true_cm": False,
        "use_io_estimation": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, 800, -1], [10, 14, 36, 57, 93, 114, 145]):  # -1 is 1014
        if clean_subset_size == -1:
            noisy_subsample_size = 968
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_ner_cm_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)


def main_hausa_nodate_base_fewshot():

    config = {
        "model_name": "../../logs/xlmroberta_conll_en/checkpoint-best-on-dev", #CHANGE_ME # Needs to be pretrained first with main_conll_en_xlmroberta()!
        "trainer": "base",
        "data_dir": "../../data/hausa_ner_no-date/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, 800, -1], [10, 14, 36, 57, 93, 114, 145]):  # -1 is 1014
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_ner_no-date_base-fewshot_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_hausa_nodate_base_zeroshot():

    config = {
        "trained_model_directory": "../../logs/xlmroberta_conll_en/", #CHANGE_ME # Needs to be pretrained first with main_conll_en_xlmroberta()!
        "trainer": "eval",
        "seed": 12345,
        "eval_data_dir": "../../data/hausa_ner_no-date/", #CHANGE_ME
        "eval_file": "test.tsv",
        "output_dir": f"../../logs/hausa_ner_no-date_base-zeroshot" #CHANGE_ME
    }
    run_experiment(config)


def main_xhosa_base():

    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "base",
        "data_dir": "../../data/xhosa_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "B-MISC", "I-MISC", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, 800, 1500, 2000, 3000, 4000, -1], [10, 12, 30, 47, 77, 95, 178, 237, 355, 473, 608]):  # -1 is 5138
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/xhosa_ner_base_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_xhosa_base_fulldev():

    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "base",
        "data_dir": "../../data/xhosa_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "B-MISC", "I-MISC", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size in [10, 100, 250, 400, 650, 800, 1500, 2000, 3000, 4000, -1]:  # -1 is 5138
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/xhosa_ner_base_full-dev_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_xhosa_base_zeroshot():

    config = {
        "trained_model_directory": "../../logs/xlmroberta_conll_en_orig-labels/", #CHANGE_ME # Needs to be pretrained first with main_conll_en_xlmroberta_origlabels()!
        "trainer": "eval",
        "seed": 12345,
        "eval_data_dir": "../../data/xhosa_ner/", #CHANGE_ME
        "eval_file": "test.tsv",
        "output_dir": f"../../logs/xhosa_ner_base-zeroshot" #CHANGE_ME
    }
    run_experiment(config)

def main_xhosa_base_fewshot():
    config = {
        "model_name": "../../logs/xlmroberta_conll_en_orig-labels/checkpoint-best-on-dev", #CHANGE_ME # Needs to be pretrained first with main_conll_en_xlmroberta_origlabels()!
        "trainer": "base",
        "data_dir": "../../data/xhosa_ner/", #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset_start": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 250, 400, 650, 800, 1500, 2000, 3000, 4000, -1], [10, 12, 30, 47, 77, 95, 178, 237, 355, 473, 608]):  # -1 is 5138
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/xhosa_ner_base_fewshot_subset-size{clean_subset_size}_seed{seed}"  #CHANGE_ME
            run_experiment(config)

def main_conll_en_mbert():

    config = {
        "model_name": "bert-base-multilingual-cased",
        "trainer": "base",
        "data_dir": "../../data/conll_en/",  #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "seed": 12345,
        "device": "cuda",
        "output_dir": "../../logs/mbert_conll_en"  #CHANGE_ME
    }

    run_experiment(config)

def main_conll_en_xlmroberta():

    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "base",
        "data_dir": "../../data/conll_en/",  #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "I-DATE", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "seed": 12345,
        "device": "cuda",
        "output_dir": "../../logs/xlmroberta_conll_en"  #CHANGE_ME
    }

    run_experiment(config)

def main_conll_en_xlmroberta_origlabels():
    config = {
        "model_name": "xlm-roberta-base",
        "trainer": "base",
        "data_dir": "../../data/conll_en_orig-labels/",  #CHANGE_ME
        "labels": ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 32,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "seed": 12345,
        "device": "cuda",
        "output_dir": "../../logs/xlmroberta_conll_en_orig-labels"  #CHANGE_ME
    }

    run_experiment(config)

if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(logging.StreamHandler(sys.stdout))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #CHANGE_ME
    # Call here the method for the model, you want to train, e.g.
    main_yoruba_base()
    
