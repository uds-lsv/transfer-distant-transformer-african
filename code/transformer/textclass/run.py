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

    trainer.train(config)

    del trainer
    torch.cuda.empty_cache()

    logging.getLogger().removeHandler(log_filehandler)

    #CHANGE_ME storing the trained models requires a lot of disc space
    # Remove trained model to save space
    #import shutil
    #model_dir = os.path.join(config["output_dir"], "checkpoint-best-on-dev")
    #shutil.rmtree(model_dir)

def main_yoruba_base_exp():
    config = {
        "trainer": "base",
        "model_name": "bert-base-multilingual-cased",
        "data_dir": "../../data/yoruba_newsclass/", #CHANGE_ME
        "labels": ["nigeria", "africa", "world", "entertainment", "health", "sport", "politics"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": True
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10,  14, 42, 71, 99, 127, 189]): # -1 is 1340
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_newsclass_base_bs128_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)


def main_yoruba_base_fewshot_exp():
    config = {
        "trainer": "base",
        "model_name": "../../logs/ag-news_newsclass_base_for-yoruba_bs128/checkpoint-best-on-dev", #CHANGE_ME # Needs to be pretrained first with main_agnews_for_yoruba_exp()!
        "data_dir": "../../data/yoruba_newsclass/",  #CHANGE_ME
        "labels": ["nigeria", "africa", "world", "entertainment", "health", "sport", "politics", "business", "sci/tech"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": True
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10,  14, 42, 71, 99, 127, 189]): # -1 is 1340
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_newsclass_base_few-shot_bs128_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_base_zeroshot_exp():
    config = {
        "trained_model_directory": "../../logs/ag-news_newsclass_base_for-yoruba_bs128",  #CHANGE_ME # Needs to be pretrained first with main_agnews_for_yoruba_exp()!
        "trainer": "eval",
        "seed": 12345,
        "eval_data_dir": "../../data/yoruba_newsclass/", #CHANGE_ME
        "eval_file": "test.tsv",
        "output_dir": f"../../logs/yoruba_newsclass_base-zeroshot_bs128" #CHANGE_ME
    }
    run_experiment(config)

def main_yoruba_base_fulldev_exp():
    config = {
        "trainer": "base",
        "model_name": "bert-base-multilingual-cased",
        "data_dir": "../../data/yoruba_newsclass/", #CHANGE_ME
        "labels": ["nigeria", "africa", "world", "entertainment", "health", "sport", "politics"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": True
    }

    for clean_subset_size in [10, 100, 300, 500, 700, 900, 1100, -1]: # -1 is 1340
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_newsclass_base_full-dev_bs128_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_base_and_noise_exp():
    config = {
        "trainer": "base",
        "model_name": "bert-base-multilingual-cased",
        "data_dir": "../../data/yoruba_newsclass/", #CHANGE_ME
        "labels": ["nigeria", "africa", "world", "entertainment", "health", "sport", "politics"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": True,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": True
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10,  14, 42, 71, 99, 127, 189]): # -1 is 1340
        if clean_subset_size == -1:
            noisy_subsample_size = 968
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_newsclass_base+noise_bs128_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_cmmodel_smoothed():
    config = {
        "trainer": "cmmodel",
        "model_name": "bert-base-multilingual-cased",
        "data_dir": "../../data/yoruba_newsclass/", #CHANGE_ME
        "labels": ["nigeria", "africa", "world", "entertainment", "health", "sport", "politics"],
        "max_seq_length": 100,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": True,
        "use_true_cm": False,
        "use_io_estimation": False,
        "matrix_smoothing_value": 0.8
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10,  14, 42, 71, 99, 127, 189]): # -1 is 1340
        if clean_subset_size == -1:
            noisy_subsample_size = 968
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_newsclass_cm-smoothed_bs128_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_cmmodel():
    config = {
        "trainer": "cmmodel",
        "model_name": "bert-base-multilingual-cased",
        "data_dir": "../../data/yoruba_newsclass/", #CHANGE_ME
        "labels": ["nigeria", "africa", "world", "entertainment", "health", "sport", "politics"],
        "max_seq_length": 100,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": True,
        "use_true_cm": False,
        "use_io_estimation": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10,  14, 42, 71, 99, 127, 189]): # -1 is 1340
        if clean_subset_size == -1:
            noisy_subsample_size = 968
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_newsclass_cm_bs128_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_yoruba_base_distilled_exp():
    config = {
        "trainer": "base",
        "model_name": "distilbert-base-multilingual-cased",
        "data_dir": "../../data/yoruba_newsclass/", #CHANGE_ME
        "labels": ["nigeria", "africa", "world", "entertainment", "health", "sport", "politics"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10,  14, 42, 71, 99, 127, 189]): # -1 is 1340
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/yoruba_newsclass_base-distilled_bs128_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_hausa_base_exp():
    config = {
        "trainer": "base",
        "model_name": "xlm-roberta-base",
        "data_dir": "../../data/hausa_newsclass/", #CHANGE_ME
        "labels": ["Nigeria", "Politics", "Africa", "Health", "World"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10, 14, 43, 71, 99, 128, 290]): # -1 is 2045
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_newsclass_base_bs128_subset-size{clean_subset_size}_seed{seed}" #CHANGE_ME
            run_experiment(config)

def main_hausa_base_zeroshot_exp():
    config = {
        "trained_model_directory": "../../logs/ag-news_newsclass_base_for-hausa_bs128", #CHANGE_ME # Needs to be pretrained first with main_agnews_for_hausa_exp()!
        "trainer": "eval",
        "seed": 12345,
        "eval_data_dir": "../../data/hausa_newsclass/", #CHANGE_ME
        "eval_file": "test.tsv",
        "output_dir": f"../../logs/hausa_newslcass_base-zeroshot_bs128" #CHANGE_ME
    }
    run_experiment(config)

def main_hausa_base_fewshot_exp():
    config = {
        "trainer": "base",
        "model_name": "../../logs/ag-news_newsclass_base_for-hausa_bs128/checkpoint-best-on-dev", #CHANGE_ME # Needs to be pretrained first with main_agnews_for_hausa_exp()!
        "data_dir": "../../data/hausa_newsclass/",  #CHANGE_ME
        "labels": ["Nigeria", "Politics", "Africa", "Health", "World", "Sports", "Business", "Sci/Tech"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10, 14, 43, 71, 99, 128, 290]): # -1 is 2045
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_newsclass_base_few-shot_bs128_subset-size{clean_subset_size}_seed{seed}"  #CHANGE_ME
            run_experiment(config)


def main_hausa_base_fulldev_exp():
    config = {
        "trainer": "base",
        "model_name": "xlm-roberta-base",
        "data_dir": "../../data/hausa_newsclass/",  #CHANGE_ME
        "labels": ["Nigeria", "Politics", "Africa", "Health", "World"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": False
    }

    for clean_subset_size in [10, 100, 300, 500, 700, 900, 1100, 1500, -1]: # -1 is 2045 
        for seed in range(12345, 12355):
            config["clean_subset_size"] = clean_subset_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_newsclass_base_bs128_full-dev_subset-size{clean_subset_size}_seed{seed}"  #CHANGE_ME
            run_experiment(config)

def main_hausa_base_and_noise_exp():
    config = {
        "trainer": "base",
        "model_name": "xlm-roberta-base",
        "data_dir": "../../data/hausa_newsclass/",  #CHANGE_ME
        "labels": ["Nigeria", "Politics", "Africa", "Health", "World"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": True,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10, 14, 43, 71, 99, 128, 290]): # -1 is 2045
        if clean_subset_size == -1:
            noisy_subsample_size = 968
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12365):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_newsclass_base+noise_bs128_subset-size{clean_subset_size}_seed{seed}"  #CHANGE_ME
            run_experiment(config)

def main_hausa_cmmodel_exp():
    config = {
        "trainer": "cmmodel",
        "model_name": "xlm-roberta-base",
        "data_dir": "../../data/hausa_newsclass/",  #CHANGE_ME
        "labels": ["Nigeria", "Politics", "Africa", "Health", "World"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": True,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": False,
        "use_true_cm": False,
        "use_io_estimation": False
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10, 14, 43, 71, 99, 128, 290]): # -1 is 2045
        if clean_subset_size == -1:
            noisy_subsample_size = 968
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12365):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_newsclass_cmmodel_bs128_subset-size{clean_subset_size}_seed{seed}"  #CHANGE_ME
            run_experiment(config)

def main_hausa_cmmodel_smoothed_exp():
    config = {
        "trainer": "cmmodel",
        "model_name": "xlm-roberta-base",
        "data_dir": "../../data/hausa_newsclass/",  #CHANGE_ME
        "labels": ["Nigeria", "Politics", "Africa", "Health", "World"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": True,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 50,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": False,
        "use_true_cm": False,
        "use_io_estimation": False,
        "matrix_smoothing_value": 0.8
    }

    for clean_subset_size, limit_devset_size in zip([10, 100, 300, 500, 700, 900, -1], [10, 14, 43, 71, 99, 128, 290]): # -1 is 2045
        if clean_subset_size == -1:
            noisy_subsample_size = 968
        else:
            noisy_subsample_size = clean_subset_size
        for seed in range(12345, 12365):
            config["clean_subset_size"] = clean_subset_size
            config["limit_devset_size"] = limit_devset_size
            config["noisy_subsample_size"] = noisy_subsample_size
            config["seed"] = seed
            config["output_dir"] = f"../../logs/hausa_newsclass_cmmodel_smoothed_bs128_subset-size{clean_subset_size}_seed{seed}"  #CHANGE_ME
            run_experiment(config)

def main_agnews_for_yoruba_exp():
    config = {
        "trainer": "base",
        "model_name": "bert-base-multilingual-cased",
        "data_dir": "../../data/ag-news/AG/ag-for-yoruba/",  #CHANGE_ME
        "labels": ["nigeria", "africa", "world", "entertainment", "health", "sport", "politics", "business", "sci/tech"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 20,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": True,
        "seed": 12345,
        "output_dir": "../../logs/ag-news_newsclass_base_for-yoruba_bs128"  #CHANGE_ME
    }

    run_experiment(config)


def main_agnews_for_hausa_exp():
    config = {
        "trainer": "base",
        "model_name": "xlm-roberta-base",
        "data_dir": "../../data/ag-news/AG/ag-for-hausa/",  #CHANGE_ME
        "labels": ["Nigeria", "Politics", "Africa", "Health", "World", "Sports", "Business", "Sci/Tech"],
        "max_seq_length": 100,
        "use_clean": True,
        "use_noisy": False,
        "batch_size": 128,
        "weight_decay": 0.0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1,
        "warmup_steps": 0,
        "num_epochs": 20,
        "device": "cuda",
        "random_subset": False,
        "model_with_token_types": False,
        "seed": 12345,
        "output_dir": "../../logs/ag-news_newsclass_base_for-hausa_bs128"  #CHANGE_ME
    }
    run_experiment(config)


if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(logging.StreamHandler(sys.stdout))

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    #CHANGE_ME
    # Call here the method for the model, you want to train, e.g.
    main_yoruba_base_exp()

