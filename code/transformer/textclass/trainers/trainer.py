from abc import ABC
import os
import logging
import json
import time

import numpy as np
import sklearn.metrics
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_constant_schedule_with_warmup
from tqdm import tqdm

from utils_textclass import load_dataset, batch_to_inputs, read_instances_from_file

logger = logging.getLogger(__name__)


class Trainer(ABC):

    def __init__(self, model, exp_config):
        self.main_model = model
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(exp_config["model_name"])
        self.best_f1 = -1
        Trainer.save_config(exp_config)

    @staticmethod
    def save_config(exp_config):
        exp_config["timestamp"] = time.strftime("%Y-%m-%d %H:%M")
        with open(os.path.join(exp_config["output_dir"], "exp_config.json"), "w") as out_file:
            json.dump(exp_config, out_file, sort_keys=True, indent=4)

    def _load_train_clean(self, exp_config):
        # load full dataset
        # needed, even if we use a subset, to obtain the length
        instances = read_instances_from_file(exp_config["data_dir"], "train_clean.tsv")
        subset_indices = range(len(instances))
        if "clean_subset_size" in exp_config and exp_config["clean_subset_size"] != -1:
            assert exp_config["clean_subset_size"] <= len(subset_indices)

            if exp_config["random_subset"]:
                subset_indices = np.random.RandomState(exp_config["seed"]).choice(subset_indices,
                                                                                  size=exp_config["clean_subset_size"],
                                                                                  replace=False)
            else:
                subset_indices = range(0, exp_config["clean_subset_size"])
        train_clean = load_dataset("train_clean.tsv", self.tokenizer, exp_config, subset_indices)

        dataloader_train_clean = DataLoader(train_clean, batch_size=exp_config["batch_size"], shuffle=False,
                                            num_workers=1)
        logging.info(f"Loaded {len(train_clean)} clean train instances.")
        return dataloader_train_clean, subset_indices

    def _load_train_noisy(self, exp_config):
        train_noisy = load_dataset("train_noisy.tsv", self.tokenizer, exp_config)
        return DataLoader(train_noisy, batch_size=exp_config["batch_size"], shuffle=True,
                                            num_workers=1)

    def eval(self, filename, exp_config, subset_indices=None):
        data_eval = load_dataset(filename, self.tokenizer, exp_config, subset_indices=subset_indices)
        dataloader_eval = DataLoader(data_eval, batch_size=exp_config["batch_size"], shuffle=False, num_workers=1)
        logger.info(f"Evaluating on {len(data_eval)} instances from {filename}.")

        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        true_label_ids = []
        self.main_model.eval()
        for batch in tqdm(dataloader_eval, desc="Evaluating"):
            with torch.no_grad():
                inputs, labels = batch_to_inputs(batch, exp_config)
                outputs = self.main_model(**inputs)
                logits = outputs[0]

            preds.extend([np.argmax(pred_instance) for pred_instance in logits.detach().cpu().numpy()])
            true_label_ids.extend(labels.detach().cpu().numpy()) # TODO should not be necessary to get it back from the CPU
            nb_eval_steps += 1

        eval_report = sklearn.metrics.classification_report(true_label_ids, preds,
                                                            labels=range(len(exp_config["labels"])),
                                                            target_names=exp_config["labels"],
                                                            output_dict=True
                                                            )
        f1 = eval_report["weighted avg"]["f1-score"]
        return eval_report, f1

    @staticmethod
    def _save_eval_report(eval_name, eval_report, f1, exp_config):
        with open(os.path.join(exp_config["output_dir"], f"{eval_name}_report.json"), "w") as out_file:
            json.dump(eval_report, out_file, sort_keys=True, indent=4)
        with open(os.path.join(exp_config["output_dir"], f"{eval_name}_f1.txt"), "w") as out_file:
            out_file.write(str(f1))
            out_file.write("\n")

    def _dev(self, exp_config, epoch):
        if "limit_devset_size" in exp_config:
            subset_indices = range(exp_config["limit_devset_size"])
            logger.info(f"For dev set using subset with indices {subset_indices}.")
        else:
            subset_indices = None

        eval_report, f1 = self.eval("dev.tsv", exp_config, subset_indices=subset_indices)

        if f1 > self.best_f1:
            logger.info(f"Saving model in epoch {epoch} because new f1 score {f1} > old f1 score {self.best_f1}")
            Trainer.save_checkpoint(self.main_model, self.tokenizer, Trainer.get_best_checkpoint_path(exp_config))
            eval_report["epoch"] = epoch
            Trainer._save_eval_report("best-dev", eval_report, f1, exp_config)
            self.best_f1 = f1
        else:
            logger.info(f"F1 score of {f1} in {epoch} < then best f1 score {self.best_f1}")

    def _test(self, exp_config):
        self.load_model(Trainer.get_best_checkpoint_path(exp_config), exp_config["device"])
        eval_report, f1 = self.eval("test.tsv", exp_config)
        logger.info(f"F1-score on test: {f1}")
        Trainer._save_eval_report("test", eval_report, f1, exp_config)

    def load_model(self, model_dir_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
        self.main_model = AutoModelForSequenceClassification.from_pretrained(model_dir_path)
        self.main_model.to(device)

    @staticmethod
    def save_checkpoint(model, tokenizer, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    @staticmethod
    def get_best_checkpoint_path(exp_config):
        return os.path.join(exp_config["output_dir"], f"checkpoint-best-on-dev")


