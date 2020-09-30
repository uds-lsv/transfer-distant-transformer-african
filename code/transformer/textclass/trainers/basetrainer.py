import os
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np

from .trainer import Trainer
from models import create_base_model, create_optimizer, compute_loss
from utils_textclass import batch_to_inputs, read_instances_from_file, load_dataset

logger = logging.getLogger(__name__)


class BaseTrainer(Trainer):

    def __init__(self, exp_config):
        model = create_base_model(exp_config)
        super().__init__(model, exp_config)

    def train(self, exp_config):
        assert exp_config["use_clean"]
        dataloader_train_clean, _ = self._load_train_clean(exp_config)

        if exp_config["use_noisy"]:
            self.noisy_subsample_random = np.random.RandomState(exp_config["seed"])
            self.noisy_dataset_full_size = len(read_instances_from_file(exp_config["data_dir"], "train_noisy.tsv"))

        optimizer, scheduler = create_optimizer(self.main_model, exp_config)

        self.main_model.zero_grad()

        # first epoch only clean
        self.train_one_epoch(dataloader_train_clean, optimizer, scheduler, exp_config, "clean - 1. epoch")

        epoch_iterator = trange(0, int(exp_config["num_epochs"]-1), desc="Epoch")
        for epoch in epoch_iterator:
            if exp_config["use_noisy"]:
                dataloader_train_noisy = self.load_train_noisy_subsample(exp_config)
                self.train_one_epoch(dataloader_train_noisy, optimizer, scheduler, exp_config, "noisy")

            self.train_one_epoch(dataloader_train_clean, optimizer, scheduler, exp_config, "clean")

            # after each epoch, evaluate on dev and save if better than previous
            self._dev(exp_config, epoch)

        # test in the end
        self._test(exp_config)

    def load_train_noisy_subsample(self, exp_config):
        subset_indices = self.noisy_subsample_random.choice(range(self.noisy_dataset_full_size),
                                                            exp_config["noisy_subsample_size"],
                                                            replace=False)
        train_noisy_subset = load_dataset("train_noisy.tsv", self.tokenizer, exp_config, subset_indices)
        return DataLoader(train_noisy_subset, batch_size=exp_config["batch_size"], shuffle=True, num_workers=1)

    def train_one_epoch(self, dataloader, optimizer, scheduler, exp_config, type):
        self.main_model.train()
        batch_iterator = tqdm(dataloader, desc=f"Step {type}")
        for step, batch in enumerate(batch_iterator):

            inputs, labels = batch_to_inputs(batch, exp_config)
            clean_outputs = self.main_model(**inputs)
            logits = clean_outputs[0] # if we do not provide labels to the BERT model, the first output is the logits
            loss = compute_loss(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), exp_config["max_grad_norm"])

            optimizer.step()
            scheduler.step()
            self.main_model.zero_grad()


