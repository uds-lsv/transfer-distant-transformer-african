import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .trainer import Trainer
from models import create_optimizer, create_base_model, compute_loss
from utils_ner import load_dataset, batch_to_inputs, read_sentences_from_file

logger = logging.getLogger(__name__)

class BaseTrainer(Trainer):

    def __init__(self, exp_config):
        model = create_base_model(exp_config)
        super().__init__(model, exp_config)

        num_parameters = np.sum([p.numel() for p in self.main_model.parameters()])
        logger.info(f"The model has {num_parameters} parameters.")
        raise Exception("Done")

    def train(self, exp_config):
        # making sure that all nn_models are trainable
        for name, param in self.main_model.named_parameters():
            if not param.requires_grad:
                raise Exception(f"Frozen:  {name}, {param.data}")

        dataloader_train_clean, _ = self._load_train_clean(exp_config)

        if exp_config["use_noisy"]:
            self.noisy_subsample_random = np.random.RandomState(exp_config["seed"])
            self.noisy_dataset_full_size = len(read_sentences_from_file(exp_config["data_dir"], "train_noisy.tsv"))

        optimizer, scheduler = create_optimizer(self.main_model, exp_config)

        self.main_model.zero_grad()

        # first epoch only clean
        self.train_one_epoch(dataloader_train_clean, optimizer, scheduler, exp_config, "clean")

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
            loss = compute_loss(logits, inputs["attention_mask"], labels, len(exp_config["labels"]))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), exp_config["max_grad_norm"])

            optimizer.step()
            scheduler.step()
            self.main_model.zero_grad()





