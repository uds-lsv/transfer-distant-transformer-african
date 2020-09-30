import torch
import numpy as np
import logging
import itertools
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .trainer import Trainer
from utils_textclass import create_label_map, read_instances_from_file, batch_to_inputs, load_dataset
from models import create_base_model, create_cm_model, create_optimizer, compute_loss

logger = logging.getLogger(__name__)

class ConfusionMatrixTrainer(Trainer):

    def __init__(self, exp_config):
        main_model = create_base_model(exp_config)
        super().__init__(main_model, exp_config)

    def train(self, exp_config):
        # load clean dataset
        dataloader_train_clean, clean_indices = self._load_train_clean(exp_config)

        # load noisy dataset
        self.noisy_subsample_random = np.random.RandomState(exp_config["seed"])
        self.noisy_dataset_full_size = len(read_instances_from_file(exp_config["data_dir"], "train_noisy.tsv"))

        # creating noise matrix from pairs of clean and noisy labels
        # making sure that the noise matrix is build on the same subset indices as the actual training data
        clean_instances = read_instances_from_file(exp_config["data_dir"], "train_clean.tsv")
        clean_noisy_labels_instances = read_instances_from_file(exp_config["data_dir"], "train_clean_noisy_labels.tsv")

        if not "use_true_cm" in exp_config or not exp_config["use_true_cm"]:
            if clean_indices is None:
                logging.info("Would be using the clean subsample, but the full clean data is used for the training.")
            else:
                logging.info("Using the clean subsample to estimate the noise matrix.")
                clean_sentences = np.array(clean_instances)[clean_indices]
                clean_noisy_labels_sentences = np.array(clean_noisy_labels_instances)[clean_indices]
        else:
            logging.info("Using the full clean data to estimate the noise matrix.")

        if not "use_io_estimation" in exp_config or not exp_config["use_io_estimation"]:
            logging.info("Using BIO format for noise matrix estimation.")
            noise_matrix = ConfusionMatrixTrainer.compute_noise_matrix(clean_instances, clean_noisy_labels_instances,
                                                                   exp_config["labels"])
        else:
            logging.info("Using IO format for noise matrix estimation.")
            noise_matrix = ConfusionMatrixTrainer.compute_noise_matrix_io(clean_instances, clean_noisy_labels_instances,
                                                                   exp_config["labels"])

        logger.info(f"Using noise matrix {repr(noise_matrix)}")

        # matrix smoothing following https://arxiv.org/pdf/2003.11904.pdf
        if "matrix_smoothing_value" in exp_config:
            logger.info(f"Using matrix smoothing with value {exp_config['matrix_smoothing_value']}.")
            smoothing_beta = exp_config["matrix_smoothing_value"] # 0.8 is their chosen hyperparameter in the paper, section 4.2
            noise_matrix = noise_matrix**smoothing_beta

            # renormalize
            for row in noise_matrix:
                row_sum = np.sum(row)
                if row_sum != 0:
                    row /= row_sum

        self.cm_model = create_cm_model(self.main_model, noise_matrix, exp_config)

        optimizer, scheduler = create_optimizer(self.cm_model, exp_config)

        self.cm_model.zero_grad()

        # first epoch only clean
        self.train_one_epoch(dataloader_train_clean, optimizer, scheduler, exp_config, "clean")

        epoch_iterator = trange(0, int(exp_config["num_epochs"] - 1), desc="Epoch")
        for epoch in epoch_iterator:
            dataloader_train_noisy = self.load_train_noisy_subsample(exp_config)
            self.train_one_epoch(dataloader_train_noisy, optimizer, scheduler, exp_config, "noisy")

            self.train_one_epoch(dataloader_train_clean, optimizer, scheduler, exp_config, "clean")

            # after each epoch, evaluate on dev and save if better than previous
            self._dev(exp_config, epoch)

        # test in the end
        self._test(exp_config)

    @staticmethod
    def compute_noise_matrix(instances_clean, instances_noisy, labels):
        assert len(instances_clean) == len(instances_noisy)
        labels_clean = [instance.label for instance in instances_clean]
        labels_noisy = [instance.label for instance in instances_noisy]
        assert len(labels_clean) == len(labels_noisy)

        num_labels = len(labels)
        label_map = create_label_map(labels)
        noise_matrix = np.zeros((num_labels, num_labels))

        for label_clean, label_noisy in zip(labels_clean, labels_noisy):
            label_clean = label_map[label_clean]
            label_noisy = label_map[label_noisy]
            noise_matrix[label_clean][label_noisy] += 1

        for row in noise_matrix:
            row_sum = np.sum(row)
            if row_sum != 0:
                row /= row_sum

        return noise_matrix

    @staticmethod
    def compute_noise_matrix_io(instances_clean, instances_noisy, labels):
        # convert to IO format
        assert len(sentences_clean) == len(sentences_clean)
        labels_clean = [instance.label for instance in instances_clean]
        labels_clean = [ConfusionMatrixTrainer._remove_ib_prefix(label_clean) for label_clean in labels_clean]
        labels_noisy = [instance.label for instance in instances_noisy]
        labels_noisy = [ConfusionMatrixTrainer._remove_ib_prefix(label_noisy) for label_noisy in labels_noisy]
        assert len(labels_clean) == len(labels_noisy)

        # Noise matrix for IO
        io_labels = [ConfusionMatrixTrainer._remove_ib_prefix(label) for label in labels if (label.startswith("B-") or label == "O")]
        io_label_map = {v:k for k,v in enumerate(io_labels)}
        num_io_labels = len(io_labels)
        io_noise_matrix = np.zeros((num_io_labels, num_io_labels))
        for label_clean, label_noisy in zip(labels_clean, labels_noisy):
            label_clean = io_label_map[label_clean]
            label_noisy = io_label_map[label_noisy]
            io_noise_matrix[label_clean][label_noisy] += 1

        for row in io_noise_matrix:
            row_sum = np.sum(row)
            if row_sum != 0:
                row /= row_sum

        # Upscale to BIO format
        num_labels = len(labels)
        label_map = create_label_map(labels)
        noise_matrix = np.zeros((num_labels, num_labels))
        for label_i in labels:
            io_label_i = ConfusionMatrixTrainer._remove_ib_prefix(label_i)
            for label_j in labels:
                io_label_j = ConfusionMatrixTrainer._remove_ib_prefix(label_j)
                noise_matrix[label_map[label_i]][label_map[label_j]] = io_noise_matrix[io_label_map[io_label_i]][io_label_map[io_label_j]]

        return noise_matrix

    @staticmethod
    def _remove_ib_prefix(label):
        if label == "O":
            return label
        if label.startswith("B-") or label.startswith("I-"):
            return label[2:]
        raise Exception(f"Label {label} is not in BIO format.")


    def train_one_epoch(self, dataloader, optimizer, scheduler, exp_config, type):
        model = None
        if type == "clean":
            model = self.main_model
        elif type == "noisy":
            model = self.cm_model

        batch_iterator = tqdm(dataloader, desc=f"Step {type}")
        for step, batch in enumerate(batch_iterator):
            model.train()

            inputs, labels = batch_to_inputs(batch, exp_config)

            if type == "clean":
                logits = model(**inputs)[0]
                loss = compute_loss(logits, labels, is_log_prob=False)
            elif type == "noisy":
                log_probs = model(inputs)
                loss = compute_loss(log_probs, labels, is_log_prob=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config["max_grad_norm"])

            optimizer.step()
            scheduler.step()
            model.zero_grad()


    def load_train_noisy_subsample(self, exp_config):
        subset_indices = self.noisy_subsample_random.choice(range(self.noisy_dataset_full_size),
                                                            exp_config["noisy_subsample_size"],
                                                            replace=False)
        train_noisy_subset = load_dataset("train_noisy.tsv", self.tokenizer, exp_config, subset_indices)
        return DataLoader(train_noisy_subset, batch_size=exp_config["batch_size"], shuffle=True, num_workers=1)