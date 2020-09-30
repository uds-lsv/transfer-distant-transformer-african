import os
import logging
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .trainer import Trainer
from models import create_optimizer, create_base_model, compute_loss
from utils_ner import load_dataset, batch_to_inputs, read_sentences_from_file

logger = logging.getLogger(__name__)

class EvalTrainer(Trainer):
    """
    Loads a pretrained model and runs an evaluation.
    """

    def __init__(self, exp_config):
        self._load_config(exp_config)
        model = create_base_model(exp_config)
        super().__init__(model, exp_config)

    def _load_config(self, exp_config):
        orig_config = json.load(open(os.path.join(exp_config["trained_model_directory"], "exp_config.json"), "r"))
        # copy original configuration into new one
        # overwrite in place to work with the rest of the structure
        exp_config["eval_output_dir"] = exp_config["output_dir"]
        # TODO Only update if not already exists
        exp_config.update(orig_config)
        #we have to overwrite the output dir
        exp_config["orig_output_dir"] = exp_config["output_dir"]
        exp_config["output_dir"] = exp_config["eval_output_dir"]
        # we have to overwrite the model name to load the trained model and not the original
        exp_config["orig_model_name"] = exp_config["model_name"]
        exp_config["model_name"] = os.path.join(exp_config["trained_model_directory"], "checkpoint-best-on-dev")
        # we need to override data dir
        exp_config["orig_data_dir"] = exp_config["data_dir"]
        exp_config["data_dir"] = exp_config["eval_data_dir"]

    def train(self, exp_config):
        # count number of parameters
        num_parameters = np.sum([p.numel() for p in self.main_model.parameters()])
        logger.info(f"The model has {num_parameters} parameters.")

        # no actual training as we have a pretrained model
        # test in the end
        eval_prefix = f"eval_{exp_config['eval_file']}"
        eval_report, f1 = self.eval(exp_config["eval_file"], exp_config,
                                    output_prediction_file=os.path.join(exp_config["output_dir"],
                                                                       eval_prefix + ".tsv"))
        logger.info(f"F1-score on evaluation file {exp_config['eval_file']}: {f1}")
        Trainer._save_eval_report(eval_prefix, eval_report, f1, exp_config)





