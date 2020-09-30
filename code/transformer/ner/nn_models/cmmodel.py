import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfusionMatrixModel(nn.Module):
    def __init__(self, base_model, channel_weights, exp_config):
        """
        :param base_model: A
        :param channel_weights:
        :param exp_config:
        """
        super().__init__()
        self.base_model = base_model
        self.noise_mat = torch.tensor(channel_weights, requires_grad=True).float().to(exp_config["device"])

    def forward(self, inputs):
        base_model_output = self.base_model(**inputs)
        clean_logits = base_model_output[0]
        out = F.softmax(clean_logits, dim=-1)
        out = torch.matmul(out, F.softmax(self.noise_mat, dim=1))
        # output is already a probability distribution, so we just apply the log to obtain
        # log(softmax) for NLLLoss
        out = torch.log(out)
        return out
