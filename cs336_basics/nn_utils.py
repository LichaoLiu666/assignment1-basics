import torch
import torch.nn as nn
from einops import einsum
import numpy as np


class cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):

        m = inputs.max(dim=-1, keepdim=True).values
        normalized_input = inputs - m

        lse = torch.logsumexp(normalized_input, dim=-1)

        idx = targets.unsqueeze(-1)
        correct = normalized_input.gather(-1, idx).squeeze(-1)
        return (lse - correct).mean()


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    # print(parameters)
    l2_norm = torch.cat([parameter.grad.view(-1) for parameter in parameters if parameter.grad is not None]).norm()
    if l2_norm >= max_l2_norm:
        clip_coef = max_l2_norm / (l2_norm + eps)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * clip_coef


def get_batch(dataset, batch_size, context_length, device):
    max_batch_indices = len(dataset) - context_length - 1
    batch_sample_indices = np.random.randint(0, max_batch_indices + 1, size=batch_size)
    context_indices = np.arange(0, context_length)

    input_batch_indices = batch_sample_indices.reshape(len(batch_sample_indices), 1) + context_indices.reshape(
        1, len(context_indices)
    )

    input_seq = torch.tensor(dataset[input_batch_indices].astype(np.int64), device=device)
    target_seq = torch.tensor(dataset[input_batch_indices + 1].astype(np.int64), device=device)

    return input_seq, target_seq


def save_checkpoint(model, optimizer, iteration, out, best_val=None):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    out_dict = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "iteration": iteration,
    }
    if best_val is not None:
        out_dict["best_val"] = best_val
    torch.save(out_dict, out)


def load_checkpoint(src, model, optimizer=None, map_location=None):
    checkpoint = torch.load(src, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    # inference-only callers (e.g. decode) have no optimizer to restore
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
