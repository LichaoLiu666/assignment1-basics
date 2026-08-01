import torch
import torch.nn as nn
from einops import einsum


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
    print(l2_norm)
    if l2_norm >= max_l2_norm:
        clip_coef = max_l2_norm / (l2_norm + eps)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * clip_coef
