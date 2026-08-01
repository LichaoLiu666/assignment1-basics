# import torch
# from collections.abc import Callable, Iterable
# from typing import Optional


# class adamw(torch.optim.Optimizer):
#     def __init__(self, params, lr, betas, eps, weight_decay):
#         if lr < 0:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
#         super().__init__(params, defaults)

#     @torch.no_grad()
#     def step(self, closure: Optional[Callable] = None):
#         loss = None if closure is None else closure()
#         for group in self.param_groups:
#             lr = group["lr"]  # Get the learning rate.
#             beta1, beta2 = group["betas"]
#             eps = group["eps"]
#             weight_decay = group["weight_decay"]
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 state = self.state[p]  # Get state associated with p.
#                 t = state.get("t", 1)  # Get iteration number from the state, or 0.
#                 m = state.get("m", torch.zeros_like(p))
#                 v = state.get("v", torch.zeros_like(p))
#                 grad = p.grad.data  # Get the gradient of loss with respect to p.

#                 lr_t = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
#                 p.data -= lr * weight_decay * p.data  # Update weight tensor in-place.

#                 m = beta1 * m + (1 - beta1) * grad
#                 v = beta2 * v + (1 - beta2) * grad**2

#                 p.data -= lr_t * m / (v**0.5 + eps)

#                 state["m"] = m
#                 state["v"] = v
#                 state["t"] = t + 1  # Increment iteration number.
#         return loss


from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, eps, betas):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "eps": eps, "betas": betas}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                lr_t = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                p.data -= lr_t * m / (v**0.5 + eps)

                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss


def lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        lr = min_learning_rate + 1 / 2 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)

    else:
        lr = min_learning_rate

    return lr
