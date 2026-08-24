import argparse
import os

import torch


def top_p_sampling(probs, top_p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    # zero out everything after the cumulative prob first exceeds top_p,
    # always keeping at least the highest-probability token
    cutoff_mask = (cumsum - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum()

    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(0, sorted_indices, sorted_probs)
    return new_probs


def decode(prompt_ids, model, max_sequence_length, temperature, top_p, end_token_id):
    model.eval()
    device = next(model.parameters()).device
    context_length = getattr(model, "context_length", None)
    out = prompt_ids.to(device)

    for _ in range(max_sequence_length):
        if context_length and out.numel() > context_length:
            effective_out = out[-context_length:]
        else:
            effective_out = out

        logits = model(effective_out.unsqueeze(0))
        next_logits = logits[0, -1, :]
        if temperature == 0:
            next_id = int(torch.argmax(next_logits))
        else:
            scaled = next_logits / float(temperature)
            probs = torch.softmax(scaled, dim=0)

            if top_p < 1:
                probs = top_p_sampling(probs, top_p)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        out = torch.cat([out, torch.tensor([next_id], device=device, dtype=torch.long)], dim=0)

        if next_id == int(end_token_id):
            break

    return out


def torch_dtype_from_string(s):
    if s == "fp32":
        return torch.float32
    elif s == "fp16":
        return torch.float16
    elif s == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported model dtype")


if __name__ == "__main__":
    from cs336_basics.modules import transformer_lm
    from cs336_basics.tokenizer import tokenizer
    from cs336_basics.config import getConfig
    from cs336_basics.nn_utils import load_checkpoint

    EOT = "<|endoftext|>"
    tok = tokenizer.from_files(
        "/home/lichao-liu/assignment1-basics/cs336_basics/output/tinystories_vocabs.pkl",
        "/home/lichao-liu/assignment1-basics/cs336_basics/output/tinystories_merges.pkl",
        special_tokens=[EOT],
    )

    end_token_id = tok.encode(EOT)[0]
    print(end_token_id)

    cfg = getConfig()
    device = torch.device(cfg.data.device)

    model_dtype = torch_dtype_from_string(cfg.model.torch_dtype)
    d_ff = cfg.model.d_ff if cfg.model.d_ff else cfg.model.d_model * 8 / 3

    model = transformer_lm(
        cfg.model.vocab_size,
        cfg.model.context_length,
        cfg.model.d_model,
        cfg.model.num_layers,
        cfg.model.num_heads,
        d_ff,
        cfg.model.max_seq_len,
        cfg.model.rope_theta,
        device=device,
        dtype=model_dtype,
    )
    load_checkpoint(
        "/home/lichao-liu/assignment1-basics/runs/baseline-batch-512_20260821_050405/checkpoints/best_checkpoint.pt",
        model,
    )
    prompt = "Once upon a time,"
    prompt_ids = torch.tensor(tok.encode(prompt))
    out_ids = decode(prompt_ids, model, max_sequence_length=256, temperature=0, top_p=0.7, end_token_id=end_token_id)

    print(tok.decode(out_ids.tolist()))
