from cs336_basics.config import getConfig
from cs336_basics.experiment_tracking import experimentTracker
from cs336_basics.modules import transformer_lm
from cs336_basics.optimizer import adamw, lr_cosine_schedule
from cs336_basics.nn_utils import save_checkpoint, load_checkpoint, get_batch, cross_entropy, gradient_clipping
import time
import os
import numpy as np
import torch
import math


def open_memmap_1d(path, np_dtype):
    dtype = np.dtype(np_dtype)
    itemsize = dtype.itemsize
    nbtypes = os.path.getsize(path)

    if nbtypes % itemsize != 0:
        raise ValueError(f"file size is not divisible by dtype size: {path} {dtype} btypes,itemsize= {itemsize}")

    length = nbtypes // itemsize
    return np.memmap(path, mode="r", dtype=dtype, shape=(length,))


def torch_dtype_from_string(s):
    if s == "fp32":
        return torch.float32
    elif s == "fp16":
        return torch.float16
    elif s == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported model dtype")


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def main():
    cfg = getConfig()

    if cfg.run.run_name is not None:
        run_name = f"{cfg.run.run_name_prefix}_{cfg.run.run_name}"
    else:
        run_name = f"{cfg.run.run_name_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

    run_dir = os.path.join(cfg.run.run_dirs, run_name)

    tracker = experimentTracker(run_dir, cfg)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")

    train_mm = open_memmap_1d(cfg.data.train_data_path, cfg.data.np_dtype)
    val_mm = open_memmap_1d(cfg.data.val_data_path, cfg.data.np_dtype)

    device = torch.device(cfg.data.device)
    model_dtype = torch_dtype_from_string(cfg.model.torch_dtype)

    d_ff = cfg.model.d_ff if cfg.model.d_ff else int(cfg.model.d_model * 8 / 3)

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
    ).to(device)

    optimizer = adamw(
        model.parameters(),
        cfg.optimizer.lr,
        cfg.optimizer.weight_decay,
        cfg.optimizer.eps,
        (cfg.optimizer.beta1, cfg.optimizer.beta2),
    )

    criterion = cross_entropy()

    start_it = 0
    best_val = float("inf")
    resume_path = cfg.run.resume_path
    if resume_path is None:
        resume_path = checkpoint_path
    if resume_path is not None and os.path.exists(resume_path):
        start_it = load_checkpoint(resume_path, model, optimizer, map_location=device)
        best_val = torch.load(resume_path, map_location=device).get("best_val", float("inf"))
        print(f"[resume] loade checkpoints: {resume_path} start_it={start_it} best_val={best_val}")

    last_log_at = time.time()
    last_log_it = start_it

    for it in range(start_it, cfg.run.max_iterations):
        lr = lr_cosine_schedule(
            it, cfg.optimizer.max_lr, cfg.optimizer.min_lr, cfg.optimizer.warmup_iters, cfg.optimizer.cosine_cycle_iters
        )
        set_optimizer_lr(optimizer, lr)

        xb, yb = get_batch(train_mm, cfg.run.batch_size, cfg.model.context_length, cfg.data.device)
        # forward
        logits = model(xb)
        B, S, V = logits.shape
        loss = criterion(logits.reshape(B * S, V), yb.reshape(B * S))

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.optimizer.grad_clip > 0:
            gradient_clipping(model.parameters(), cfg.optimizer.grad_clip)
        optimizer.step()

        if (it + 1) % cfg.run.log_interval == 0:
            now = time.time()
            dt = max(now - last_log_at, 1e-9)
            iters_since_log = it + 1 - last_log_it
            tok_s = (cfg.run.batch_size * cfg.model.context_length * iters_since_log) / dt
            msg = f"it={it + 1} loss={loss.item():.4f} tok/s={tok_s:.1f}"
            print(msg)
            tracker.log(step=it + 1, metrics={"train loss": float(loss.item()), "train_tok_s": float(tok_s)})
            last_log_at = now
            last_log_it = it + 1
        if (it + 1) % cfg.run.eval_interval == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(cfg.run.eval_batches):
                    xb, yb = get_batch(val_mm, cfg.run.batch_size, cfg.model.context_length, device=cfg.data.device)
                    logits = model(xb)
                    B, S, V = logits.shape
                    loss = criterion(logits.reshape(B * S, V), yb.reshape(B * S))
                    losses.append(loss.item())
            model.train()
            val_loss = float(np.mean(losses))
            val_ppl = float(math.exp(val_loss))
            print(f"[eval] it = {it + 1} val_loss={val_loss:.4f} val_ppl={val_ppl:.4f}")
            tracker.log(
                step=it + 1,
                metrics={"val_loss": float(val_loss), "val_ppl": float(val_ppl)},
            )

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, it + 1, best_checkpoint_path, best_val=best_val)
        if (it + 1) % cfg.run.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, it + 1, checkpoint_path, best_val=best_val)
    save_checkpoint(model, optimizer, cfg.run.max_iterations, checkpoint_path, best_val=best_val)
    tracker.close()


if __name__ == "__main__":
    main()
