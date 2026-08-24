# Experiment Results — TinyStories LM

Runs from `hyperpod:/home/lichao-liu/assignment1-basics/runs`, timestamped after `20260821_050000`.

All runs below share: `weight_decay=0.1`, `beta1=0.9`, `beta2=0.95`, `eps=1e-8`,
`grad_clip=1`, `max_iterations=5000`, `eval_batches=64`, `context_length=256`,
`d_model=512`, `d_ff=1344`, `num_layers=4`, `num_heads=16`, `vocab_size=10000`, fp32.

**Target: validation loss (per-token) ≤ 1.45**

## Completed runs

| Run | batch | `lr` | `max_lr` | `min_lr` | warmup | cosine_cycle | Steps | Wall time | Tokens | Best val_loss (step) | Final val_loss | ≤1.45 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline-batch-64_20260821_052631` | 64 | 3e-4 | 1e-3 | 3e-5 | 200 | 5000 | 5000 | 12.2 min | 81.9M | 1.5163 (4800) | 1.5215 | ✗ |
| `baseline-lr_20260821_053446` | 64 | 3e-2 | 1e-3 | 3e-5 | 200 | 5000 | 5000 | 12.1 min | 81.9M | 1.5095 (4770) | 1.5215 | ✗ |
| `baseline-lr-1_20260821_053505` | 64 | 3e-1 | 1e-3 | 3e-5 | 200 | 5000 | 5000 | 12.2 min | 81.9M | 1.5128 (4980) | 1.5128 | ✗ |
| `baseline-batch-128_20260821_050222` | 128 | 3e-4 | 1e-3 | 3e-5 | 200 | 5000 | 5000 | 21.5 min | 163.8M | **1.4259 (4920)** | 1.4311 | ✓ |
| `baseline-batch-256_20260821_050351` | 256 | 3e-4 | 1e-3 | 3e-5 | 200 | 5000 | 5000 | 40.2 min | 327.7M | **1.3635 (4860)** | 1.3663 | ✓ |
| `baseline-batch-512_20260821_050405` | 512 | 3e-4 | 1e-3 | 3e-5 | 200 | 5000 | 5000 | 77.5 min | 655.4M | **1.3234 (4890)** | 1.3248 | ✓ |

Best result: **batch 512 → val_loss 1.3234**, comfortably under the 1.45 target.

## Notes

### Batch size shows diminishing returns

Each doubling roughly doubles wall time for a shrinking gain:

| Transition | val_loss gain | Wall time cost |
|---|---|---|
| 64 → 128 | 0.090 | 1.8× |
| 128 → 256 | 0.062 | 1.9× |
| 256 → 512 | 0.040 | 1.9× |

Batch 512 costs 6.4× the wall time of batch 64 for a 0.19 improvement. For the single
best number, use 512; for fast iteration, batch 128 clears the target in 21 minutes.

### `cfg.optimizer.lr` is dead config

The three batch-64 runs varied `lr` across 3e-4 → 3e-2 → 3e-1 (1000× range) and produced
1.5163 / 1.5095 / 1.5128 — a spread of only 0.007. Cause: at
`cs336_basics/train.py:102-106` the cosine schedule overwrites `group["lr"]` at the top of
every iteration, before the first `optimizer.step()`. The value passed to the `adamw`
constructor never affects an update.

Effective LR is determined entirely by `max_lr`, `min_lr`, `warmup_iters`, and
`cosine_cycle_iters`. **To sweep LR, vary `max_lr`** (keeping `min_lr ≈ max_lr/100`).

Silver lining: those three runs are a clean seed-variance measurement (spread 0.007),
confirming the batch-size differences above are real signal, far outside noise.

### Batch 512 is the memory ceiling

Batch 640 (×3 attempts), 768, and 1024 all failed with CUDA OOM in 6-8 seconds — same wall
every attempt, so deterministic rather than contention. Root cause: the logits tensor is
`B×S×V` fp32 (`1024×256×10000×4` ≈ 10.5 GB per copy), and `cross_entropy` in
`cs336_basics/nn_utils.py:13-16` materializes several full-size copies (`m`,
`normalized_input`, then `logsumexp`).

Going above batch 512 at `context_length=256` requires rewriting the loss to avoid those
copies (e.g. chunked/fused cross-entropy).

### Prior runs were capped at 3.54 by a weight-init bug

Earlier runs plateaued at val_loss ~3.54 regardless of batch size or step count. Cause:
`swiglu.__init__` and both attention classes in `cs336_basics/modules.py` allocated weights
with `torch.empty(...)` and never initialized them, leaving q/k/v/output projections and
w1/w2/w3 at zero. With `v_proj = output_proj = 0` attention output is 0, and with
`w1 = w2 = w3 = 0` SwiGLU output is 0, so every block collapsed to the identity
(`x + 0 = x`) and the model reduced to a bigram predictor `lm(ln(embedding(x)))`.

This was a true fixed point of gradient descent — all gradients to those weights were
identically zero, so no learning rate or step count could escape it. Adding
`nn.init.trunc_normal_` with Xavier scaling (`std = sqrt(2/(fan_in+fan_out))`, truncated at
±3σ) fixed it, taking val_loss from a frozen 3.54 to 1.32.

## Failed runs — CUDA OOM, 0 steps

| Run | batch | `lr` | `max_lr` | `min_lr` | warmup | cosine_cycle |
|---|---|---|---|---|---|---|
| `baseline-batch-640_20260821_051844` | 640 | 3e-4 | 1e-3 | 3e-5 | 200 | 5000 |
| `baseline-batch-640_20260821_052116` | 640 | 3e-4 | 1e-3 | 3e-5 | 200 | 5000 |
| `baseline-batch-640_20260821_052411` | 640 | 3e-4 | 1e-3 | 3e-5 | 200 | 5000 |
| `baseline-batch-768_20260821_051743` | 768 | 3e-4 | 1e-3 | 3e-5 | 200 | 5000 |
| `baseline-batch-1024_20260821_051125` | 1024 | 3e-4 | 1e-3 | 3e-5 | 200 | 5000 |

## Suggested next steps

1. **Sweep `max_lr`** at batch 256 over `3e-4 / 1e-3 / 3e-3` to check whether `1e-3` is on
   the right side of the optimum. Batch 256 clears the target in 40 min, so three parallel
   runs cost ~40 min of wall time given idle cluster capacity.
2. **Remove or rename `lr`** in `cs336_basics/config.py:42` so it can't mislead a future sweep.
3. **Set a stable `run_name`** if using the `preemptible` partition — `train.py:43-48`
   generates a fresh timestamped dir per launch, so a requeued job won't find its checkpoint
   and will silently restart from step 0.