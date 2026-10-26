from dataclasses import dataclass, field
from typing import Optional


@dataclass
class runConfig:
    run_dirs: str = "runs"
    run_name_prefix: str = "owt-contextlength512-batch128-numheads8"
    run_name: str = None
    resume_path: str = None
    max_iterations: int = 1000000
    batch_size: int = 128
    log_interval: int = 100
    eval_interval: int = 100
    checkpoint_interval: int = 100
    eval_batches: int = 64


@dataclass
class dataConfig:
    # train_data_path: str = "/home/lichao-liu/assignment1-basics/cs336_basics/output/tinystories_token_ids_np_train.pkl"
    # val_data_path: str = "/home/lichao-liu/assignment1-basics/cs336_basics/output/tinystories_token_ids_np_valid.pkl"
    train_data_path: str = "/home/lichao-liu/assignment1-basics/cs336_basics/output/owt_token_ids_np_train.pkl"
    val_data_path: str = "/home/lichao-liu/assignment1-basics/cs336_basics/output/owt_token_ids_np_valid.pkl"
    np_dtype: str = "uint16"
    device: str = "cuda:0"


@dataclass
class modelConfig:
    torch_dtype: str = "fp32"
    context_length: int = 512
    d_ff: Optional[int] = 1344
    d_model: int = 512
    vocab_size: int = 32_000
    num_layers: int = 4
    num_heads: int = 8
    max_seq_len: int = 1000
    rope_theta: int = 10000


@dataclass
class optimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    max_lr: float = 1e-3
    min_lr: float = 3e-5
    warmup_iters: int = 200
    cosine_cycle_iters: int = 5000
    grad_clip: int = 1


@dataclass
class trainConfig:
    run: field = field(default_factory=runConfig)
    data: field = field(default_factory=dataConfig)
    optimizer: field = field(default_factory=optimizerConfig)
    model: field = field(default_factory=modelConfig)


def getConfig():
    cfg = trainConfig()
    return cfg
