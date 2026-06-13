from dataclasses import dataclass


@dataclass
class ModelConfig:
    layers: int = 24
    embedding_dim: int = 4096
    query_heads: int = 32
    key_value_heads: int = 8
    block_size: int = 8192
    norm_eps: float = 1e-5
    hidden_dim: int = (
        11008  # 4 * embedding_dim * 2 / 3 = 10922.67 -> 11008 (incremented to next closest multiple of 256)
    )
    rope_base: int = 50000
    vocab_size: int = 128000


@dataclass
class DatasetConfig:
    name: str = ""
    subset: str = ""
    split: str = ""
    tokenizer_path: str = ""


@dataclass
class TrainerConfig:
    batch_size: int = 1
    num_workers: int = 6
    weight_lr: float = 2e-2
    lm_head_lr: float = 4e-3
    embedding_lr: float = 0.2
    bias_lr: float = 4e-3
    min_lr_ratio: float = 0.1
    n_iter: int = 2e5
    warmup_iters: int = 2e4
    constant_iters: int = 1e5
    ckpt_path: str = ""
    save_every: int = 1000
    keep_last: int = 3
    accum_steps: int = 1
    device: str = "cuda"
