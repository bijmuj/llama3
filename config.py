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


MODEL_300M = ModelConfig(
    layers=16,
    embedding_dim=768,
    hidden_dim=2048,
    block_size=2048,
)
MODEL_3B = ModelConfig()
MODEL_7B = ModelConfig(layers=32, hidden_dim=14336)


@dataclass
class DatasetConfig:
    name: str = ""
    subset: str = ""
    split: str = ""
    tokenizer_path: str = ""


DATASET_C4 = DatasetConfig(
    name="allenai/c4",
    subset="en",
    split="train",
    tokenizer_path="./tokenizers/c4-128k",
)


@dataclass
class TrainerConfig:
    batch_size: int = 1
    num_workers: int = 6
    n_iter: int = 2_000_000
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_iters: int = 2000
    ckpt_path: str = ""
    save_every: int = 1000
    keep_last: int = 3
    device: str = "cuda"
