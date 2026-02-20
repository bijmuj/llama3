import torch

import wandb
from config import DATASET_C4, MODEL_300M, TrainerConfig
from trainer import Trainer


def main():
    trainer_config = TrainerConfig(ckpt_path="./checkpoints/llama3-1b")
    run = wandb.init(
        entity="bijin",
        project="llama3",
        config=dict(
            model_config=vars(MODEL_300M),
            trainer_config=vars(trainer_config),
            dataset_config=vars(DATASET_C4),
        ),
    )
    trainer = Trainer(trainer_config, MODEL_300M, DATASET_C4)
    trainer.train(run)


if __name__ == "__main__":
    main()
