import argparse

import torch
import yaml

import wandb
from config import DatasetConfig, ModelConfig, TrainerConfig
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trains a llama 3 style model with given config values."
    )
    parser.add_argument(
        "--trainer_config", type=str, default="./configs/default_trainer.yml"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="./configs/dataset_c4.yml"
    )
    parser.add_argument(
        "--model_config", type=str, default="./configs/model_300m.yml"
    )
    parser.add_argument("--wandb_run", type=str, default="")
    args = parser.parse_args()
    return args


def load_yml(path):
    with open(path, "r") as f:
        file = yaml.safe_load(f)

    return file


def main(args):
    model_config_dict = load_yml(args.model_config)
    trainer_config_dict = load_yml(args.trainer_config)
    dataset_config_dict = load_yml(args.dataset_config)

    wandb_dict = dict(
        entity="bijin",
        project="llama3",
        config=dict(
            model_config=model_config_dict,
            trainer_config=trainer_config_dict,
            dataset_config=dataset_config_dict,
        ),
    )
    if len(args.wandb_run):
        wandb_dict["resume"] = "allow"
        wandb_dict["id"] = args.wandb_run

    run = wandb.init(**wandb_dict)
    trainer = Trainer(
        TrainerConfig(**trainer_config_dict),
        ModelConfig(**model_config_dict),
        DatasetConfig(**dataset_config_dict),
    )
    trainer.load_most_recent()
    trainer.train(run)


if __name__ == "__main__":
    main(parse_args())
