import os

import torch
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingLR,
    LinearLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config import DatasetConfig, ModelConfig, TrainerConfig
from dataset import PackedStreamingDataset
from datasets import load_dataset
from hf_tokens import READ_ONLY_TOKEN
from model import Transformer


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
    ):
        self.config = config
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.start_iter = 0

        self.model = (
            Transformer(model_config).to(self.config.device).to(torch.float16)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            dataset_config.tokenizer_path
        )

        self.base_dataset = load_dataset(
            dataset_config.name,
            dataset_config.subset,
            split=dataset_config.split,
            streaming=True,
            token=READ_ONLY_TOKEN,
        )
        self.dataset = PackedStreamingDataset(
            self.base_dataset, self.tokenizer, self.model_config.block_size
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=0.1,
        )
        self.lr_scheduler = ChainedScheduler(
            [
                LinearLR(
                    self.optimizer,
                    start_factor=self.config.min_lr_ratio,
                    end_factor=1.0,
                    total_iters=self.config.warmup_iters,
                ),
                CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.n_iter - self.config.warmup_iters,
                    eta_min=self.config.learning_rate
                    * self.config.min_lr_ratio,
                ),
            ]
        )

    def train(self):
        os.makedirs(self.config.ckpt_path, exist_ok=True)
        self.model.train()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        data_iter = iter(dataloader)
        old_files = []

        for n_iters in tqdm(range(self.start_iter, self.config.n_iter)):
            try:
                batch = next(data_iter)
            except:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            x, y = batch
            x = x[:, :-1].to(self.config.device)
            y = y[:, 1:].to(self.config.device)

            self.optimizer.zero_grad()

            logits = self.model(x)

            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            if (n_iters + 1) % self.config.save_every == 0:
                file_path = os.path.join(
                    self.config.ckpt_path, f"trainer-{n_iters+1}.pt"
                )
                print(f"saving to: {file_path}")
                torch.save(
                    {
                        "trainer_config": self.config,
                        "model_config": self.model_config,
                        "dataset_config": self.dataset_config,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.lr_scheduler,
                        "start_iter": n_iters + 1,
                    },
                    file_path,
                )

                old_files.append(file_path)
                while len(old_files) > self.config.keep_last:
                    os.remove(old_files[0])
                    old_files.pop(0)
