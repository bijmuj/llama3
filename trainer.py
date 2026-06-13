import os
from glob import glob
from itertools import islice

import torch
from normuon import SingleDeviceNorMuonWithAuxAdam
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
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
            Transformer(model_config).to(self.config.device).to(torch.bfloat16)
        )
        print(
            "model param count: ",
            sum(p.numel() for p in self.model.parameters()),
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

        self.optimizer = self.configure_optimizer()

        linear_warmup = LinearLR(
            self.optimizer,
            start_factor=self.config.min_lr_ratio,
            end_factor=1.0,
            total_iters=self.config.warmup_iters,
        )
        constant_schedule = ConstantLR(
            self.optimizer,
            factor=1.0,
            total_iters=self.config.constant_iters,
        )
        cosine_anneal = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.n_iter
            - self.config.warmup_iters
            - self.config.constant_iters,
        )
        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                linear_warmup,
                constant_schedule,
                cosine_anneal,
            ],
            milestones=[
                self.config.warmup_iters,
                self.config.warmup_iters + self.config.constant_iters,
            ],
        )

    def configure_optimizer(self):
        hidden_weights = [
            p for p in self.model.layers.parameters() if p.ndim >= 2
        ]
        hidden_biases = [
            p for p in self.model.layers.parameters() if p.ndim < 2
        ]
        embeddings = [p for p in self.model.embedding.parameters()]
        lm_head_weights = [
            p for p in self.model.lm_head.parameters() if p.ndim >= 2
        ]
        lm_head_biases = [
            p for p in self.model.lm_head.parameters() if p.ndim < 2
        ]

        # create the pytorch optimizer object
        optim_groups = [
            # hidden weights : NorMuon + decay
            {
                "params": hidden_weights,
                "weight_decay": 0.01,
                "lr": self.config.weight_lr,
                "momentum": 0.95,
                "beta2": 0.95,
                "use_muon": True,
            },
            # lm head weights : AdamW + decay
            {
                "params": lm_head_weights,
                "weight_decay": 0.01,
                "lr": self.config.lm_head_lr,
                "betas": (0.9, 0.95),
                "eps": 1e-5,
                "use_muon": False,
            },
            # embedding : AdamW
            {
                "params": embeddings,
                "weight_decay": 0,
                "lr": self.config.embedding_lr,
                "betas": (0.9, 0.95),
                "eps": 1e-5,
                "use_muon": False,
            },
            # biases : AdamW
            {
                "params": [*hidden_biases, *lm_head_biases],
                "weight_decay": 0,
                "lr": self.config.bias_lr,
                "betas": (0.9, 0.95),
                "eps": 1e-5,
                "use_muon": False,
            },
        ]
        optimizer = SingleDeviceNorMuonWithAuxAdam(optim_groups)
        return optimizer

    def load_most_recent(self):
        files = glob(f"{self.config.ckpt_path}/*.pt")
        if len(files):
            latest_file = max(files, key=os.path.getctime)
            print(f"trying to load {latest_file}")
            ckpt = torch.load(latest_file, weights_only=False)

            self.start_iter = ckpt["start_iter"]
            self.base_dataset = self.base_dataset.skip(self.start_iter)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.model.load_state_dict(ckpt["model"])
            self.lr_scheduler.load_state_dict(ckpt["scheduler"])

            print(f"loaded checkpoint: {latest_file}")

    def train(self, wandb_run=None):
        os.makedirs(self.config.ckpt_path, exist_ok=True)
        self.model.train()

        dataset = PackedStreamingDataset(
            self.base_dataset, self.tokenizer, self.model_config.block_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        data_iter = iter(dataloader)
        old_files = glob(f"{self.config.ckpt_path}/*.pt")
        losses = []
        for n_iters in tqdm(range(self.start_iter, self.config.n_iter)):
            try:
                batch = next(data_iter)
            except:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            x, y = batch
            x = x[:, :-1].to(self.config.device)
            y = y[:, 1:].to(self.config.device)

            logits = self.model(x)
            loss = cross_entropy(
                logits.float().view(-1, logits.size(-1)), y.view(-1)
            )

            if torch.isnan(loss).any():
                raise f"Encountered NaN on iter {n_iters+1}"

            loss = loss.float() / self.config.accum_steps
            loss.backward()
            losses.append(loss.detach())

            if (n_iters + 1) % self.config.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            if (n_iters + 1) % (self.config.save_every) == 0:
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
                        "scheduler": self.lr_scheduler.state_dict(),
                        "start_iter": n_iters + 1,
                    },
                    file_path,
                )
                if wandb_run is not None:
                    for l in losses:
                        wandb_run.log({"loss": l.item()})
                losses = []

                old_files.append(file_path)
                while len(old_files) > self.config.keep_last:
                    os.remove(old_files[0])
                    old_files.pop(0)
