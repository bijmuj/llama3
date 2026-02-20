import os
from glob import glob

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
            Transformer(model_config).to(self.config.device).to(torch.bfloat16)
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

        self.optimizer = self.configure_optimizer()
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

    def configure_optimizer(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.RMSNorm, torch.nn.Embedding)

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, blacklist_weight_modules
                ):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, whitelist_weight_modules
                ):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params),
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.1,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-5,
        )
        return optimizer

    def load_most_recent(self):
        files = glob(f"{self.config.ckpt_path}/*.pt")
        latest_file = max(files, key=os.path.getctime)
        ckpt = torch.load(latest_file, weights_only=False)

        self.start_iter = ckpt["start_iter"]
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.model.load_state_dict(ckpt["model"])
        self.lr_scheduler.load_state_dict(ckpt["scheduler"])

    def train(self, wandb_run=None):
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
            loss = cross_entropy(
                logits.float().view(-1, logits.size(-1)), y.view(-1)
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.lr_scheduler.step()

            if torch.isnan(loss).any():
                raise f"Encountered NaN on iter {n_iters+1}"

            if wandb_run is not None:
                wandb_run.log({"loss": loss.item()})

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
                        "scheduler": self.lr_scheduler.state_dict(),
                        "start_iter": n_iters + 1,
                    },
                    file_path,
                )

                old_files.append(file_path)
                while len(old_files) > self.config.keep_last:
                    os.remove(old_files[0])
                    old_files.pop(0)
