import torch
from torch.utils.data import IterableDataset


class PackedStreamingDataset(IterableDataset):
    def __init__(self, base_dataset, tokenizer, block_size):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        buffer = []

        for example in self.base_dataset:
            tokens = self.tokenizer(example["text"])["input_ids"]
            buffer.extend(tokens)

            while len(buffer) >= self.block_size:
                chunk = buffer[: self.block_size]
                buffer = buffer[self.block_size :]

                yield [
                    torch.tensor(chunk),
                    torch.tensor(chunk),
                ]
