import argparse

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, UnicodeScripts, Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from hf_tokens import READ_ONLY_TOKEN

LLAMA_3_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    "<|start_header_id|>",
    "<|reserved_special_token_4|>",
    "<|end_header_id|>",
    "<|eot_id|>",  # end of turn
]
UNKNOWN_TOKEN = "<|unk|>"
PAD_TOKEN = "<|pad|>"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trains a BPE tokenizer using huggingface tokenizers on a huggingface dataset."
    )
    parser.add_argument("--dataset_name", type=str, default="allenai/c4")
    parser.add_argument("--subset", type=str, default="en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--out_dir", type=str, default="checkpoints/llama3-c4-128k"
    )
    parser.add_argument("--vocab_size", type=int, default=128000)
    return parser.parse_args()


def batch_iterator(
    dataset, batch_size: int = 1000, max_samples: int = 5_000_000
):
    batch = []
    count = 0

    for example in dataset:
        text = example["text"]

        if len(text) < 200:  # optional filtering
            continue

        batch.append(text)
        count += 1

        if len(batch) == batch_size:
            yield batch
            batch = []

        if count >= max_samples:
            break

    if batch:
        yield batch


def train_tokenizer(args):
    tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN))
    tokenizer.pre_tokenizer = Sequence([Whitespace(), UnicodeScripts()])

    dataset = load_dataset(
        args.dataset_name,
        args.subset,
        split=args.split,
        streaming=True,
        token=READ_ONLY_TOKEN,
    )
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[*LLAMA_3_SPECIAL_TOKENS, UNKNOWN_TOKEN, PAD_TOKEN],
    )

    tokenizer.train_from_iterator(batch_iterator(dataset), trainer)

    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=LLAMA_3_SPECIAL_TOKENS[0],
        eos_token=LLAMA_3_SPECIAL_TOKENS[1],
        unk_token=UNKNOWN_TOKEN,
        pad_token=PAD_TOKEN,
        extra_special_tokens=LLAMA_3_SPECIAL_TOKENS[2:],
    ).save_pretrained(args.out_dir)


if __name__ == "__main__":
    train_tokenizer(parse_args())
