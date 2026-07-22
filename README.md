# LLMs

Small repo to learn how to build LLMs.

# Running

## 0. Install dependencies
- [Install](https://docs.astral.sh/uv/#highlights) `uv` package manager. 
- Create virtual environment: `$ uv venv`
- Activate virtual environment: `$ source .venv/bin/activate`
- Install pytorch and flash-attn separately:
    ```
    $ uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
    $ uv pip install flash-attn --no-build-isolation
    ```
- Install the other dependencies: `$ uv sync`

## 1. Setup Tokenizer

Tokenizers for the english subset of `allenai/c4` and `mlfoundations/dclm-baseline-1.0` are included in the repo. For other datasets do the following:
- Create a huggingface account.
- Login through the cli: `$ uvx hf auth login` 
- Run the tokenizer training script:
    ```
    $ uv run tokenizer.py --dataset_name mlfoundations/dclm-baseline-1.0 --subset "" --out_dir tokenizers/dclm-baseline-50k --vocab_size 50432 --max_samples=500000
    ```
- For more information see: [tokenizer.py](tokenizer.py)

## 2. Run training

- To run training with `tiny` model and `dclm baseline` dataset configs included the repo run:
    ```
    $ uv run main.py --trainer_config ./configs/trainer_tiny.yml --model_config ./configs/model_tiny.yml --dataset_config ./configs/dataset_dclm_baseline.yml
    ```
- To create your own configs create a .yml file with keys matching one of the dataclasses in [config.py](config.py).

# TODO

- add an inference script
- more attention variants (linear attention, DeepSeek MLA, etc)
- KV-caching
- quantization