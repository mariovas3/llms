import time
from pathlib import Path
from typing import Union

import tiktoken
import torch
import utils
from jsonargparse import ArgumentParser
from torch.utils.data import DataLoader

from src.data.utils import GPT2Dataset
from src.model.gpt import GPT


def main():
    # DATA PREP;
    tokenizer = tiktoken.get_encoding("gpt2")

    DATA_PATH = (
        Path(__file__).absolute().parents[3]
        / "data/downloaded/the-verdict.txt"
    )
    with open(DATA_PATH, "r") as f:
        txt_data = f.read()
    train_prop = 0.9
    split_idx = int(train_prop * len(txt_data))
    train_txt = txt_data[:split_idx]
    val_txt = txt_data[split_idx:]
    CONTEXT_LEN = 256

    torch.manual_seed(123)

    train_ids = tokenizer.encode(train_txt)
    val_ids = tokenizer.encode(val_txt)
    print(
        f"{len(train_txt)=}\t{len(train_ids)=}\n{len(val_txt)=}\t{len(val_ids)=}"
    )

    CONTEXT_LEN = 256
    STRIDE = CONTEXT_LEN  # 100
    train_dataset = GPT2Dataset(
        train_ids, context_len=CONTEXT_LEN, stride=STRIDE
    )
    val_dataset = GPT2Dataset(val_ids, context_len=CONTEXT_LEN, stride=STRIDE)

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, drop_last=False
    )
    print(f"{len(train_loader)=}\t{len(val_loader)=}")
    print(f"{len(train_dataset)=}\t{len(val_dataset)=}")

    # MODEL PREP;
    parser = ArgumentParser()
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--val_check_interval", type=Union[float, int], default=5
    )
    parser.add_argument(
        "--val_loop_iters",
        type=int,
        default=-1,
        help="How many batches to evaluate on in val loop. If -1 eval on all.",
    )
    parser = GPT.add_to_argparse(parser)
    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(123)
    model = GPT(
        num_layers=args["num_layers"],
        vocab_size=args["vocab_size"],
        context_length=args["context_length"],
        d_model=args["d_model"],
        num_heads=args["num_heads"],
        ffn_hidden=args["ffn_hidden"],
        activation=args["activation"],
        dropout=args.get("dropout", 0.1),
        qkv_bias=args.get("qkv_bias", False),
        norm_first=args.get("norm_first", False),
        pre_norm=args.get("pre_norm", False),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params=:,}")
    optim = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # START TRAINING;
    now = time.time()
    train_losses, val_losses, tokens_seen = utils.train_loop(
        max_epochs=args["max_epochs"],
        model=model,
        optim=optim,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        tokenizer=tokenizer,
        val_check_interval=args["val_check_interval"],
        golden_prompt="Every effort moves you",
        val_loop_iters=args["val_loop_iters"],
    )
    print(f"took time: {time.time() - now} secs")
    utils.plot_losses(
        args["max_epochs"], train_losses, val_losses, tokens_seen
    )


if __name__ == "__main__":
    main()
