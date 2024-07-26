from typing import AbstractSet, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch

from src.model.decoding import *


def plot_losses(num_epochs, train_losses, val_losses, tokens_seen):
    epochs_train = np.linspace(0, num_epochs, len(train_losses))
    epochs_val = np.linspace(0, num_epochs, len(val_losses))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(epochs_train, train_losses, linestyle="-", label="train loss")
    ax.plot(epochs_val, val_losses, linestyle="--", label="val loss")
    ax.set_ylabel("loss")
    ax.set_xlabel("Epochs")
    plt.legend()
    ax2 = ax.twiny()
    ax2.plot(tokens_seen, train_losses)
    ax2.set_xlabel("tokens seen")
    plt.tight_layout()
    plt.savefig(f"gpt_training_{num_epochs}_epochs.png")


def val_loop(model, loss_fn, val_loader, device, vocab_len, max_batches=1):
    model.eval()
    avg_batch_loss = 0
    n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if max_batches > 0 and i == max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out.view(-1, vocab_len), y.view(-1))
            n += 1
            avg_batch_loss = (
                avg_batch_loss + (loss.item() - avg_batch_loss) / n
            )
    model.train()
    return avg_batch_loss


def train_loop(
    max_epochs,
    model,
    optim,
    loss_fn,
    train_loader,
    val_loader,
    device,
    tokenizer,
    val_check_interval=None,
    golden_prompt="What doesn't kill you, ",
    val_loop_iters=-1,
):
    if val_check_interval is not None:
        assert val_check_interval > 0
        if val_check_interval < 1:
            every_n_batches = round(val_check_interval * len(train_loader))
        else:
            every_n_batches = val_check_interval
    golden_prompt_ids = torch.LongTensor(
        text_to_ids([golden_prompt], tokenizer)
    )

    model.train()
    # init tracking variables;
    global_step, max_iters = -1, max_epochs * len(train_loader)
    train_losses, val_losses, tokens_seen = [], [], []
    tokens_processed = 0

    # start training;
    for ep in range(max_epochs):
        print(f"\n\nEpoch {ep + 1}:\n{'-' * 30}")
        avg_epoch_train_loss = 0
        n = 0
        for x, y in train_loader:
            global_step += 1
            tokens_processed += x.numel()
            # move to device;
            x, y = x.to(device), y.to(device)
            # zero grad;
            optim.zero_grad()
            # logits;
            out = model(x)
            # get loss;
            l = loss_fn(out.view(-1, tokenizer.n_vocab), y.view(-1))
            # update avg per batch train loss for epoch;
            n += 1
            avg_epoch_train_loss = (
                avg_epoch_train_loss + (l.item() - avg_epoch_train_loss) / n
            )
            # backprop;
            l.backward()
            # grad step;
            optim.step()
            # val check;
            if val_check_interval and global_step % every_n_batches == 0:
                avg_train_loss = val_loop(
                    model=model,
                    loss_fn=loss_fn,
                    val_loader=train_loader,
                    device=device,
                    vocab_len=tokenizer.n_vocab,
                    max_batches=val_loop_iters,
                )
                avg_val_loss = val_loop(
                    model=model,
                    loss_fn=loss_fn,
                    val_loader=val_loader,
                    device=device,
                    vocab_len=tokenizer.n_vocab,
                    max_batches=val_loop_iters,
                )
                val_losses.append(avg_val_loss)
                print(
                    f"\niters done: {global_step}/{max_iters}\t{global_step=}\n"
                    f"avg_train_loss over {val_loop_iters} "
                    f"batches: {avg_train_loss:.3f}\n"
                    f"avg_val_loss over {val_loop_iters} batches: {avg_val_loss:.3f}"
                )
        new_ids = generate_from_single_input(
            model=model,
            ids=golden_prompt_ids,
            max_new_tokens=50,
            context_len=model.pos_embeddings.embed.weight.shape[0],
            temperature=1.0,
        )
        print(
            f"\nEnd of epoch {ep+1}:\n"
            f"Average batch train loss in epoch: {avg_epoch_train_loss}\n"
            f"prompt: {golden_prompt}\ngen_text: {ids_to_text(new_ids.tolist(), tokenizer, to_bytes=True)[0]}"
        )
        tokens_seen.append(tokens_processed)
        train_losses.append(avg_epoch_train_loss)
    return train_losses, val_losses, tokens_seen
