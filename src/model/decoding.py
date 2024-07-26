from typing import AbstractSet, Literal, Union

import tiktoken
import torch


def generate_from_single_input(
    model, ids, max_new_tokens, context_len, temperature=1.0, top_k=None
):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(ids[:, -context_len:], inference=True)
            if temperature == 0 or top_k is None or top_k == 1:
                new_id = out.argmax(-1)
            else:
                v, temp_idxs = torch.topk(
                    out / temperature, k=min(top_k, out.size(-1)), dim=-1
                )
                new_id = torch.multinomial(
                    torch.softmax(v, -1), num_samples=1
                ).squeeze()
                new_id = temp_idxs[:, new_id]
            ids = torch.cat((ids, new_id.unsqueeze(0)), dim=-1)
    model.train()
    return ids


def text_to_ids(
    text,
    tokenizer: tiktoken.Encoding,
    num_threads=1,
    allowed_special: Union[Literal["all"], AbstractSet[str]] = "all",
):
    return tokenizer.encode_batch(
        text, num_threads=num_threads, allowed_special=allowed_special
    )


def ids_to_text(
    ids, tokenizer: tiktoken.Encoding, num_threads=1, to_bytes=False
):
    """
    For safe decoding, set to_bytes=True.

    If text is non-utf-8, decode_batch is lossy, so set to_bytes=True.
    """
    if to_bytes:
        return tokenizer.decode_bytes_batch(ids, num_threads=num_threads)
    return tokenizer.decode_batch(ids, num_threads=num_threads)
