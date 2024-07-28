import os

import torch
from torch import nn

import src.model.utils as mutils
from src.data.lit_data import get_alpaca_instruction
from src.metadata import metadata
from src.model import decoding, lora_utils
from src.model.gpt import GPT

os.environ["TIKTOKEN_CACHE_DIR"] = str(metadata.SAVED_MODELS_PATH)
import tiktoken

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_inference(checkpoint_path):
    # get the checkpoint;
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    # get hyper params;
    hparams = ckpt["hyper_parameters"]
    model_name = hparams["from_pretrained_model"]
    assert model_name is not None

    # see if should load in bf16;
    is_bf16 = "bf16" in str(checkpoint_path).split("/")[-1]

    # get tokenizer;
    tokenizer = tiktoken.get_encoding("gpt2")

    # get gpt;
    config = metadata.BASE_CONFIG.copy()
    config.update(metadata.MODEL_CONFIGS[model_name])
    config["ffn_hidden"] = 4 * config["d_model"]
    my_gpt = GPT(**config)

    # load relevant openai weights;
    mutils.load_pretrained_weights_(
        my_gpt=my_gpt,
        cache_dir=metadata.SAVED_MODELS_PATH,
        name=model_name,
    )
    my_gpt.requires_grad_(False)
    if is_bf16:
        my_gpt = my_gpt.to(torch.bfloat16)

    # load lora;
    if hparams["do_lora"]:
        # load lora module list;
        lora_module_list = lora_utils.init_lora_module_list_qv(
            my_gpt,
            rank=hparams["lora_rank"],
            alpha=hparams["lora_alpha"],
            do_ffn=hparams["do_ffn"],
        )
        if is_bf16:
            lora_module_list = lora_module_list.to(torch.bfloat16)
        lora_module_list.load_state_dict(
            ckpt["state_dict"]["lora_module_list"]
        )
        # load lora modules into gpt;
        lora_utils.load_lora_layers_qv_(
            my_gpt,
            lora_layers=lora_module_list,
            do_ffn=hparams["do_ffn"],
        )
        # bind lora weights into gpt weights for fast inference;
        lora_utils.bind_lora_qv_(my_gpt, do_ffn=hparams["do_ffn"])

    return CombinedModel(my_gpt, tokenizer)


class CombinedModel(nn.Module):
    def __init__(self, my_gpt, tokenizer):
        super().__init__()
        self.my_gpt = my_gpt
        self.tokenizer = tokenizer

    def get_response(self, flow: dict, temperature=0) -> str:
        """
        flow should have 'instruction' and 'input' keys.
        """
        instruction = get_alpaca_instruction(flow)
        device = next(self.my_gpt.parameters()).device
        in_ids = torch.tensor(
            decoding.text_to_ids([instruction], self.tokenizer),
            dtype=torch.long,
        ).to(device)
        ids = decoding.generate_from_single_input(
            self.my_gpt,
            ids=in_ids,
            temperature=temperature,
            top_k=5 if temperature > 0 else None,
            max_new_tokens=128,
            context_len=metadata.BASE_CONFIG["context_length"],
            eos_id=50256,
        ).to(torch.device("cpu"))
        pred = decoding.ids_to_text(
            ids.tolist(), self.tokenizer, to_bytes=True
        )[0]
        return pred[len(instruction) :].strip().decode("utf-8")
