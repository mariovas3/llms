import torch

from src.model import lora_utils
from src.model.gpt import GPT


def test_lora_workflow_do_ffn():
    do_ffn = True
    model = GPT()
    # get module list of LoRALayer
    lora_module_list = lora_utils.init_lora_module_list_qv(
        model, rank=8, alpha=1, do_ffn=do_ffn
    )
    # load the module list to the model;
    lora_utils.load_lora_layers_qv_(model, lora_module_list, do_ffn=do_ffn)
    # bind;
    lora_utils.bind_lora_qv_(model, do_ffn=do_ffn)
    # unbind;
    lora_utils.unbind_lora_qv_(model, do_ffn=do_ffn)
    # extract module list;
    lora_module_list2 = lora_utils.extract_lora_layers_qv(model, do_ffn=do_ffn)
    for mod1, mod2 in zip(lora_module_list, lora_module_list2):
        assert mod1 is mod2
        assert torch.all(mod1.A == mod2.A)
        assert torch.all(mod1.B == mod2.B)


def test_lora_workflow():
    do_ffn = False
    model = GPT()
    # get module list of LoRALayer
    lora_module_list = lora_utils.init_lora_module_list_qv(
        model, rank=8, alpha=1, do_ffn=do_ffn
    )
    # load the module list to the model;
    lora_utils.load_lora_layers_qv_(model, lora_module_list, do_ffn=do_ffn)
    # bind;
    lora_utils.bind_lora_qv_(model, do_ffn=do_ffn)
    # unbind;
    lora_utils.unbind_lora_qv_(model, do_ffn=do_ffn)
    # extract module list;
    lora_module_list2 = lora_utils.extract_lora_layers_qv(model, do_ffn=do_ffn)
    for mod1, mod2 in zip(lora_module_list, lora_module_list2):
        assert mod1 is mod2
        assert torch.all(mod1.A == mod2.A)
        assert torch.all(mod1.B == mod2.B)
