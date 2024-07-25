"""
Contains the LoRA utilities for my version of GPT implementation.

Note that in the transformer decoder blocks, the MLP is of the type:

MLP(
  (net): Sequential(
    (0): Linear(in_features=768, out_features=3072, bias=True)
    (1): GELU(approximate='tanh')
    (2): Linear(in_features=3072, out_features=768, bias=True)
  )
)

So mlp.net[0] is the first nn.Linear and mlp.net[2] is the second
nn.Linear.
"""

import torch
from torch import nn


def load_lora_layers_qv_(my_gpt, lora_layers: nn.ModuleList, do_ffn=False):
    """
    Inplace modify query and value layers, also ffn weights if do_ffn is True.

    Assumes lora_layers is list of [query_layer, value_layer] if do_ffn is False
    or [query_layer, value_layer, intermediate_ffn, output_ffn] if do_ffn is True.
    """
    mult = 4 if do_ffn else 2
    for i, layer in enumerate(my_gpt.decoder.layers):
        layer.mod.mha.Uq = LinearWithLoRA.from_lora_layer(
            layer.mod.mha.Uq, lora_layers[mult * i]
        )
        layer.mod.mha.Uv = LinearWithLoRA.from_lora_layer(
            layer.mod.mha.Uv, lora_layers[mult * i + 1]
        )
        if do_ffn:
            layer.mod.mlp.net[0] = LinearWithLoRA.from_lora_layer(
                layer.mod.mlp.net[0], lora_layers[mult * i + 2]
            )
            layer.mod.mlp.net[2] = LinearWithLoRA.from_lora_layer(
                layer.mod.mlp.net[2], lora_layers[mult * i + 3]
            )


def extract_lora_layers_qv(my_gpt, do_ffn=False):
    """
    Return nn.ModuleList of LoRA layers.

    Assumes lora layers only in query and value if do_ffn is False
    and also considers ffn's intermediate and output if do_ffn is True.
    """
    ml = nn.ModuleList()
    for layer in my_gpt.decoder.layers:
        ml.append(layer.mod.mha.Uq.lora)
        ml.append(layer.mod.mha.Uv.lora)
        if do_ffn:
            ml.append(layer.mod.mlp.net[0].lora)
            ml.append(layer.mod.mlp.net[2].lora)
    return ml


def bind_lora_qv_(my_gpt, do_ffn=False):
    """
    Adds the lora approximation to the frozen weights inplace.

    This is for faster inference. Only query and value weights
    are considered if do_ffn is False, otherwise also consider
    ffn's intermediate and output weights.
    """
    for layer in my_gpt.decoder.layers:
        query = layer.mod.mha.Uq
        if hasattr(query, "lora"):
            query.bind()
        value = layer.mod.mha.Uv
        if hasattr(value, "lora"):
            value.bind()
        if do_ffn:
            intermediate = layer.mod.mlp.net[0]
            if hasattr(intermediate, "lora"):
                intermediate.bind()
            output = layer.mod.mlp.net[2]
            if hasattr(output, "lora"):
                output.bind()


def unbind_lora_qv_(my_gpt, do_ffn=False):
    """
    Subtract the lora weights from the frozen weights inplace.

    Only query and value weights are considered if do_ffn is False,
    otherwise also consider ffn's intermediate and output weights.
    """
    for layer in my_gpt.decoder.layers:
        query = layer.mod.mha.Uq
        if hasattr(query, "lora"):
            query.unbind()
        value = layer.mod.mha.Uv
        if hasattr(value, "lora"):
            value.unbind()
        if do_ffn:
            intermediate = layer.mod.mlp.net[0]
            if hasattr(intermediate, "lora"):
                intermediate.unbind()
            output = layer.mod.mlp.net[2]
            if hasattr(output, "lora"):
                output.unbind()


def init_lora_module_list_qv(my_gpt, rank, alpha, do_ffn=False):
    """
    Initialise nn.ModuleList with lora weights.

    Only init for query and value weights if do_ffn is False,
    otherwise also init for ffn's intermediate and output weights.
    """
    d_model = my_gpt.classification_head.in_features
    ffn_hidden = my_gpt.decoder.layers[0].mod.mlp.net[0].out_features
    n_layers = len(my_gpt.decoder.layers)
    lora_weights = nn.ModuleList()
    for i in range(n_layers):
        lora_weights.append(LoRALayer(d_model, d_model, rank, alpha))
        lora_weights.append(LoRALayer(d_model, d_model, rank, alpha))
        if do_ffn:
            lora_weights.append(LoRALayer(d_model, ffn_hidden, rank, alpha))
            lora_weights.append(LoRALayer(ffn_hidden, d_model, rank, alpha))
    return lora_weights


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.recip_rank, self.alpha = 1 / rank, alpha
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros((rank, out_dim)))

    def get_lora_approx(self):
        return self.A @ self.B * self.alpha * self.recip_rank

    def forward(self, x):
        return (x @ self.A) @ self.B * self.alpha * self.recip_rank


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            in_dim=linear.in_features,
            out_dim=linear.out_features,
            rank=rank,
            alpha=alpha,
        )
        self.bound = False

    @classmethod
    def from_lora_layer(cls, linear, lora_layer):
        ob = cls(linear, 1, 1)
        ob.lora = lora_layer
        return ob

    def bind(self):
        self.bound = True
        self.linear.weight.data = (
            self.linear.weight.data
            + self.lora.get_lora_approx().transpose(-1, -2)
        )

    def unbind(self):
        self.bound = False
        self.linear.weight.data = (
            self.linear.weight.data
            - self.lora.get_lora_approx().transpose(-1, -2)
        )

    def forward(self, x):
        if self.bound:
            return self.linear(x)
        return self.linear(x) + self.lora(x)
