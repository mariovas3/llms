import torch


def copy_mha_weights(naive_mha, parallel_mha):
    # make sure I have the same weights;
    Uq, Uk, Uv = [], [], []
    Uq_b, Uk_b, Uv_b = [], [], []
    for m in naive_mha.heads:
        Uq.append(m.Uq.weight.data)
        if m.Uq.bias is not None:
            Uq_b.append(m.Uq.bias.data)
        Uk.append(m.Uk.weight.data)
        if m.Uk.bias is not None:
            Uk_b.append(m.Uk.bias.data)
        Uv.append(m.Uv.weight.data)
        if m.Uv.bias is not None:
            Uv_b.append(m.Uv.bias.data)
    parallel_mha.Uo.weight.data = naive_mha.Uo.weight.data
    parallel_mha.Uo.bias.data = naive_mha.Uo.bias.data

    # in torch, nn.Linear(d_model, dk).weight.data.shape is (dk, d_model)!!!
    # so concat along first dim to get (d_model, d_model);
    parallel_mha.Uq.weight.data = torch.cat(Uq, 0)
    if Uq_b:
        parallel_mha.Uq.bias.data = torch.cat(Uq_b, -1)
    parallel_mha.Uk.weight.data = torch.cat(Uk, 0)
    if Uk_b:
        parallel_mha.Uk.bias.data = torch.cat(Uk_b, -1)
    parallel_mha.Uv.weight.data = torch.cat(Uv, 0)
    if Uv_b:
        parallel_mha.Uv.bias.data = torch.cat(Uv_b, -1)
