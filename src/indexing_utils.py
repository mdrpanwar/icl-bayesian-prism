import torch


def batched_index_select(input, dim, index):
    len_inp_shape = len(input.shape)
    if dim == -1:
        dim = len_inp_shape - 1
    for ii in range(1, len_inp_shape):
        if ii != dim:
            index = index.unsqueeze(ii)
    # print(index.shape)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    # print(index)
    return torch.gather(input, dim, index)
