import torch


def compute_pos_stats(pos):

    if pos.is_sparse:
        row_indices = pos._indices()[0]
        counts = torch.bincount(row_indices, minlength=pos.size(0))
    else:
        counts = pos.sum(dim=1)

    # 转换为浮点数以便计算
    counts = counts.float()
    avg = counts.mean().item()
    median = counts.median().item()
    max_val = counts.max().item()
    min_val = counts.min().item()
    std = counts.std().item()
    return avg, median, max_val, min_val, std