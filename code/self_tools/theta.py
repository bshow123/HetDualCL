import torch


def compute_pos_stats(pos):
    """
    计算正样本矩阵的统计信息
    pos: 稀疏或稠密张量，形状 [num_nodes, num_nodes]
    返回: (平均值, 中位数, 最大值, 最小值, 标准差)
    """
    if pos.is_sparse:
        # 稀疏矩阵：提取行索引并统计每个节点的出现次数
        row_indices = pos._indices()[0]
        counts = torch.bincount(row_indices, minlength=pos.size(0))
    else:
        # 稠密矩阵：直接对行求和
        counts = pos.sum(dim=1)

    # 转换为浮点数以便计算
    counts = counts.float()
    avg = counts.mean().item()
    median = counts.median().item()
    max_val = counts.max().item()
    min_val = counts.min().item()
    std = counts.std().item()
    return avg, median, max_val, min_val, std