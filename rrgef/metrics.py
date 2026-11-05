"""Harrell's concordance index (vectorized, numpy only)."""
import numpy as np


def concordance_index_fast(event_times, predicted_scores, event_observed):
    """向量化 C-index, 与 concordance_index 逐位等价 (已单元验证)。

    原实现是 Python 双层循环 O(N^2) 逐对迭代, N=1000 时单次调用需数百毫秒,
    bootstrap(上千次重采样 x 多方法 x 多场景) 会慢到不可行。这里用布尔矩阵广播替代。

    可比较对的等价刻画: 有序对 (a,b) 满足 e_a==1 且 t_a < t_b。
    (两个方向不可能同时成立, 故与原版"每个无序对至多计一次"一致。)
    """
    t = np.asarray(event_times, float)
    s = np.nan_to_num(np.asarray(predicted_scores, float), nan=0.0)
    e = np.asarray(event_observed, int)
    valid = (e[:, None] == 1) & (t[:, None] < t[None, :])
    total = valid.sum()
    if total == 0:
        return 0.5
    conc = (valid & (s[:, None] > s[None, :])).sum()
    tied = (valid & (s[:, None] == s[None, :])).sum()
    return float((conc + 0.5 * tied) / total)


concordance_index = concordance_index_fast
