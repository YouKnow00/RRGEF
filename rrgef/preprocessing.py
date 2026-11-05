"""Fold-safe feature selection and standardization.

All statistics (variance cap, univariate Cox ranking, mean/sd) are fitted on
the training fold only and then applied to the test fold.  The paper uses
``preselect="raw_fold"``, ``variance_k=1024`` and ``k=256``.
"""
import numpy as np
import torch
from torch import logcumsumexp

SELECT_K = 256
VARIANCE_K = 1024


# ---------------- 单变量 Cox 特征筛选 ----------------
def cox_select_topk(Xtr, ttr, etr, k):
    """向量化单变量 Cox: 同时给 F 个特征各学一个 β, 取 |β| 最大的 k 个。
    在训练折上拟合, 避免泄漏。"""
    N, F = Xtr.shape
    k = int(min(k, F, max(1, N - 1)))
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xz = (Xtr - mu) / sd
    Xz = torch.tensor(Xz, dtype=torch.float32)
    T = torch.tensor(ttr, dtype=torch.float32)
    E = torch.tensor(etr, dtype=torch.float32)
    order = torch.argsort(T, descending=True)
    Xo = Xz[order]                                   # (N, F)
    evo = E[order].unsqueeze(1)                       # (N, 1)
    d = evo.sum().clamp(min=1.0)
    beta = torch.zeros(F, requires_grad=True)
    opt = torch.optim.Adam([beta], lr=0.05)
    for _ in range(40):
        R = Xo * beta                                 # (N, F) 每特征独立风险
        logcum = logcumsumexp(R, 0)                   # 按时间降序累积 (Breslow)
        ll = (R - logcum) * evo                       # (N, F)
        loss = -ll.sum() / d
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([beta], 5.0)
        opt.step()
    with torch.no_grad():
        idx = torch.argsort(beta.abs(), descending=True)[:k].numpy()
    return idx


def select_fit_apply(Xall_tr, Xall_te, ttr, etr, k,
                     preselect="raw_fold", variance_k=VARIANCE_K):
    """对每模态在训练折筛选、标准化并应用到训练/测试。

    Feature ranking and all centering/scaling statistics are fitted on the
    training fold only.  With ``preselect='fold_variance'`` an additional
    unsupervised variance cap is also fitted on the training fold before Cox
    ranking; the default ``legacy`` path preserves the historical aligned
    feature pool for paired comparison.
    """
    Xtr_s, Xte_s = {}, {}
    for m in Xall_tr:
        xtr_all = np.asarray(Xall_tr[m], dtype=np.float32)
        xte_all = np.asarray(Xall_te[m], dtype=np.float32)
        if preselect in {"fold_variance", "raw_fold"} and xtr_all.shape[1] > k:
            # ``fold_variance`` is the historical sensitivity path over the
            # globally aligned candidate pool.  ``raw_fold`` uses the raw
            # patient-level matrices and therefore makes this the first
            # feature-pool operation in the pipeline.  Both fit the cap on
            # the training fold only before supervised Cox ranking.
            cap = int(variance_k or max(2 * k, k))
            cap = min(cap, xtr_all.shape[1])
            var = np.nanvar(xtr_all, axis=0)
            pre_idx = np.argsort(np.nan_to_num(var, nan=-np.inf))[::-1][:cap]
            xtr_all = xtr_all[:, pre_idx]
            xte_all = xte_all[:, pre_idx]
        idx = cox_select_topk(xtr_all, ttr, etr, k)
        tr = np.asarray(xtr_all[:, idx], dtype=np.float32)
        te = np.asarray(xte_all[:, idx], dtype=np.float32)
        mu = np.nanmean(tr, axis=0, keepdims=True)
        sd = np.nanstd(tr, axis=0, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        tr = np.nan_to_num((tr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        te = np.nan_to_num((te - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        Xtr_s[m] = np.ascontiguousarray(tr, dtype=np.float32)
        Xte_s[m] = np.ascontiguousarray(te, dtype=np.float32)
    return Xtr_s, Xte_s
