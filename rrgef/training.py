"""Training and inference helpers for the RRGEF network (frozen paper code).

Functions are taken unchanged from the experiment code; only imports and
machine-specific paths were removed.
"""
import numpy as np
import torch

from rrgef.model import EvidentialFusion, cox_ph_loss, evidence_reg

EPOCHS = 200
LR = 1e-3
WD = 1e-4
DEV = torch.device("cpu")


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def pseudo_label(time, event):
    """训练折内的高/低危伪标签, 供 EDL 正则用; 删失且早于中位者不参与。"""
    med = np.median(time)
    y = np.full(len(time), -1)
    y[(event == 1) & (time <= med)] = 1
    y[time > med] = 0
    return y


def train_fusion(Xtr, ttr, etr, mods, use_r, use_n, epochs=EPOCHS,
                 r_prior_init=None, lam_rprior=0.3, dropout=0.3,
                 wd=WD, d_hid=64, d_rep=32, triggered_r=False,
                 r_mode="abs", r_floor=0.15,
                 bounded_evidence=False, c_max=8.0, robust_clip=False,
                 cross_reliability=False, cross_weight=1.0,
                 tau_x_init=3.0,
                 contrastive_consistency=False, consistency_temp=0.2,
                 clip_q=0.005, clip_margin=0.05,
                 evid_drop=False, deep_cross=False, w_cross=0.1,
                 energy_reliability=False, energy_weight=0.5, tau_e_init=1.0,
                 energy_low_guard=False, energy_low_frac=0.1,
                 w_consistency=0.0, c1_rule=True, decompose=True,
                 risk_mode="evidence", risk_gain=1.0,
                 modality_dropout=0.0):
    """r_prior_init: 每模态"预后感知"硬初始化 (logit 尺度, 来自单模态预测力).
    lam_rprior: 把 r_prior 拉回 init 的 L2 强度 (越大越"硬").
    triggered_r: 触发式可靠性 (M6), 默认关, 不影响 M5 行为.
    r_mode/r_floor: M8 相对可靠性 (见 EvidentialFusion 文档)."""
    dims = {m: Xtr[m].shape[1] for m in mods}
    model = EvidentialFusion(dims, n_cls=2, use_r=use_r, use_n=use_n,
                             dropout=dropout, d_hid=d_hid, d_rep=d_rep,
                             r_prior_init=r_prior_init, triggered_r=triggered_r,
                             r_mode=r_mode, r_floor=r_floor,
                             bounded_evidence=bounded_evidence, c_max=c_max,
                             robust_clip=robust_clip,
                             cross_reliability=cross_reliability,
                             cross_weight=cross_weight,
                             tau_x_init=tau_x_init,
                             contrastive_consistency=contrastive_consistency,
                             consistency_temp=consistency_temp,
                             risk_mode=risk_mode, risk_gain=risk_gain,
                             evid_drop=evid_drop, deep_cross=deep_cross,
                             energy_reliability=energy_reliability,
                             energy_weight=energy_weight, tau_e_init=tau_e_init,
                             energy_low_guard=energy_low_guard,
                             energy_low_frac=energy_low_frac,
                             c1_rule=c1_rule, decompose=decompose).to(DEV)
    if robust_clip:
        # 用训练折的经验分位数当裁剪边界(留 margin), 推理时越界值被拉回训练支撑集内。
        # 这是树模型"分裂阈值只在训练范围内"这一稳健性来源的可微对应物。
        for m in mods:
            lo = np.quantile(Xtr[m], clip_q, axis=0)
            hi = np.quantile(Xtr[m], 1.0 - clip_q, axis=0)
            span = np.maximum(hi - lo, 1e-6)
            model.branches[m].set_clip_range(lo - clip_margin * span, hi + clip_margin * span)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    xs = {m: torch.tensor(Xtr[m], device=DEV) for m in mods}
    T = torch.tensor(ttr, dtype=torch.float32, device=DEV)
    E = torch.tensor(etr, dtype=torch.float32, device=DEV)
    y = pseudo_label(ttr, etr)
    has = y >= 0
    y1h = torch.zeros(len(y), 2, device=DEV)
    y1h[np.where(has)[0], y[has]] = 1.0
    has_t = torch.tensor(has, device=DEV)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        train_mask = None
        if modality_dropout > 0.0 and len(mods) > 1:
            train_mask = (torch.rand((len(ttr), len(mods)), device=DEV)
                          > float(modality_dropout)).float()
            empty = train_mask.sum(1) == 0
            if empty.any():
                idx = torch.randint(0, len(mods), (int(empty.sum()),), device=DEV)
                train_mask[empty, idx] = 1.0
        out = model(xs, train_mask)
        risk = model.risk(out)
        loss = cox_ph_loss(risk, T, E)
        loss = loss + 0.1 * out["rec"].mean() + w_cross * out["cross_loss"]
        loss = loss + w_consistency * out["consistency_loss"]
        if has.sum() > 5:
            lam = min(1.0, ep / max(1, epochs // 2)) * 0.05     # KL 退火
            loss = loss + evidence_reg(out["alpha"][has_t], y1h[has_t], lam)
        if model.r_prior_init.abs().sum() > 0:                  # 硬预后先验: 拉回目标值
            loss = loss + lam_rprior * (model.r_prior - model.r_prior_init).pow(2).sum()
        if getattr(model, "triggered_r", False):                 # 触发式 r: 把 tau/k 温和拉回初值, 防漂移
            loss = loss + 0.01 * ((model.tau - 3.0).pow(2).sum() + (model.k - 2.0).pow(2).sum())
            if getattr(model, "cross_reliability", False):
                loss = loss + 0.01 * (model.tau_x - 3.0).pow(2).sum()
            if getattr(model, "evid_drop", False):
                loss = loss + 0.01 * (model.tau_d - 1.5).pow(2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    if getattr(model, "robust_stats", False):
        model.calibrate(xs)          # V2: 训练后用训练集标定稳健阈值(分位数)
    return model


# ---------------- 集成 (小数据抗过拟合) ----------------
def train_fusion_ens(Xtr, ttr, etr, mods, use_r, use_n, K, base_seed,
                     r_prior_init=None, triggered_r=False, **kw):
    if K <= 1:
        return train_fusion(Xtr, ttr, etr, mods, use_r, use_n,
                            r_prior_init=r_prior_init, triggered_r=triggered_r, **kw)
    models = []
    for k in range(K):
        torch.manual_seed(base_seed * 1000 + k + 7)
        models.append(train_fusion(Xtr, ttr, etr, mods, use_r, use_n,
                                   r_prior_init=r_prior_init, triggered_r=triggered_r, **kw))
    return models


@torch.no_grad()
def eval_fusion_ens(models, Xte, mods, mask=None, return_dict=False):
    if not isinstance(models, list):
        return eval_fusion(models, Xte, mods, mask, return_dict=return_dict)
    outs = [eval_fusion(mdl, Xte, mods, mask, return_dict=True) for mdl in models]
    if not return_dict:
        return (np.mean([o["risk"] for o in outs], 0),
                np.mean([o["conflict"] for o in outs], 0),
                np.mean([o["r"] for o in outs], 0),
                np.mean([o["n"] for o in outs], 0),
                np.mean([o["u"] for o in outs], 0))
    merged = {}
    for k in outs[0].keys():
        merged[k] = np.mean(np.stack([o[k] for o in outs], 0), 0)
    return merged


@torch.no_grad()
def eval_fusion(model, Xte, mods, mask=None, return_dict=False):
    xs = {m: torch.tensor(Xte[m], device=DEV) for m in mods}
    mk = None if mask is None else torch.tensor(mask, dtype=torch.float32, device=DEV)
    out = model(xs, mk)
    if return_dict:
        d = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in out.items()}
        d["risk"] = model.risk(out).detach().cpu().numpy()
        return d
    return (model.risk(out).cpu().numpy(),
            out["conflict"].squeeze(1).cpu().numpy(),
            out["r"].squeeze(-1).cpu().numpy(),
            out["n"].squeeze(-1).cpu().numpy(),
            out["u"].squeeze(1).cpu().numpy())


def eval_fusion_ens_full(models, Xte, mods, mask=None):
    """返回完整信号字典(供可靠性/冲突/公理实验), 不破坏旧 5 元组接口。

    字典含: risk, alpha, p, u_total, u_data, u_ign, u_miss, U_conflict,
            r_m(M,N 每模态可靠性), n_m(M,N), e_stack。
    """
    out = eval_fusion_ens(models, Xte, mods, mask, return_dict=True)
    alpha = out["alpha"]
    risk = np.log(alpha[:, 1] + 1e-8) - np.log(alpha[:, 0] + 1e-8)
    # `u` is the evidence-mass ignorance signal used by the existing
    # risk-coverage code.  Keep it under an explicit name instead of
    # silently conflating it with the entropy-based decomposition term
    # `u_ign`.
    u_total = out["u"].squeeze(-1)
    return {
        "risk": risk,
        "alpha": alpha,
        "p": out["p"],
        "u_total": u_total,                   # evidence-mass ignorance for abstention
        "u_ign": out["u_ign"].squeeze(-1),  # entropy decomposition epistemic term
        "u_data": out["u_data"].squeeze(-1),  # aleatoric (digamma 期望熵)
        "u_miss": out["u_miss"].squeeze(-1),  # missing/unreliable = -log(平均可靠性)
            "U_conflict": out["conflict"].squeeze(-1),
            "U_conflict_raw": out["conflict_raw"].squeeze(-1),
            "r_m": out["r"],            # (M,N)
            "n_m": out["n"],            # (M,N)
            "e_stack": out["e_stack"],
        "energy": out["energy"],
        "energy_excess": out["energy_excess"],
        "consistency": out["consistency"],  # (M,N,1), matched-patient mismatch
    }
