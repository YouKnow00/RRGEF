"""RRGEF network: modality branches, relative reliability gating and evidential fusion.

Each omics modality has an encoder whose latent representation feeds
  - an evidence head (softplus, two channels: lower-/higher-risk direction), and
  - a reconstruction decoder whose error drives the modality's reliability.
Reconstruction errors are calibrated on the training fold, thresholded and
compared within each patient (``r_mode="rel"``) to give relative reliabilities
r_m; fused evidence is sum_m v_m * r_m * e_m and alpha = e + 1.  The risk score
is log(alpha_high) - log(alpha_low).

This file is the frozen implementation used for the paper (v2, 2026-08-03).
Several constructor flags expose development options that are switched off in
the paper configuration (see ``rrgef.RRGEF`` for the settings actually used).
Some inline comments are in Chinese.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class ModalityBranch(nn.Module):
    """单模态分支：编码 + 证据头 + 自重建头(→可靠性) + 跨模态预测头(→新颖度)"""

    def __init__(self, d_in, d_hid=64, d_rep=32, n_cls=2, dropout=0.3,
                 bounded_evidence=False, c_max=8.0, robust_clip=False):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid, d_rep), nn.ReLU(),
        )
        self.evid = nn.Linear(d_rep, n_cls)          # -> softplus 得证据
        self.dec = nn.Linear(d_rep, d_in)            # 自重建 -> 可靠性代理
        # 可靠性标定参数 (可学的温度/偏置，把重建误差映射到 [0,1])
        self.r_scale = nn.Parameter(torch.tensor(1.0))
        self.r_bias = nn.Parameter(torch.tensor(0.0))
        # 训练集重建误差的 running 统计量: 推理时以「训练时见过的正常水平」为基准,
        # 这样整批被污染的模态不会因为减 batch 均值而把异常抵消掉。
        self.register_buffer("rec_mean", torch.tensor(1.0))
        self.register_buffer("rec_var", torch.tensor(1.0))
        # ---- V2 开关 (默认关, 关闭时与 V1 逐位等价) ----
        # bounded_evidence: V1 的 softplus 证据是**无界**的, 污染放大输入 -> 证据幅度爆炸
        #   (LUAD 实测 σ=8 时 ×3.14), 把 r 的降权吃掉大半 (r 降 85%, 融合占比只降 36%)。
        #   V2 拆成 "方向 × 有界质量": e = c_max·sigmoid(mass(h)) · softmax(evid(h)),
        #   总量 ≤ c_max, 于是 w=z·r·n 能真正决定融合占比, 可靠性降权不再被反噬。
        # robust_clip: 编码器输入按训练集分位数裁剪 (树模型稳健性的可微对应物)。
        #   关键: 重建误差仍对**未裁剪的原始 x** 计算 -> 越界信号完整保留给可靠性头,
        #   而表示 h 不被越界值带崩 —— 检测通道与表示通道解耦。
        self.bounded_evidence = bool(bounded_evidence)
        self.c_max = float(c_max)
        self.robust_clip = bool(robust_clip)
        # 证据总量上限 + 重建误差的稳健基准(中位数/分位距)。
        # 为什么不用 mean+3sd: 实测这些量是**重尾**的(LUAD: rec_mean=0.37 但 sd=0.77,
        # xcons_mean=16.1 但 sd=29.3), 均值+3倍标准差的阈值高得永远触发不了。
        # 改用 med 与 (p90-med) 做稳健标准化, 由 calibrate() 在训练后用训练集一次性标定。
        self.register_buffer("evid_cap", torch.tensor(1e30))
        self.register_buffer("rec_med", torch.tensor(0.0))
        self.register_buffer("rec_iqr", torch.tensor(1.0))
        self.register_buffer("clip_lo", torch.full((d_in,), -1e30))
        self.register_buffer("clip_hi", torch.full((d_in,), 1e30))

    @torch.no_grad()
    def set_clip_range(self, lo, hi):
        self.clip_lo.copy_(torch.as_tensor(lo, dtype=self.clip_lo.dtype))
        self.clip_hi.copy_(torch.as_tensor(hi, dtype=self.clip_hi.dtype))

    def forward(self, x):
        x_enc = torch.clamp(x, self.clip_lo, self.clip_hi) if self.robust_clip else x
        h = self.enc(x_enc)
        e = F.softplus(self.evid(h))                 # (N,K) >=0
        if self.bounded_evidence:
            # 证据"软封顶": 总量在训练集正常范围内时**原样通过**(与 V1 逐位相同),
            # 只有超过训练分布上限时才按比例压回 —— 方向不变, 只削幅度。
            # 这样 clean 判别力零损失, 而污染导致的幅度爆炸无法再反噬 r 的降权。
            if not self.training:
                s = e.sum(dim=1, keepdim=True)
                e = e * torch.clamp(self.evid_cap / (s + EPS), max=1.0)
        xr = self.dec(h)
        # 重建误差始终对**原始** x 计算: 裁剪只保护表示, 不掩盖越界证据
        rec = ((xr - x) ** 2).mean(dim=1, keepdim=True)      # (N,1)
        if self.training:
            with torch.no_grad():
                self.rec_mean.mul_(0.9).add_(0.1 * rec.mean())
                self.rec_var.mul_(0.9).add_(0.1 * rec.var(unbiased=False).clamp(min=1e-8))
        return h, e, rec


class EvidentialFusion(nn.Module):
    def __init__(self, dims, n_cls=2, d_hid=64, d_rep=32, dropout=0.3,
                 use_r=True, use_n=True, r_prior_init=None, triggered_r=False,
                 r_mode="abs", r_floor=0.15,
                 bounded_evidence=False, c_max=8.0, robust_clip=False,
                 cross_reliability=False, tau_x_init=3.0,
                 cross_weight=1.0,
                 contrastive_consistency=False, consistency_temp=0.2,
                 evid_drop=False, tau_d_init=1.5, deep_cross=False, c1_rule=True,
                 energy_reliability=False, energy_weight=0.5, tau_e_init=1.0,
                 energy_low_guard=False, energy_low_frac=0.1,
                 decompose=True, risk_mode="evidence", risk_gain=1.0):
        """dims: dict 模态名 -> 输入维度
        triggered_r: 触发式可靠性 (M6 新增, 默认关, 不影响 M5 行为)。
          干净(重建误差在标准范围内)时 gate≈1(等权); 仅当重建误差显著越界才降权。
          不叠加预后先验, 避免把 r 钉成常数、掩盖对脏数据的响应(M5 的 bug)。
        r_mode (仅 triggered_r=True 时生效):
          "abs" = M6 绝对开关: gate 直接当 r。缺陷: 多个强模态同时被污染时全被压到 0,
                  只剩最弱模态撑场面 -> 崩到接近瞎猜 (M7 双污染实测 BRCA 0.512 / LUAD 0.475)。
          "rel" = M8 相对可靠性: 先按当前样本上"可用模态里最干净的那个"做归一化,
                  再加地板 r_floor。含义变成"谁比谁更可信", 而不是"绝对有多脏"。
                  => 全脏时所有 gate 一起变小, 归一化后 r≈1, 自动退回等权平均(不劣于基线);
                     部分脏时脏的一方仍被压到 ~r_floor, 保留 M6 的降权收益。
        r_floor: 相对模式下的最低权重, 防止把某模态彻底清零(0=纯相对)。"""
        super().__init__()
        self.mods = list(dims.keys())
        self.M = len(self.mods)
        self.n_cls = n_cls
        self.use_r, self.use_n = use_r, use_n
        self.triggered_r = triggered_r
        self.r_mode = r_mode
        self.r_floor = float(r_floor)
        self.risk_mode = str(risk_mode)
        self.risk_gain = float(risk_gain)
        # 触发式 r 的参数: 阈值 tau 与锐度 k (每模态)。
        # 初始化 tau=3, k=2 -> 干净(zs≈0) gate=1; 重建误差越界才降权。
        self.tau = nn.Parameter(torch.full((self.M,), 3.0))
        self.k = nn.Parameter(torch.full((self.M,), 2.0))
        # ---- V2: 第二可靠性通道 (跨模态一致性) ----
        # 动机: 重建误差只能抓"边际分布异常"(越界/尺度变化)。对 featshuf 这类
        #   **分布内**污染(逐特征独立打乱, 边际分布一字不变、越界率 0%),
        #   重建误差几乎不动 -> r 保持 1.000 完全不降权 -> C-index 掉 0.078 (LUAD 实测)。
        # 方案: 用"别的模态能否预测出我的表示"作为第二探针。样本错配/特征打乱会
        #   彻底破坏跨模态对应关系 -> 一致性误差相对训练基准暴涨 -> 触发降权。
        #   合并方式 z = max(z_rec, z_cross): 任一通道报警即降权。
        self.cross_reliability = bool(cross_reliability)
        self.cross_weight = float(cross_weight)
        self.contrastive_consistency = bool(contrastive_consistency)
        self.consistency_temp = float(consistency_temp)
        self.c1_rule = bool(c1_rule)
        self.decompose = bool(decompose)
        self.robust_stats = bool(cross_reliability or bounded_evidence or evid_drop
                                 or energy_reliability)
        self.tau_x = nn.Parameter(torch.full((self.M,), float(tau_x_init)))
        self.register_buffer("xcons_mean", torch.ones(self.M))
        self.register_buffer("xcons_var", torch.ones(self.M))
        self.register_buffer("xcons_med", torch.zeros(self.M))
        self.register_buffer("xcons_iqr", torch.ones(self.M))
        # Training-fold quality of the cross-modal probe.  A source modality
        # that cannot predict other modalities better than a null predictor
        # must not be allowed to veto a valid sample at test time.
        self.register_buffer("cross_quality", torch.ones(self.M))
        # ---- V2 可靠性通道 C: 证据塌缩 ----
        # 观测: 模态被打乱/清零后, 它的表示塌缩、证据总量整体掉下来
        #   (LUAD methyl: clean 5.30 -> featshuf 3.44 -> wipeout 1.74)。
        # 这是"该模态在这批数据上整体失声"的直接读数, 且不依赖 cross 头的预测能力。
        # 用**批中位数**判定(整模态污染是批级事件), 阈值同样由训练集分位数标定。
        self.evid_drop = bool(evid_drop)
        self.tau_d = nn.Parameter(torch.full((self.M,), float(tau_d_init)))
        self.register_buffer("evid_med", torch.ones(self.M))
        self.register_buffer("evid_lo", torch.zeros(self.M))
        # ---- V2 可靠性通道 D: 双侧特征能量/OOD ----
        # 对训练折内已经标准化的输入计算 mean(x^2)。高尾抓加性/尺度
        # OOD，低尾抓 wipeout/模态塌缩；中心和两侧 q01/q99 支撑边界均只由训练折校准。
        # 该通道只在 eval 阶段启用，避免未校准统计量干扰训练。
        self.energy_reliability = bool(energy_reliability)
        self.energy_weight = float(energy_weight)
        # Optional development-only guard for the lower energy tail.  The
        # default preserves the original q01/q99 two-sided rule.  When
        # enabled, low-tail evidence is triggered only by near-zero energy
        # (a modality collapse/wipeout), avoiding false alarms from ordinary
        # low-but-valid samples.
        self.energy_low_guard = bool(energy_low_guard)
        self.energy_low_frac = float(energy_low_frac)
        self.register_buffer("tau_e", torch.full((self.M,), float(tau_e_init)))
        self.register_buffer("energy_med", torch.ones(self.M))
        self.register_buffer("energy_lo_scale", torch.ones(self.M))
        self.register_buffer("energy_hi_scale", torch.ones(self.M))
        self.deep_cross = bool(deep_cross)
        self.branches = nn.ModuleDict(
            {m: ModalityBranch(dims[m], d_hid, d_rep, n_cls, dropout,
                               bounded_evidence=bounded_evidence, c_max=c_max,
                               robust_clip=robust_clip) for m in self.mods})
        # Direct prognostic head shares the reliability-weighted latent fusion
        # but remains separate from the Dirichlet evidence readout.  This
        # prevents the uncertainty representation from becoming the only
        # source of clean survival discrimination.
        rep_dim = d_rep
        self.risk_head = nn.Linear(rep_dim, 1)
        # 跨模态预测头: 从模态 i 的表示预测模态 j 的表示
        self.cross = nn.ModuleDict()
        for i in self.mods:
            for j in self.mods:
                if i != j:
                    # deep_cross: 单层线性头预测力太弱(实测比"输出均值"还差), 加一层非线性,
                    # 让跨模态探针真正具备判别力; 关闭时与 V1 结构完全一致。
                    self.cross[f"{i}->{j}"] = (
                        nn.Sequential(nn.Linear(d_rep, d_rep), nn.ReLU(),
                                      nn.Linear(d_rep, d_rep))
                        if deep_cross else nn.Linear(d_rep, d_rep))
        self.consistency_proj = nn.ModuleDict({
            m: nn.Linear(d_rep, d_rep, bias=False) for m in self.mods
        })
        # 跨模态可预测误差的训练集基准 (每模态一个)
        self.register_buffer("nov_scale", torch.ones(self.M))
        # 预后感知可靠性先验: 每模态一个"硬"偏置, 直接用"该模态单独预测准不准"初始化
        # (见 m5_run.prog_logits), 再由 Cox 损失做小幅微调 + L2 拉回, 防止被过度调偏。
        # 这样 r 的"懂事"是决定性强的, 不再像旧版那样只学到 ±0.03 的软漂移。
        # r 的样本级 corruption 检测仍由重建标准分 zs 提供, 此处只做"模态级"的整体抬升/压低。
        self.r_prior = nn.Parameter(torch.zeros(self.M))
        if r_prior_init is not None:
            init = torch.as_tensor(r_prior_init, dtype=torch.float32).reshape(self.M)
        else:
            init = torch.zeros(self.M)
        self.register_buffer("r_prior_init", init)       # 预后感知的"硬"目标值
        with torch.no_grad():
            self.r_prior.copy_(self.r_prior_init)

    @torch.no_grad()
    def calibrate(self, xs, q_hi=0.90, q_cap=0.99):
        """训练结束后用**训练集**一次性标定稳健阈值 (V2)。

        为什么必须单独标定而不是训练中 EMA:
          - 这些量重尾, 均值/标准差不具代表性, 必须用分位数;
          - 训练中带 dropout, 与推理(eval)分布不一致, 阈值会偏。
        标定内容: 每模态重建误差与跨模态不一致度的 med / (p90-med); 证据总量的 p99 上限。
        """
        was_training = self.training
        self.eval()
        H, REC, S, ENERGY = {}, {}, {}, {}
        for m in self.mods:
            b = self.branches[m]
            x = xs[m]
            x_enc = torch.clamp(x, b.clip_lo, b.clip_hi) if b.robust_clip else x
            h = b.enc(x_enc)
            H[m] = h
            S[m] = F.softplus(b.evid(h)).sum(1)
            REC[m] = ((b.dec(h) - x) ** 2).mean(1)
            ENERGY[m] = x.pow(2).mean(1)
        for i, m in enumerate(self.mods):
            b = self.branches[m]
            med = REC[m].median()
            b.rec_med.copy_(med)
            b.rec_iqr.copy_((REC[m].quantile(q_hi) - med).clamp(min=1e-8))
            b.evid_cap.copy_(S[m].quantile(q_cap).clamp(min=1e-6))
            self.evid_med[i] = S[m].median()
            self.evid_lo[i] = S[m].quantile(0.10)
            emed = ENERGY[m].median()
            elo = ENERGY[m].quantile(0.01)
            ehi = ENERGY[m].quantile(0.99)
            self.energy_med[i] = emed
            self.energy_lo_scale[i] = (emed - elo).clamp(min=1e-8)
            self.energy_hi_scale[i] = (ehi - emed).clamp(min=1e-8)
            others = [k for k in self.mods if k != m]
            if others:
                if self.contrastive_consistency:
                    zc = {k: F.normalize(self.consistency_proj[k](H[k]), dim=1)
                          for k in self.mods}
                    xc = torch.stack([
                        1.0 - (zc[m] * zc[k]).sum(-1)
                        for k in others
                    ], 0).mean(0)
                else:
                    # 与 forward 一致: m 作为**源**去预测其他模态的表示
                    xc = torch.stack([((self.cross[f"{m}->{k}"](H[m]) - H[k]) ** 2).mean(1)
                                      for k in others], 0).mean(0)
                mx = xc.median()
                self.xcons_med[i] = mx
                self.xcons_iqr[i] = (xc.quantile(q_hi) - mx).clamp(min=1e-8)
                if self.contrastive_consistency:
                    negs = []
                    for k in others:
                        perm = torch.roll(torch.arange(H[k].shape[0], device=H[k].device), 1)
                        negs.append(1.0 - (zc[m] * zc[k][perm]).sum(-1))
                    neg_mean = torch.stack(negs).mean()
                    quality = ((neg_mean - xc.mean()) / (neg_mean.abs() + EPS)).clamp(0.0, 1.0)
                else:
                    # Compare the fitted probe with a null predictor (the target
                    # representation mean).  This is a training-fold statistic,
                    # not a test-set or cohort-specific hand-tuned switch.
                    null_err = torch.stack([
                        (H[k] - H[k].mean(0, keepdim=True)).pow(2).mean()
                        for k in others
                    ]).mean()
                    probe_err = xc.mean()
                    quality = (1.0 - probe_err / (null_err + EPS)).clamp(0.0, 1.0)
                self.cross_quality[i] = quality
        if was_training:
            self.train()

    def forward(self, xs, mask=None):
        """xs: dict m -> (N,d_m);  mask: (N,M) 1=可用 0=缺失"""
        N = next(iter(xs.values())).shape[0]
        dev = next(iter(xs.values())).device
        if mask is None:
            mask = torch.ones(N, self.M, device=dev)
        # ---- C1 缺失中性硬规则: 自动检测"全零/常数"输入, 视为缺失 ----
        # 重构误差 z_m 在常数输入上不可信(可能输出零/常数 -> 误差≈0, 但实际该路已不可用)。
        # 用样本级方差判定: z-scored 特征正常 std≈1, 阈值 1e-4 只命中真正的"全没了"。
        if self.c1_rule:
            for mi_, m in enumerate(self.mods):
                x = xs[m]
                sample_var = x.var(dim=1)                        # (N,)
                is_constant = (sample_var < 1e-4).float()       # (N,) 1=常数=缺失
                mask[:, mi_] = mask[:, mi_] * (1.0 - is_constant)

        H, E, REC, ENERGY, ENERGY_EXCESS = {}, {}, {}, {}, {}
        for m in self.mods:
            h, e, rec = self.branches[m](xs[m])
            H[m], E[m], REC[m] = h, e, rec
            ENERGY[m] = xs[m].pow(2).mean(dim=1, keepdim=True)

        # ---- 跨模态预测误差矩阵 (可靠性通道 B 与新颖度共用, 只算一次) ----
        CE = {}
        for m in self.mods:
            tgt = H[m].detach()
            for k in self.mods:
                if k == m:
                    continue
                CE[(k, m)] = ((self.cross[f"{k}->{m}"](H[k]) - tgt) ** 2).mean(dim=1, keepdim=True)

        # Optional matched-patient contrastive consistency.  Unlike a raw
        # cross-MSE probe, this explicitly compares the true patient pairing
        # with in-batch negatives, which is the signal destroyed by shuffle.
        CONS = {}
        consistency_loss = torch.tensor(0.0, device=dev)
        if self.contrastive_consistency and self.M > 1:
            Zc = {m: F.normalize(self.consistency_proj[m](H[m]), dim=1)
                  for m in self.mods}
            labels = torch.arange(N, device=dev)
            pair_losses = []
            for m in self.mods:
                vals = []
                for k in self.mods:
                    if k == m:
                        continue
                    k_mask = mask[:, self.mods.index(k)].unsqueeze(1)
                    vals.append((1.0 - (Zc[m] * Zc[k]).sum(-1, keepdim=True)) * k_mask)
                    if self.training:
                        logits = (Zc[m] @ Zc[k].t()) / max(self.consistency_temp, 1e-3)
                        pair_losses.extend([F.cross_entropy(logits, labels),
                                            F.cross_entropy(logits.t(), labels)])
                CONS[m] = torch.stack(vals, 0).sum(0) / (
                    torch.stack([mask[:, self.mods.index(k)].unsqueeze(1)
                                 for k in self.mods if k != m], 0).sum(0) + EPS)
            if pair_losses:
                consistency_loss = torch.stack(pair_losses).mean()
        else:
            CONS = {m: torch.zeros(N, 1, device=dev) for m in self.mods}

        # ---- V2 可靠性通道 B: 跨模态"解释力" ----
        # XC[m] = mean_k err(m->k): 用 m 的表示去预测**别的**模态, 预测得越差说明 m 越可疑。
        #
        # 方向很关键 —— 首版写成 min_k err(k->m)("别人能否预测出我")是错的:
        #   污染会让 m 的表示塌缩到均值附近, 反而**更容易**被别人预测,
        #   实测 featshuf 后该量从 7.63 降到 0.89 (信号反向)。
        # 反过来看"我还能不能解释别人"就单调了: 模态被打乱/清零后, 它与其他组学共享的
        # 信息被摧毁, 预测别人必然变差, 且塌缩成常数时误差=目标方差, 一定高于训练水平。
        XC = {}
        if self.cross_reliability:
            for mi_, m in enumerate(self.mods):
                others = [k for k in self.mods if k != m]
                if not others:
                    XC[m] = torch.zeros(N, 1, device=dev)
                    continue
                if self.contrastive_consistency:
                    XC[m] = CONS[m]
                    continue
                cols, wts = [], []
                for k in others:
                    mk = mask[:, self.mods.index(k)].unsqueeze(1)
                    cols.append(CE[(m, k)] * mk)          # 缺失的目标模态不计入
                    wts.append(mk)
                XC[m] = torch.stack(cols, 0).sum(0) / (torch.stack(wts, 0).sum(0) + EPS)

        # ---- 可靠性 r ----
        R = {}
        if self.triggered_r:
            # 触发式: 干净(zs<=tau) gate≈1(等权); 重建误差显著越界才降权。
            # 不叠加预后先验, 修复 M5 "灌垃圾后 r 没掉" 的 bug。
            # 在对数空间算 gate: log_g = -relu(zs-tau)*k <= 0。
            # 直接 exp 在重度污染时会下溢成 0, 导致相对归一化失效, 故先在 log 空间做减法。
            log_g = []
            for mi_, m in enumerate(self.mods):
                b = self.branches[m]
                if self.robust_stats:
                    zs = (REC[m] - b.rec_med) / (b.rec_iqr + EPS)     # 稳健标准分(抗重尾)
                else:
                    zs = (REC[m] - b.rec_mean) / (b.rec_var.sqrt() + EPS)
                excess = torch.relu(zs - self.tau[mi_])
                if self.cross_reliability:
                    # 通道 B: 跨模态解释力稳健标准分。用软权重融合，避免
                    # 一个小样本队列的 cross probe 噪声直接覆盖重构通道。
                    zx = (XC[m] - self.xcons_med[mi_]) / (self.xcons_iqr[mi_] + EPS)
                    cross_excess = torch.relu(zx - self.tau_x[mi_])
                    if not self.training:
                        cross_excess = cross_excess * self.cross_quality[mi_]
                    excess = excess + self.cross_weight * cross_excess
                if self.evid_drop and not self.training:
                    # 通道 C: 证据塌缩(批级)。以"相对训练水平的下降百分比"计量,
                    # 每掉 10% 记 1 个单位 -> tau_d=1.5 即"掉超过 15% 才报警"。
                    # 用样本级分位距当分母不行: 证据量样本间方差极大, 会把批级偏移稀释掉。
                    s_b = E[m].sum(1).median()
                    zd = (self.evid_med[mi_] - s_b) / (0.1 * self.evid_med[mi_].abs() + EPS)
                    excess = torch.maximum(excess, torch.relu(zd - self.tau_d[mi_]))
                e_excess = torch.zeros_like(excess)
                if self.energy_reliability and not self.training:
                    # Two-sided robust deviation: large positive energy catches
                    # additive/scale OOD; large negative energy catches wipeout.
                    z_hi = ((ENERGY[m] - self.energy_med[mi_]) /
                            (self.energy_hi_scale[mi_] + EPS))
                    if self.energy_low_guard:
                        low_floor = self.energy_med[mi_] * self.energy_low_frac
                        # A direct fractional collapse score avoids the
                        # generic z-score threshold: energy==0 must produce a
                        # positive signal even when tau_e is 1.0.
                        low_excess = torch.relu(
                            1.0 - ENERGY[m] / (low_floor.abs() + EPS))
                    else:
                        z_lo = ((self.energy_med[mi_] - ENERGY[m]) /
                                (self.energy_lo_scale[mi_] + EPS))
                        low_excess = torch.relu(z_lo - self.tau_e[mi_])
                    e_excess = torch.maximum(
                        torch.relu(z_hi - self.tau_e[mi_]),
                        low_excess)
                    excess = excess + self.energy_weight * e_excess
                ENERGY_EXCESS[m] = e_excess
                log_g.append(-excess * self.k[mi_].abs())               # (N,1)
            LG = torch.stack(log_g, 0)                                   # (M,N,1)
            Z = mask.t().unsqueeze(-1)                                   # (M,N,1) 可用性
            if self.r_mode == "rel":
                # 相对可靠性: 以当前样本上"可用模态里最干净的那个"为参照做归一化(log 空间减 max)。
                # 全脏时大家一起变脏 -> 归一化后都≈1 -> 自动退回等权(不劣于基线);
                # 部分脏时脏的一方被压到 ~r_floor。
                NEG = torch.finfo(LG.dtype).min / 4
                lg_m = torch.where(Z > 0, LG, torch.full_like(LG, NEG)).amax(0, keepdim=True)
                G = torch.exp((LG - lg_m).clamp(max=0.0))                # 最干净的那个 = 1
                G = self.r_floor + (1.0 - self.r_floor) * G
            else:
                G = torch.exp(LG.clamp(min=-60.0))
            G = G * Z
            for mi_, m in enumerate(self.mods):
                R[m] = G[mi_] if self.use_r else torch.ones_like(REC[m])
        else:
            for mi_, m in enumerate(self.mods):
                b = self.branches[m]
                zs = (REC[m] - b.rec_mean) / (b.rec_var.sqrt() + EPS)
                # 旧版(M5): 重建误差标准分 + 可学预后先验偏置
                z = -b.r_scale.abs() * zs + b.r_bias + self.r_prior[mi_]
                R[m] = torch.sigmoid(z) if self.use_r else torch.ones_like(REC[m])

        # ---- 新颖度 n (非对称冗余降权): 只在"更可靠的模态已覆盖我"时才扣我 ----
        # 原版对冗余对两方都等扣(对称), 会误伤有用信号; 改为把冗余"记在可靠性较低的一方",
        # 从而保留更有价值的那一份(满足"冗余不增信"公理且不误伤判别力)。
        Nv, cross_loss = {}, 0.0
        for mi_, m in enumerate(self.mods):
            others = [k for k in self.mods if k != m]
            if not others or not self.use_n:
                Nv[m] = torch.ones(N, 1, device=dev)
                continue
            charges = []
            for k in others:
                err = CE[(k, m)]                       # 复用上面算好的跨模态预测误差
                ki = self.mods.index(k)
                mk = mask[:, ki].unsqueeze(1)
                err_eff = err * mk + (1 - mk) * 1e3          # 缺失模态视作不可预测
                if self.training:
                    with torch.no_grad():
                        valid = err_eff[err_eff < 1e2]
                        if valid.numel() > 0:
                            self.nov_scale[mi_] = (0.9 * self.nov_scale[mi_]
                                                   + 0.1 * valid.mean().clamp(min=EPS))
                scale = self.nov_scale[mi_] + EPS
                redundancy = torch.exp(-err_eff / scale)      # k 越能预测 m, 越接近 1
                # 仅当 k 比 m 更可靠时, 才把这份冗余"记在 m 头上"
                gate = torch.sigmoid((R[k] - R[m]) * 4.0)
                charges.append(redundancy * gate)
                cross_loss = cross_loss + (err * mk).mean()
            charge_m = torch.stack(charges, 0).max(0).values   # 取最严重的竞争者
            Nv[m] = (1.0 - charge_m.clamp(0.0, 0.95)).clamp(0.05, 1.0)

        # ---- 融合 ----
        e_stack = torch.stack([E[m] for m in self.mods], 0)            # (M,N,K)
        r_stack = torch.stack([R[m] for m in self.mods], 0)            # (M,N,1)
        n_stack = torch.stack([Nv[m] for m in self.mods], 0)           # (M,N,1)
        z_stack = mask.t().unsqueeze(-1)                               # (M,N,1)
        w = z_stack * r_stack * n_stack
        e_fused = (w * e_stack).sum(0)                                 # (N,K)
        h_stack = torch.stack([H[m] for m in self.mods], 0)             # (M,N,d_rep)
        h_fused = (w * h_stack).sum(0) / (w.sum(0) + EPS)               # (N,d_rep)
        risk_direct = self.risk_head(h_fused).squeeze(1)
        alpha = e_fused + 1.0
        S = alpha.sum(1, keepdim=True)
        p = alpha / S
        u = self.n_cls / S                                             # 认知不确定性

        if self.decompose:
            # ---- 冲突 (只报告) ----
            p_m = e_stack + 1.0
            p_m = p_m / p_m.sum(-1, keepdim=True)                      # (M,N,K)
            Uc = torch.zeros(N, 1, device=dev)
            Uc_raw = torch.zeros(N, 1, device=dev)
            for i in range(self.M):
                for j in range(i + 1, self.M):
                    mij = (z_stack[i] * z_stack[j])
                    js = _js(p_m[i], p_m[j])
                    Uc = Uc + r_stack[i] * r_stack[j] * js * mij
                    # Keep semantic disagreement before reliability
                    # down-weighting as a separate diagnostic.  The weighted
                    # conflict is useful for fusion accounting, but it can
                    # hide conflict caused by a corrupted modality.
                    Uc_raw = Uc_raw + js * mij
            denom = max(self.M * (self.M - 1) / 2, 1)
            Uc = Uc / denom
            Uc_raw = Uc_raw / denom
        else:
            Uc = torch.zeros(N, 1, device=dev)
            Uc_raw = torch.zeros(N, 1, device=dev)

        if self.decompose:
            # ---- 可解释 4 路不确定性分解 (ignorance / data / conflict / missing) ----
            # 预测分布总不确定性 H(E[p]) 可分解为:
            #   aleatoric (data) = E_p[-log p] = Σ_k p_k (ψ(S) - ψ(α_k))   (固有噪声)
            #   epistemic (ignorance) = H(E[p]) - aleatoric                (没见过/证据少)
            # 冲突 conflict = Uc (跨模态 JS 加权, 已算)
            # 缺失 missing = -log(平均可靠性)  某模态不可信 → 缺失不确定性升高
            psi_S = torch.digamma(S.squeeze(-1))                      # (N,)
            psi_a = torch.digamma(alpha)                             # (N,K)
            u_data = (p * (psi_S.unsqueeze(-1) - psi_a)).sum(-1, keepdim=True).clamp(min=0.0)  # (N,1)
            ent_mean = -(p * torch.log(p + EPS)).sum(-1, keepdim=True)   # (N,1)
            u_ign = (ent_mean - u_data).clamp(min=0.0)               # (N,1)
            u_conf = Uc                                              # (N,1)
            mean_r = (r_stack * z_stack).sum(0) / (z_stack.sum(0) + EPS)  # (N,1)
            # Include the explicit available-modality fraction.  Averaging
            # only over available modalities made wipeout paradoxically look
            # less uncertain because the missing modality vanished from the
            # denominator.
            availability = z_stack.mean(0)
            u_miss = (-torch.log((availability * mean_r).clamp(min=1e-3))).clamp(min=0.0)
        else:
            # 不含 4 路分解: 风险/门控不受影响(仍鲁棒), 但无 u_* 分量 -> 无选择性预测能力
            sh = (N, 1)
            u_data = torch.zeros(sh, device=dev)
            u_ign = torch.zeros(sh, device=dev)
            u_conf = Uc
            Uc_raw = torch.zeros(sh, device=dev)
            u_miss = torch.zeros(sh, device=dev)

        return {
            "alpha": alpha, "p": p, "u": u, "conflict": Uc,
            "conflict_raw": Uc_raw,
            "h_fused": h_fused, "risk_direct": risk_direct,
            "u_data": u_data, "u_ign": u_ign, "u_miss": u_miss,
            "r": r_stack, "n": n_stack, "rec": torch.stack([REC[m] for m in self.mods], 0),
            "energy": torch.stack([ENERGY[m] for m in self.mods], 0),
            "energy_excess": torch.stack([ENERGY_EXCESS.get(m, torch.zeros_like(REC[m]))
                                           for m in self.mods], 0),
            "cross_loss": cross_loss if torch.is_tensor(cross_loss) else torch.tensor(0.0, device=dev),
            "consistency_loss": consistency_loss,
            "consistency": torch.stack([CONS[m] for m in self.mods], 0),
            "e_stack": e_stack,
        }

    def risk(self, out):
        """排序用风险分：证据对数比 log(α_high / α_low)。单调、无饱和, 适合 Cox 偏似然。
        (不确定性 u 单独输出, 用于 risk-coverage 弃权分析, 不混进排序分。)"""
        a = out["alpha"]
        evidence_risk = torch.log(a[:, 1] + EPS) - torch.log(a[:, 0] + EPS)
        if self.risk_mode == "direct":
            return out["risk_direct"]
        if self.risk_mode == "hybrid":
            gain = self.risk_gain
            # The direct head is useful only when the cross-modal probe has
            # demonstrable training-fold signal.  This makes the hybrid head
            # self-calibrating instead of imposing one global mixture on
            # weakly related or small multi-omics cohorts.
            if self.cross_reliability and not self.training:
                gain = gain * self.cross_quality.mean().clamp(0.0, 1.0)
            return evidence_risk + gain * out["risk_direct"]
        return evidence_risk


def _js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + EPS) - torch.log(b + EPS))).sum(-1, keepdim=True)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).clamp(min=0.0)


# ---------------- 损失 ----------------
def cox_ph_loss(risk, time, event):
    """Breslow 负偏似然 (可微)。risk: (N,)"""
    order = torch.argsort(time, descending=True)
    r = risk[order]
    ev = event[order].float()
    log_cum = torch.logcumsumexp(r, dim=0)
    ll = (r - log_cum) * ev
    d = ev.sum()
    return -ll.sum() / torch.clamp(d, min=1.0)


def evidence_reg(alpha, target_onehot, lam=1.0):
    """EDL 正则: 把"错误类"上的证据往 Dirichlet(1) 拉。"""
    alpha_t = target_onehot + (1 - target_onehot) * alpha
    K = alpha.shape[1]
    ones = torch.ones_like(alpha_t)
    S = alpha_t.sum(1, keepdim=True)
    kl = (torch.lgamma(S.squeeze(1)) - torch.lgamma(torch.tensor(float(K), device=alpha.device))
          - torch.lgamma(alpha_t).sum(1)
          + ((alpha_t - ones) * (torch.digamma(alpha_t) - torch.digamma(S))).sum(1))
    return lam * kl.mean()
