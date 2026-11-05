"""RRGEF estimator: fit / predict interface around the evidential fusion network."""
import numpy as np
import torch

from rrgef.training import train_fusion_ens, eval_fusion_ens, eval_fusion_ens_full


class _EvidentialFusionSurvival:
    """Shared estimator logic (frozen paper code; development flags kept)."""

    def __init__(self, r_floor=0.15, ensemble_small=True,
                 cross_reliability=False, deep_cross=False,
                 cross_weight=1.0,
                 tau_x_init=3.0,
                 cross_loss_weight=0.1,
                 contrastive_consistency=False, consistency_loss_weight=0.0,
                 risk_mode="evidence", risk_gain=1.0,
                 modality_dropout=0.0,
                 use_r=True, triggered_r=True, c1_rule=True,
                 small_sample_fallback=False,
                 force_ensemble=False,
                 bounded_evidence=False, robust_clip=False,
                 evid_drop=False,
                 energy_reliability=False, energy_weight=0.5, tau_e_init=1.0,
                 energy_low_guard=False, energy_low_frac=0.1,
                 pair_diagnostic=False, pair_pca_dim=8,
                 pair_ridge_alpha=10.0):
        self.mods_ = None
        self.r_floor = r_floor
        self.ensemble_small = ensemble_small
        self.cross_reliability = bool(cross_reliability)
        self.deep_cross = bool(deep_cross)
        self.cross_weight = float(cross_weight)
        self.tau_x_init = float(tau_x_init)
        self.cross_loss_weight = float(cross_loss_weight)
        self.contrastive_consistency = bool(contrastive_consistency)
        self.consistency_loss_weight = float(consistency_loss_weight)
        self.risk_mode = str(risk_mode)
        self.risk_gain = float(risk_gain)
        self.modality_dropout = float(modality_dropout)
        self.use_r = bool(use_r)
        self.triggered_r = bool(triggered_r)
        self.c1_rule = bool(c1_rule)
        self.small_sample_fallback = bool(small_sample_fallback)
        self.force_ensemble = bool(force_ensemble)
        self.bounded_evidence = bool(bounded_evidence)
        self.robust_clip = bool(robust_clip)
        self.evid_drop = bool(evid_drop)
        self.energy_reliability = bool(energy_reliability)
        self.energy_weight = float(energy_weight)
        self.tau_e_init = float(tau_e_init)
        self.energy_low_guard = bool(energy_low_guard)
        self.energy_low_frac = float(energy_low_frac)
        self.energy_diagnostic = False
        self.pair_diagnostic = bool(pair_diagnostic)
        self.pair_pca_dim = int(pair_pca_dim)
        self.pair_ridge_alpha = float(pair_ridge_alpha)

    def fit(self, X_dict, time, event):
        self.mods_ = sorted(X_dict.keys())
        small = len(time) < 250
        K = 3 if (self.force_ensemble or (small and self.ensemble_small)) else 1
        sp = dict(dropout=0.5, wd=1e-3, d_hid=32, d_rep=16) if small else {}
        safe_fallback = bool(small and self.small_sample_fallback)
        cross_enabled = self.cross_reliability and not safe_fallback
        risk_mode = "evidence" if safe_fallback else self.risk_mode
        risk_gain = 0.0 if safe_fallback else self.risk_gain
        cross_weight = 0.0 if safe_fallback else self.cross_weight
        contrastive_enabled = self.contrastive_consistency and not safe_fallback
        consistency_loss_weight = 0.0 if safe_fallback else self.consistency_loss_weight
        modality_dropout = 0.0 if safe_fallback else self.modality_dropout
        # Match the stable base configuration when the cross probe is not
        # statistically supported by the training fold.
        # Preserve the historical small-sample fallback for the full model
        # (w_cross=.1), while allowing an explicit no-consistency ablation to
        # keep its zero cross-loss setting under the same fallback.
        if safe_fallback and self.cross_loss_weight > 0.0:
            cross_loss_weight = 0.1
        else:
            cross_loss_weight = self.cross_loss_weight
        self.model_ = train_fusion_ens(
            X_dict, time, event, self.mods_, self.use_r, False, K,
            int(torch.initial_seed() % 100000),
            triggered_r=self.triggered_r, r_mode="rel", r_floor=self.r_floor,
            cross_reliability=cross_enabled,
            cross_weight=cross_weight,
            tau_x_init=self.tau_x_init,
            contrastive_consistency=contrastive_enabled,
            w_consistency=consistency_loss_weight,
            w_cross=cross_loss_weight,
            risk_mode=risk_mode,
            risk_gain=risk_gain,
            modality_dropout=modality_dropout,
            deep_cross=self.deep_cross,
            bounded_evidence=self.bounded_evidence,
            robust_clip=self.robust_clip,
            evid_drop=self.evid_drop,
            energy_reliability=self.energy_reliability,
            energy_weight=self.energy_weight,
            tau_e_init=self.tau_e_init,
            energy_low_guard=self.energy_low_guard,
            energy_low_frac=self.energy_low_frac,
            c1_rule=self.c1_rule,
            **sp)
        if getattr(self, "energy_diagnostic", False):
            # Calibrate the diagnostic head outside the prognostic model.  It
            # uses only the training-fold feature matrix and therefore leaves
            # the fitted risk path byte-for-byte equivalent to the locked
            # cross-modal candidate.
            self.energy_med_ = np.asarray([
                np.median(np.mean(np.asarray(X_dict[m], dtype=float) ** 2, axis=1))
                for m in self.mods_
            ], dtype=float)
            self.energy_lo_scale_ = np.asarray([
                max(self.energy_med_[i] - np.quantile(
                    np.mean(np.asarray(X_dict[m], dtype=float) ** 2, axis=1), 0.01),
                    1e-8)
                for i, m in enumerate(self.mods_)
            ], dtype=float)
            self.energy_hi_scale_ = np.asarray([
                max(np.quantile(
                    np.mean(np.asarray(X_dict[m], dtype=float) ** 2, axis=1), 0.99)
                    - self.energy_med_[i], 1e-8)
                for i, m in enumerate(self.mods_)
            ], dtype=float)
        if self.pair_diagnostic:
            # Self-supervised matched-patient readout.  It is calibrated only
            # on the training fold and is never consumed by the risk head.
            from rrgef.diagnostics import fit_pair_state
            self.pair_state_ = fit_pair_state(
                X_dict, self.mods_, self.pair_pca_dim, self.pair_ridge_alpha)
        self.K_ = K

    def predict_risk(self, X_dict):
        s, *_ = eval_fusion_ens(self.model_, X_dict, self.mods_)
        return s

    def predict_all(self, X_dict):
        """Return risk plus all readouts (no retraining).

        Keys used in the paper: ``risk`` (survival risk score), ``u_total``
        (evidence insufficiency u = 2 / (alpha_0 + alpha_1)), ``r_m``
        (per-modality relative reliability, shape (M, N), modality order
        given by ``self.mods_``), ``energy_excess`` (feature-energy
        diagnostic, (M, N)) and ``pair_gap`` / ``pair_gap_m`` /
        ``maha_pair_gap`` / ``maha_pair_gap_m`` (pairing diagnostics).
        """
        out = eval_fusion_ens_full(self.model_, X_dict, self.mods_)
        cons = np.asarray(out.get("consistency", np.zeros_like(out["r_m"])), dtype=float)
        if cons.ndim == 3:
            cons = cons[..., 0]
        energy = np.asarray(out.get("energy", np.zeros_like(out["r_m"])), dtype=float)
        energy_excess = np.asarray(
            out.get("energy_excess", np.zeros_like(out["r_m"])), dtype=float)
        if energy.ndim == 3:
            energy = energy[..., 0]
        if energy_excess.ndim == 3:
            energy_excess = energy_excess[..., 0]
        if getattr(self, "energy_diagnostic", False):
            # Readout-only calibrated energy/OOD signal.  The model itself is
            # trained with energy_reliability=False, so this branch cannot
            # alter risk or evidence fusion.
            energy = np.stack([
                np.mean(np.asarray(X_dict[m], dtype=float) ** 2, axis=1)
                for m in self.mods_
            ], axis=0)
            med = self.energy_med_[:, None]
            hi = self.energy_hi_scale_[:, None]
            z_hi = (energy - med) / (hi + 1e-8)
            if self.energy_low_guard:
                low_floor = med * self.energy_low_frac
                low_excess = np.maximum(0.0,
                                        1.0 - energy / (np.abs(low_floor) + 1e-8))
            else:
                z_lo = (med - energy) / (self.energy_lo_scale_[:, None] + 1e-8)
                low_excess = np.maximum(0.0, z_lo - self.tau_e_init)
            energy_excess = np.maximum(np.maximum(0.0, z_hi - self.tau_e_init),
                                       low_excess)
        # Keep a conservative trust readout separate from prognostic risk.
        # The diagnostic candidate adds a small, fixed contribution from the
        # calibrated feature-space OOD signal; it cannot change the C-index
        # while supporting selective prediction.
        u_miss = np.asarray(out["u_miss"], dtype=float)
        trust_energy_weight = float(getattr(self, "trust_energy_weight", 0.0))
        if trust_energy_weight > 0.0:
            u_trust = u_miss + trust_energy_weight * energy_excess.mean(axis=0)
        else:
            u_trust = u_miss.copy()
        result = {
            "risk": out["risk"],
            # This is the evidence-mass ignorance signal used for
            # risk-coverage.  It is deliberately distinct from the
            # entropy-based `u_ign` decomposition term.
            "u_total": out["u_total"],
            "u_ign": out["u_ign"],
            "u_data": out["u_data"],        # aleatoric
            "u_miss": out["u_miss"],        # missing/unreliable
            "u_trust": u_trust,
            "U_conflict": out["U_conflict"],
            "U_conflict_raw": out["U_conflict_raw"],
            "r_m": out["r_m"],              # (M,N)
            "n_m": out["n_m"],              # (M,N)
            "consistency": cons,            # (M,N), matched-patient mismatch
            "u_consistency": cons.mean(axis=0),
            "energy": energy,               # (M,N), standardized feature energy
            "energy_excess": energy_excess, # (M,N), calibrated two-sided OOD excess
        }
        if self.pair_diagnostic:
            from rrgef.diagnostics import score_pair_state, score_maha_pair_state
            pair_gap, pair_gap_m = score_pair_state(
                self.pair_state_, X_dict, self.mods_)
            result["pair_gap"] = pair_gap
            result["pair_gap_m"] = pair_gap_m
            maha_pair_gap, maha_pair_gap_m = score_maha_pair_state(
                self.pair_state_, X_dict, self.mods_)
            result["maha_pair_gap"] = maha_pair_gap
            result["maha_pair_gap_m"] = maha_pair_gap_m
        return result

    def predict_uncertainty(self, X_dict):
        """Evidence-insufficiency score u_total, shape (N,)."""
        return self.predict_all(X_dict)["u_total"]

    def predict_reliability(self, X_dict):
        """Per-modality relative reliability r_m, shape (M, N)."""
        return self.predict_all(X_dict)["r_m"]

    def predict_conflict(self, X_dict):
        """Reliability-weighted cross-modal conflict, shape (N,)."""
        return self.predict_all(X_dict)["U_conflict"]

    def predict_proba(self, X_dict):
        """Dirichlet mean probabilities, shape (N, 2)."""
        return eval_fusion_ens_full(self.model_, X_dict, self.mods_)["p"]


class RRGEF(_EvidentialFusionSurvival):
    """Relative reliability-guided evidential fusion (paper configuration).

    Usage::

        model = RRGEF()
        model.fit(X_train, time_train, event_train)   # X: {modality: (N, d)}
        risk = model.predict_risk(X_test)             # larger = higher risk
        out = model.predict_all(X_test)               # risk + readouts

    Inputs should already be fold-wise selected and standardized
    (see ``rrgef.preprocessing.select_fit_apply``).  Defaults reproduce the
    configuration reported in the paper: r_min = 0.05, modality dropout 0.30,
    consistency weight beta = 0.05, consistency loss weight 0.02, a
    three-model ensemble, and a compact small-sample configuration when the
    training fold has fewer than 250 patients.
    """

    def __init__(self, r_floor=0.05, ensemble_small=True, cross_weight=0.05,
                 risk_gain=0.0, risk_mode="evidence",
                 small_sample_fallback=True, tau_x_init=5.0,
                 modality_dropout=0.30, pair_diagnostic=True,
                 pair_pca_dim=8, pair_ridge_alpha=10.0,
                 use_r=True, triggered_r=True, c1_rule=True,
                 cross_reliability=True, contrastive_consistency=True,
                 cross_loss_weight=0.01, consistency_loss_weight=0.02):
        # Locked configuration after the raw/fold-safe sensitivity probe
        # (r_floor=.05, modality_dropout=.30):
        # contrastive consistency is deliberately a *bounded reliability
        # correction* (not a second prognostic head).  The direct head stays
        # available in FusionBaseline for a separately reported ablation, but
        # is disabled here because it was unstable on LIHC.
        super().__init__(r_floor=r_floor, ensemble_small=ensemble_small,
                         cross_reliability=cross_reliability, cross_weight=cross_weight,
                         tau_x_init=tau_x_init, cross_loss_weight=cross_loss_weight,
                         contrastive_consistency=contrastive_consistency,
                         consistency_loss_weight=consistency_loss_weight,
                         risk_mode=risk_mode, risk_gain=risk_gain,
                         modality_dropout=modality_dropout,
                         use_r=use_r, triggered_r=triggered_r, c1_rule=c1_rule,
                         small_sample_fallback=small_sample_fallback,
                         force_ensemble=True,
                         deep_cross=False,
                         pair_diagnostic=pair_diagnostic,
                         pair_pca_dim=pair_pca_dim,
                         pair_ridge_alpha=pair_ridge_alpha)
        # Absolute feature-energy is a readout-only sentinel.  It is calibrated
        # on the training fold and exposed through ``predict_all``; the model's
        # prognostic energy gate remains disabled, so this cannot change risk.
        self.energy_diagnostic = True
        self.energy_low_guard = True
