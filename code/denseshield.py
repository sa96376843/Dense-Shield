#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DenseShield: A Density-Centric Framework for Adversarially Robust,
Backdoor-Purified, and Drift-Adaptive Malware Detection.
============================================================================
Core library: SFC (Sparse Feature Compression), DBAT (Density-Boosted
Adversarial Training), D-PUR (saliency x rarity trigger search +
counter-example unlearning + BN recalibration), DMLA (drift-aware
minimal-label adaptation).

Datasets (all publicly available, downloaded separately):
  - DREBIN-215 : https://github.com/Malware-Hunter/SBSeg23-XAI (drebin215.csv.zip)
  - ClaMP-5184 : https://github.com/urwithajit9/ClaMP (dataset/ClaMP_Integrated-5184.csv)
  - EMBER2018  : https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# Stage 1 - Sparse Feature Compression (SFC)
# =====================================================================
class SFC:
    """Compress features whose relative frequency < tau into B hash bins."""
    def __init__(self, tau=0.01, n_hash=24, seed=42):
        self.tau, self.n_hash, self.seed = tau, n_hash, seed

    def fit(self, Xtr):
        freq = Xtr.mean(axis=0)
        self.freq_ = freq
        self.sparse_idx_ = np.where(freq < self.tau)[0]
        self.dense_idx_ = np.where(freq >= self.tau)[0]
        rng = np.random.RandomState(self.seed)
        self.hash_map_ = rng.randint(0, self.n_hash, size=len(self.sparse_idx_))
        return self

    def transform(self, X):
        dense_part = X[:, self.dense_idx_]
        sp = X[:, self.sparse_idx_]
        bins = np.zeros((X.shape[0], self.n_hash), dtype=np.float32)
        for b in range(self.n_hash):
            cols = np.where(self.hash_map_ == b)[0]
            if len(cols):
                bins[:, b] = np.clip(sp[:, cols].sum(axis=1), 0, 1)
        return np.concatenate([dense_part, bins], axis=1)


# =====================================================================
# Stage 2 - Density-Boosted weighting
# =====================================================================
def density_weights(X, k=15, lam=0.6, lo=0.2, hi=5.0):
    """w_i = clip((rho_max / rho_i)^lam), rho_i = k-NN density estimate."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    r_k = dists[:, -1]
    rho = k / (X.shape[1] * (r_k + 1e-6))
    w = (rho.max() / (rho + 1e-9)) ** lam
    w = w / w.mean()
    return np.clip(w, lo, hi).astype(np.float32)


# =====================================================================
# MLP backbone with BatchNorm (needed for post-training purification)
# =====================================================================
class MalMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, w=None, epochs=25, lr=1e-3, seed=42, model_cls=MalMLP):
    torch.manual_seed(seed)
    model = model_cls(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y)
    w_t = torch.tensor(w, dtype=torch.float32) if w is not None else None
    n, bs = len(X_t), 256
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = nn.functional.cross_entropy(model(X_t[idx]), y_t[idx], reduction='none')
            loss = (loss * w_t[idx]).mean() if w_t is not None else loss.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_model(model, X, y):
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor(X, dtype=torch.float32)), 1)[:, 1].numpy()
    pred = (proba > 0.5).astype(int)
    return accuracy_score(y, pred), f1_score(y, pred), roc_auc_score(y, proba), proba


# =====================================================================
# Evasion attacks (problem-space constrained)
# =====================================================================
def pgd_evasion_binary(model, X, max_flips=10, steps=25):
    """Additive PGD on binary feature space: only 0->1 flips (feature insertion)."""
    X_t = torch.tensor(X, dtype=torch.float32)
    X_adv = X_t.clone()
    for _ in range(steps):
        X_adv.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(X_adv), torch.zeros(len(X_adv), dtype=torch.long))
        grad = torch.autograd.grad(loss, X_adv)[0]
        with torch.no_grad():
            cand = (X_adv < 0.5).float()
            score = -grad * cand
            cur_flips = (X_adv - X_t).sum(dim=1)
            budget_left = (max_flips - cur_flips).clamp(min=0)
            vals, idxs = torch.topk(score, k=score.shape[1], dim=1)
            new = X_adv.clone()
            for r in range(X.shape[0]):
                for j in range(int(min(budget_left[r].item(), 2))):
                    if vals[r, j] > 0:
                        new[r, idxs[r, j]] = 1.0
            X_adv = new
    return X_adv.detach().numpy()


def pgd_evasion_cont(model, X, eps=0.3, steps=30):
    """l_inf PGD on continuous (standardised) feature space, targeting benign."""
    alpha = eps / steps * 2.5
    X_t = torch.tensor(X, dtype=torch.float32)
    X_adv = X_t.clone()
    for _ in range(steps):
        X_adv.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(X_adv), torch.zeros(len(X_adv), dtype=torch.long))
        g = torch.autograd.grad(loss, X_adv)[0]
        with torch.no_grad():
            X_adv = X_adv - alpha * torch.sign(g)
            X_adv = torch.max(torch.min(X_adv, X_t + eps), X_t - eps)
    return X_adv.detach().numpy()


def attack_success_rate(model, X_adv):
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(X_adv, dtype=torch.float32)), 1)[:, 1].numpy()
    return float((p < 0.5).mean())


# =====================================================================
# Density-Boosted Adversarial Training (DBAT)
# =====================================================================
def train_mlp_dbat(X, y, w=None, at_flips=12, epochs=25, lr=1e-3, seed=42):
    """Density-boosted adversarial training on binary feature space."""
    torch.manual_seed(seed)
    model = MalMLP(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y)
    w_t = torch.tensor(w, dtype=torch.float32) if w is not None else torch.ones(len(X_t))
    for ep in range(epochs):
        if ep % 2 == 1:  # inject adversarial examples every other epoch
            mal = torch.where(y_t == 1)[0]
            sub = mal[torch.randperm(len(mal))[:2048]]
            Xa = pgd_evasion_binary(model, X_t[sub].numpy(), max_flips=at_flips, steps=12)
            X_aug = torch.cat([X_t, torch.tensor(Xa, dtype=torch.float32)])
            y_aug = torch.cat([y_t, torch.ones(len(sub), dtype=torch.long)])
            w_aug = torch.cat([w_t, w_t[sub] * 1.5])
        else:
            X_aug, y_aug, w_aug = X_t, y_t, w_t
        model.train()
        perm = torch.randperm(len(X_aug))
        for i in range(0, len(X_aug), 256):
            idx = perm[i:i + 256]
            loss = (nn.functional.cross_entropy(model(X_aug[idx]), y_aug[idx], reduction='none') * w_aug[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# =====================================================================
# Stage 3 - D-PUR: saliency x rarity trigger search + unlearning
# =====================================================================
def saliency_trigger_search(model, Xc):
    """Per-feature saliency of the benign logit on the purification set."""
    model.eval()
    X_t = torch.tensor(Xc, dtype=torch.float32, requires_grad=True)
    loss = model(X_t)[:, 0].sum()
    return torch.autograd.grad(loss, X_t)[0].abs().mean(dim=0).numpy()


def dpur_purify(model, Xc, yc, freq_vec, tau=0.01, top_m=8, trig_val=1.0,
                ft_epochs=15, lr=8e-4):
    """D-PUR purification.
    1) suspicion_j = saliency_j * 1[freq_j < tau];  top-m features = candidate trigger
    2) counter-examples: clean malware + candidate trigger set ON, correct label
    3) BN-stat recalibration + density-free fine-tuning on clean + counter-examples
    """
    m2 = copy.deepcopy(model)
    g = saliency_trigger_search(m2, Xc) * (freq_vec < tau)
    cand = np.argsort(-g)[:top_m]
    mal_c = np.where(yc == 1)[0]
    Xce = Xc[mal_c].copy(); Xce[:, cand] = trig_val
    Xall = np.concatenate([Xc, Xce]); yall = np.concatenate([yc, yc[mal_c]])
    wall = np.concatenate([np.ones(len(Xc)), 2.0 * np.ones(len(Xce))]).astype(np.float32)
    for mod in m2.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.reset_running_stats(); mod.momentum = None
    m2.train()
    with torch.no_grad():
        for i in range(0, len(Xall), 256):
            m2(torch.tensor(Xall[i:i + 256], dtype=torch.float32))
    opt = optim.Adam(m2.parameters(), lr=lr, weight_decay=5e-4)
    X_t = torch.tensor(Xall, dtype=torch.float32)
    y_t = torch.tensor(yall); w_t = torch.tensor(wall)
    for _ in range(ft_epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 128):
            idx = perm[i:i + 128]
            loss = nn.functional.cross_entropy(m2(X_t[idx]), y_t[idx], reduction='none')
            (loss * w_t[idx]).mean().backward(); opt.step(); opt.zero_grad()
    return m2, cand


# =====================================================================
# Stage 4/5 - Drift detection & minimal-label adaptation (DMLA)
# =====================================================================
def uncertainty_score(model, X):
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(X, dtype=torch.float32)), 1).numpy()
    return 1.0 - p.max(axis=1)


def dmla_update(model, X_new_all, y_true, q=0.01, conf_thr=0.95,
                replay_X=None, replay_y=None, epochs=3, lr=3e-4):
    """One monthly DMLA update: label top-q% uncertain samples, pseudo-label
    the confident remainder, fine-tune with replay."""
    m2 = copy.deepcopy(model)
    s = uncertainty_score(m2, X_new_all)
    n_sel = max(1, int(q * len(X_new_all)))
    sel = np.argsort(-s)[:n_sel]
    Xnew, ynew = X_new_all[sel], y_true[sel]
    with torch.no_grad():
        pcur = torch.softmax(m2(torch.tensor(X_new_all, dtype=torch.float32)), 1).numpy()
    conf = pcur.max(axis=1) > conf_thr
    Xnew = np.concatenate([Xnew, X_new_all[conf]])
    ynew = np.concatenate([ynew, pcur[conf].argmax(axis=1)])
    X_t = torch.tensor(Xnew, dtype=torch.float32); y_t = torch.tensor(ynew)
    if replay_X is not None:
        X_t = torch.cat([X_t, torch.tensor(replay_X, dtype=torch.float32)])
        y_t = torch.cat([y_t, torch.tensor(replay_y)])
    opt = optim.Adam(m2.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        m2.train()
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            idx = perm[i:i + 256]
            loss = nn.functional.cross_entropy(m2(X_t[idx]), y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m2
