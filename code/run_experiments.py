#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DenseShield - full experimental pipeline (reproduces all tables of the paper).
Run:  python3 run_experiments.py
Requires: denseshield.py and the three public datasets (see README).
Outputs: results_summary.json + prints every metric used in the paper.
"""
import json, time, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from denseshield import (SFC, density_weights, MalMLP, train_mlp, eval_model,
                         pgd_evasion_binary, pgd_evasion_cont, attack_success_rate,
                         train_mlp_dbat, dpur_purify, saliency_trigger_search,
                         uncertainty_score, dmla_update)

rng = np.random.RandomState(42)
OUT = {}

def aut_f1(values):
    """Trapezoidal mean of adjacent monthly F1 values (paper Eq. AUT)."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError('AUT requires at least two monthly F1 values')
    return float(np.mean((values[:-1] + values[1:]) / 2))

# ---------------------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------------------
dr = pd.read_csv('../dataset/drebin215.csv')
Xd = dr.drop(columns=['class']).values.astype(np.float32)
yd = dr['class'].values.astype(np.int64)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=42, stratify=yd)

cl = pd.read_csv('../dataset/ClaMP_Integrated-5184.csv').drop(columns=['packer_type']).dropna()
Xc_all = np.log1p(np.clip(cl.drop(columns=['class']).values.astype(np.float32), 0, None))
yc_all = cl['class'].values.astype(np.int64)
Xctr, Xcte, yctr, ycte = train_test_split(Xc_all, yc_all, test_size=0.25, random_state=42, stratify=yc_all)
sc = StandardScaler().fit(Xctr)
Xctr_s, Xcte_s = sc.transform(Xctr), sc.transform(Xcte)

# ---------------------------------------------------------------------
# 2. SFC + Density weights (Drebin)
# ---------------------------------------------------------------------
sfc = SFC(tau=0.01, n_hash=24).fit(Xtr)
Xtr_c, Xte_c = sfc.transform(Xtr), sfc.transform(Xte)
w_tr = density_weights(Xtr_c, k=15, lam=0.6)
print(f"SFC: {Xtr.shape[1]} -> {Xtr_c.shape[1]} dims")

# ---------------------------------------------------------------------
# 3. Clean performance (RQ1) - Drebin baselines
# ---------------------------------------------------------------------
m_lr = LogisticRegression(max_iter=1000).fit(Xtr_c, ytr)
m_rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(Xtr_c, ytr)
m_gb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=63,
                          n_jobs=-1, random_state=42, verbose=-1).fit(Xtr_c, ytr)
m_vanilla = train_mlp(Xtr_c, ytr)
m_full = train_mlp_dbat(Xtr_c, ytr, w=w_tr, at_flips=12)      # DenseShield

def eval_sk(m, X, y):
    p = m.predict_proba(X)[:, 1]
    return accuracy_score(y, p > 0.5), f1_score(y, p > 0.5), roc_auc_score(y, p)

OUT['clean_drebin'] = {
    'LR': eval_sk(m_lr, Xte_c, yte), 'RF': eval_sk(m_rf, Xte_c, yte),
    'LightGBM': eval_sk(m_gb, Xte_c, yte),
    'VanillaMLP': eval_model(m_vanilla, Xte_c, yte)[:3],
    'DenseShield': eval_model(m_full, Xte_c, yte)[:3]}
print('RQ1 Drebin:', {k: tuple(round(float(x), 4) for x in v) for k, v in OUT['clean_drebin'].items()})

# ---------------------------------------------------------------------
# 4. Evasion robustness (RQ2) - additive PGD, budgets 0..20
# ---------------------------------------------------------------------
m_at = train_mlp_dbat(Xtr_c, ytr, at_flips=8)                  # AT without density
mal_te = Xte_c[yte == 1]
budgets = [0, 2, 4, 6, 8, 10, 15, 20]
ev = {}
for name, mdl in [('Vanilla MLP', m_vanilla), ('AT', m_at), ('DenseShield', m_full)]:
    ev[name] = [0.0 if k == 0 else
                attack_success_rate(mdl, pgd_evasion_binary(mdl, mal_te, max_flips=k))
                for k in budgets]
OUT['evasion_drebin'] = {'budgets': budgets, **ev}
print('RQ2 Drebin evasion:', {k: [round(float(x), 3) for x in v] for k, v in ev.items()})

# transferability: AEs from vanilla surrogate -> RF / LightGBM
Xa_tr = pgd_evasion_binary(m_vanilla, mal_te, max_flips=10)
OUT['transfer_asr_k10'] = {'RF': float((m_rf.predict_proba(Xa_tr)[:, 1] < 0.5).mean()),
                           'LightGBM': float((m_gb.predict_proba(Xa_tr)[:, 1] < 0.5).mean())}
print('RQ2 transfer:', OUT['transfer_asr_k10'])

# ---------------------------------------------------------------------
# 5. Backdoor attack + purification (RQ3) - Drebin sparse trigger
# ---------------------------------------------------------------------
freq215 = Xtr.mean(axis=0)
sparse_idx = np.where(freq215 < 0.01)[0]
trig = sparse_idx[rng.choice(len(sparse_idx), 5, replace=False)]
mal_tr_idx = np.where(ytr == 1)[0]
pidx = rng.choice(mal_tr_idx, int(0.02 * len(Xtr)), replace=False)
Xtr_bd = Xtr.copy(); Xtr_bd[pidx[:, None], trig] = 1.0
ytr_bd = ytr.copy(); ytr_bd[pidx] = 0
m_bd = train_mlp(Xtr_bd, ytr_bd, epochs=22, seed=7)
Xte_trig = Xte.copy(); Xte_trig[:, trig] = 1.0
mal_mask = yte == 1
def asr_trig(m):
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(torch.tensor(Xte_trig[mal_mask], dtype=torch.float32)), 1)[:, 1].numpy()
    return float((p < 0.5).mean())

clean_sub = rng.choice(len(Xtr), int(0.01 * len(Xtr)), replace=False)
Xs215, ys215 = Xtr[clean_sub], ytr[clean_sub]

from denseshield import dpur_purify
def purify_finetune(model, Xc, yc, ft_epochs=40, lr=1e-3):
    m2 = copy.deepcopy(model)
    opt = torch.optim.Adam(m2.parameters(), lr=lr)
    X_t = torch.tensor(Xc, dtype=torch.float32); y_t = torch.tensor(yc)
    for _ in range(ft_epochs):
        m2.train(); perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 128):
            idx = perm[i:i + 128]
            loss = torch.nn.functional.cross_entropy(m2(X_t[idx]), y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m2

def purify_bn(model, Xc, yc, ft_epochs=40, lr=1e-3):
    m2 = copy.deepcopy(model)
    for mod in m2.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.reset_running_stats(); mod.momentum = None
    m2.train()
    with torch.no_grad():
        for i in range(0, len(Xc), 256):
            m2(torch.tensor(Xc[i:i + 256], dtype=torch.float32))
    for n_, p in m2.named_parameters():
        p.requires_grad = any(f'net.{i}.' in n_ for i in [1, 4, 7])
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m2.parameters()), lr=lr)
    X_t = torch.tensor(Xc, dtype=torch.float32); y_t = torch.tensor(yc)
    for _ in range(ft_epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 128):
            idx = perm[i:i + 128]
            loss = torch.nn.functional.cross_entropy(m2(X_t[idx]), y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m2

m_ft = purify_finetune(m_bd, Xs215, ys215)
m_pbp = purify_bn(m_bd, Xs215, ys215)
m_dpur, cand = dpur_purify(m_bd, Xs215, ys215, freq215, tau=0.01, top_m=8)
OUT['backdoor_drebin'] = {
    'ASR_bd': asr_trig(m_bd), 'F1_bd': eval_model(m_bd, Xte, yte)[1],
    'ASR_ft': asr_trig(m_ft), 'F1_ft': eval_model(m_ft, Xte, yte)[1],
    'ASR_pbp': asr_trig(m_pbp), 'F1_pbp': eval_model(m_pbp, Xte, yte)[1],
    'ASR_dpur': asr_trig(m_dpur), 'F1_dpur': eval_model(m_dpur, Xte, yte)[1]}
print('RQ3 Drebin backdoor:', {k: round(v, 4) for k, v in OUT['backdoor_drebin'].items()})

# ---------------------------------------------------------------------
# 6. ClaMP: clean, evasion, backdoor (cross-domain generality)
# ---------------------------------------------------------------------
m_cl = train_mlp(Xctr_s, yctr)
w_ctr = density_weights(Xctr_s, k=10, lam=0.6)
m_cl_full = train_mlp(Xctr_s, yctr, w=w_ctr)   # density-boosted (DBAT-cont optional)
OUT['clean_clamp'] = {'VanillaMLP': eval_model(m_cl, Xcte_s, ycte)[:3],
                      'DenseShield': eval_model(m_cl_full, Xcte_s, ycte)[:3]}
mal_cte = Xcte_s[ycte == 1]
eps_list = [0.05, 0.1, 0.2, 0.3, 0.5]
OUT['evasion_clamp'] = {'eps': eps_list,
    'Vanilla': [attack_success_rate(m_cl, pgd_evasion_cont(m_cl, mal_cte, eps=e)) for e in eps_list]}
# ClaMP backdoor (continuous extreme-value trigger)
c_freq = (Xctr_s != 0).mean(axis=0)
trig_c = np.where(c_freq < 0.05)[0][:4]
rng2 = np.random.RandomState(11)
mal_ctr = np.where(yctr == 1)[0]
pidx_c = rng2.choice(mal_ctr, int(0.02 * len(Xctr_s)), replace=False)
Xctr_bd = Xctr_s.copy(); Xctr_bd[pidx_c[:, None], trig_c] = 5.0
yctr_bd = yctr.copy(); yctr_bd[pidx_c] = 0
m_cl_bd = train_mlp(Xctr_bd, yctr_bd, epochs=25, seed=7)
Xcte_trig = Xcte_s.copy(); Xcte_trig[:, trig_c] = 5.0
mal_cmask = ycte == 1
def asr_c(m):
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(torch.tensor(Xcte_trig[mal_cmask], dtype=torch.float32)), 1)[:, 1].numpy()
    return float((p < 0.5).mean())
cs = rng2.choice(len(Xctr_s), int(0.01 * len(Xctr_s)), replace=False)
m_cl_dpur, _ = dpur_purify(m_cl_bd, Xctr_s[cs], yctr[cs], c_freq, tau=0.05, top_m=8, trig_val=5.0)
OUT['backdoor_clamp'] = {'ASR_bd': asr_c(m_cl_bd), 'ASR_dpur': asr_c(m_cl_dpur)}
print('RQ3 ClaMP backdoor:', OUT['backdoor_clamp'])

# ---------------------------------------------------------------------
# 7. EMBER2018 concept drift (RQ4) - requires ember_proc.py preprocessing
# ---------------------------------------------------------------------
def run_drift():
    ember_proc = Path(__file__).resolve().parents[1] / 'dataset' / 'ember_proc'
    trE = np.load(ember_proc / 'train_m1_4.npz')
    XEtr, yEtr = trE['X'], trE['y']
    months = {m: np.load(ember_proc / f'test_m{m}.npz') for m in range(5, 11)}
    scE = StandardScaler().fit(np.log1p(np.clip(XEtr, 0, None)))
    prep = lambda X: scE.transform(np.log1p(np.clip(X, 0, None)))
    XEtr_s = prep(XEtr)
    mE = train_mlp(XEtr_s, yEtr, epochs=12)  # small MLP variant used in the paper
    f1s = [eval_model(mE, prep(months[m]['X']), months[m]['y'])[1] for m in range(5, 11)]
    res = {'No update': f1s}
    for q, ep in [(0.01, 8), (0.05, 4)]:
        mcur = copy.deepcopy(mE); cur = []
        for t in range(5, 11):
            Xm, ym = months[t]['X'], months[t]['y']
            cur.append(eval_model(mcur, prep(Xm), ym)[1])
            if t < 10:
                mcur = dmla_update(mcur, prep(Xm), ym, q=q, conf_thr=0.95, epochs=ep)
        res[f'DMLA {int(q*100)}%'] = cur
    return res

try:
    OUT['drift_ember'] = run_drift()
    print('RQ4 drift AUT:', {k: round(aut_f1(v), 4) for k, v in OUT['drift_ember'].items()})
except FileNotFoundError:
    print('EMBER preprocessed files not found - run ember_proc.py first.')

with open('results_summary.json', 'w') as f:
    json.dump(OUT, f, indent=1, default=float)
print('Saved results_summary.json')
