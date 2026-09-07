#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DenseShield - empirical figure generation.
Reproduces the empirical plots from cached experiment outputs
(results_summary.json). All plots use English labels (IEEE style). The
framework schematic is maintained with the manuscript rather than this
code-and-data repository.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('results_summary.json'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.titlesize': 9.5, 'axes.labelsize': 9,
                     'legend.fontsize': 7.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'figure.dpi': 200,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': ':'})
C = {'blue': '#2166ac', 'red': '#b2182b', 'green': '#1b7837',
     'orange': '#e08214', 'purple': '#762a83', 'grey': '#636363', 'teal': '#1f9e89'}

# ---------------- Fig.5: evasion ASR curves ----------------
ev = R['evasion_drebin']; budgets = ev['budgets']
fig, ax = plt.subplots(figsize=(4.4, 3.0))
sty = {'Vanilla MLP': (C['blue'], 'o'), 'AT': (C['orange'], 's'), 'DenseShield': (C['red'], '^')}
for name, (c, mk) in sty.items():
    ax.plot(budgets, ev[name], color=c, marker=mk, ms=4, lw=1.6, label=name)
ax.set_xlabel('Feature-insertion budget $k$'); ax.set_ylabel('Attack Success Rate')
ax.legend(loc='center right'); ax.set_ylim(-0.02, 1.05)
fig.tight_layout(); fig.savefig('fig5_evasion.pdf')

# ---------------- Fig.6: backdoor purification ----------------
bd = R['backdoor_drebin_mean']
fig, ax = plt.subplots(figsize=(4.4, 3.0))
methods = ['No defense', 'Fine-tuning', 'PBP', 'D-PUR (ours)']
vals = [bd['ASR_bd'] * 100, bd['ASR_ft'] * 100, bd['ASR_pbp'] * 100, bd['ASR_dpur'] * 100]
cols = [C['grey'], C['blue'], C['orange'], C['red']]
bars = ax.bar(methods, vals, color=cols, alpha=0.88)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, f'{v:.1f}', ha='center', fontsize=7.5)
ax.set_ylabel('ASR (%)'); ax.set_ylim(0, 115); ax.tick_params(axis='x', labelsize=7)
fig.tight_layout(); fig.savefig('fig6_backdoor.pdf')

# ---------------- Fig.7: concept drift on EMBER ----------------
drift = R['drift_ember']; mlist = list(range(5, 11))
fig, ax = plt.subplots(figsize=(5.2, 3.1))
mk = {'No update': ('o', C['grey']), 'Random 5%': ('s', C['blue']),
      'DMLA 1% (ours)': ('d', C['green']), 'DMLA 5% (ours)': ('^', C['red']),
      'Oracle 20%': ('v', C['purple'])}
for name, v in drift.items():
    m, c = mk[name]
    ax.plot(mlist, v, marker=m, ms=4.5, lw=1.7, color=c, label=f'{name} (AUT={np.mean(v):.3f})')
ax.set_xticks(mlist); ax.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
ax.set_xlabel('Evaluation month (2017)'); ax.set_ylabel('Monthly F1')
ax.legend(fontsize=7, loc='lower left'); ax.set_ylim(0.80, 0.92)
fig.tight_layout(); fig.savefig('fig7_drift.pdf')

print('empirical figures written')
