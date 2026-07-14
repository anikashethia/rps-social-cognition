#!/usr/bin/env python3
"""
Plot TypeScript vs MATLAB bot comparison results.
Run from repo root: python3 analysis/plot_bot_comparison.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ts_path     = os.path.join(ROOT, 'analysis', 'ts_probs.json')
matlab_path = os.path.join(ROOT, 'analysis', 'matlab_probs.json')

with open(ts_path) as f:
    ts_data = json.load(f)
with open(matlab_path) as f:
    ml_data = json.load(f)

# Collect probability values by level
colors = {0: '#3880cf', 1: '#ed9923', 2: '#2ba12b'}
labels = {0: 'Level 0', 1: 'Level 1', 2: 'Level 2'}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for level in [0, 1, 2]:
    ax = axes[level]

    # Use attr (bot attraction tracker) — evolves gradually, shows more spread
    ts_vals = np.array([r['attr'] for r in ts_data if r['level'] == level]).flatten()
    ml_vals = np.array([r['attr'] for r in ml_data if r['level'] == level]).flatten()

    ax.scatter(ts_vals, ml_vals, s=15, color=colors[level], alpha=0.6)

    lims = [0, 1]
    ax.plot(lims, lims, '--', color='gray', linewidth=1.2)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('TypeScript attr', fontsize=10)
    ax.set_ylabel('MATLAB attr', fontsize=10)
    ax.set_title(f'Level {level}', fontsize=11)
    ax.set_aspect('equal')

    max_diff = np.abs(ts_vals - ml_vals).max()
    n_pts = len(ts_vals)
    ax.text(0.05, 0.93, f'max diff = {max_diff:.2e}\nn = {n_pts} values',
            transform=ax.transAxes, fontsize=8, color='gray')

fig.suptitle('TypeScript (agents.ts) vs MATLAB (mn_RPS_task.m)\nattraction tracker values across 40 trials',
             fontsize=11)
plt.tight_layout()

out_path = os.path.join(ROOT, 'analysis', 'bot_comparison_figure.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved -> {out_path}')
plt.show()
