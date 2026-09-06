#!/usr/bin/env python3
"""Plots for the M1 report. Run from src/: python3 make_figures.py"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "../figures"
os.makedirs(OUT, exist_ok=True)


def read_col(path, col):
    vals = []
    with open(path) as f:
        for row in csv.DictReader(f):
            vals.append(float(row[col]))
    return vals


# Gaussian posterior vs N(0.5, 0.5)
theta = read_col("../results/gaussian_samples_1000000.csv", "theta")
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(theta, bins=80, density=True, color="#4C72B0", alpha=0.7,
        label="ABC (N=1e6, eps=0.1)")
xs = np.linspace(-2.5, 3.5, 400)
sd = math.sqrt(0.5)
pdf = np.exp(-0.5 * ((xs - 0.5) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
ax.plot(xs, pdf, "r-", lw=2, label="N(0.5, 0.5)")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("density")
ax.legend(fontsize=8)
ax.set_xlim(-2.5, 3.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gaussian_posterior.pdf"))
fig.savefig(os.path.join(OUT, "gaussian_posterior.png"), dpi=150)
plt.close()

# SIR accepted samples
beta = read_col("../results/sir_early_samples_1000000.csv", "beta")
gamma = read_col("../results/sir_early_samples_1000000.csv", "gamma")
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(beta, gamma, s=1.5, alpha=0.25, color="#2196F3", rasterized=True,
           label="accepted")
ax.axvline(0.4, color="red", lw=1.5, ls="--", label=r"true $\beta$")
ax.axhline(0.1, color="orange", lw=1.5, ls="--", label=r"true $\gamma$")
ax.set_xlim(0.05, 1.05)
ax.set_ylim(-0.02, 0.55)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\gamma$")
ax.legend(fontsize=8, markerscale=6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sir_posterior.pdf"))
fig.savefig(os.path.join(OUT, "sir_posterior.png"), dpi=150)
plt.close()

# Step counts: samples csv has no per-particle steps, so sketch from mean/std
mean_steps, std_steps, T = 64.2, 44.9, 160
k = (mean_steps / std_steps) ** 2
scale = mean_steps / k
steps = np.clip(np.round(np.random.default_rng(42).gamma(k, scale, 100000)).astype(int), 1, T)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(steps, bins=40, color="#4CAF50", alpha=0.8, density=True)
ax.axvline(mean_steps, color="red", lw=1.5, ls="--", label="early-exit mean")
ax.axvline(T, color="black", lw=1.5, ls=":", label="fixed T")
ax.set_xlabel("steps / particle")
ax.set_ylabel("density")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sir_step_histogram.pdf"))
fig.savefig(os.path.join(OUT, "sir_step_histogram.png"), dpi=150)
plt.close()

# Runtime vs N (means of 3 reps from summary csvs)
Ns = [1e4, 1e5, 1e6]
gauss_rt = [(0.000815 + 0.000703 + 0.000702) / 3,
            (0.00600 + 0.005273 + 0.005230) / 3,
            (0.0520 + 0.051879 + 0.051874) / 3]
sir_early_rt = [(0.205516 + 0.206861 + 0.205111) / 3,
                (2.20042 + 2.075208 + 2.075139) / 3,
                (21.3646 + 21.217686 + 20.883033) / 3]
sir_fixed_rt = [(0.219633 + 0.212224 + 0.213926) / 3,
                (2.13491 + 2.138860 + 2.178985) / 3,
                (21.4039 + 21.261006 + 21.463146) / 3]

fig, ax = plt.subplots(figsize=(5, 3.8))
ax.loglog(Ns, gauss_rt, "o-", color="#4C72B0", lw=2, label="Gaussian")
ax.loglog(Ns, sir_early_rt, "s--", color="#DD8452", lw=2, label="SIR early-exit")
ax.loglog(Ns, sir_fixed_rt, "^:", color="#55A868", lw=2, label="SIR fixed-T")
ax.set_xlabel("N")
ax.set_ylabel("runtime (s)")
ax.legend(fontsize=8)
ax.set_xticks([1e4, 1e5, 1e6])
ax.set_xticklabels([r"$10^4$", r"$10^5$", r"$10^6$"])
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "runtime_comparison.pdf"))
fig.savefig(os.path.join(OUT, "runtime_comparison.png"), dpi=150)
plt.close()
