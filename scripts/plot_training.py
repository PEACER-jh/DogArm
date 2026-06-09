#!/usr/bin/env python3
"""Plot training curves from TensorBoard event files.

Usage:
    python scripts/plot_training.py save/velocity/2026-06-04_22-37-55/
    python scripts/plot_training.py logs/rsl_rl/go2arm_direct/2026-06-04_22-37-55/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load scalar data from TensorBoard event files."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from TensorBoard logs.")
    parser.add_argument("logdir", type=str, help="Path to the log directory")
    parser.add_argument("--smooth", type=float, default=0.6, help="Exponential smoothing factor")
    args = parser.parse_args()

    logdir = os.path.abspath(args.logdir)
    if not os.path.isdir(logdir):
        print(f"Error: directory not found: {logdir}")
        sys.exit(1)

    # Find the correct sub-directory containing events files
    event_dir = logdir
    for root, dirs, files in os.walk(logdir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                event_dir = root
                break
    logdir = event_dir

    # List available tags
    ea_tmp = EventAccumulator(logdir)
    ea_tmp.Reload()
    available = ea_tmp.Tags()["scalars"]
    print(f"[INFO] Reading logs from: {logdir}")
    print(f"[INFO] Available tags: {available}")

    # Key metrics to plot
    metrics = [
        ("Train/mean_reward", "Mean Reward"),
        ("Train/mean_episode_length", "Mean Episode Length"),
        ("Policy/mean_std", "Mean Action Std"),
        ("Loss/value", "Mean Value Loss"),
        ("Loss/surrogate", "Mean Surrogate Loss"),
        ("Loss/entropy", "Mean Entropy Loss"),
        ("Loss/learning_rate", "Learning Rate"),
    ]

    out_dir = os.path.join(args.logdir, "training_plots")
    os.makedirs(out_dir, exist_ok=True)

    for tag, label in metrics:
        steps, values = load_scalars(logdir, tag)
        if len(values) == 0:
            print(f"[SKIP] No data for '{tag}'")
            continue

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(steps, values, alpha=0.3, color="steelblue", linewidth=0.5)
        if len(values) > 1:
            smoothed = np.zeros_like(values)
            smoothed[0] = values[0]
            for i in range(1, len(values)):
                smoothed[i] = args.smooth * smoothed[i - 1] + (1 - args.smooth) * values[i]
            ax.plot(steps, smoothed, color="darkorange", linewidth=1.5)
        ax.set_ylabel(label)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

        safe_name = tag.replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {out_path}")

    print(f"[INFO] All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
