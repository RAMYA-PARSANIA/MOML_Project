"""
Post-hoc Analysis & Visualization for MOO Results
===================================================
Reads trial data from results/ and generates:
  - 3D Pareto front scatter plot
  - Pairwise 2D Pareto projections
  - Parallel coordinates plot
  - Hypervolume indicator
  - Spacing & spread metrics
  - Pareto tabulation (sorted table)

All figures saved to report/figures/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.lines import Line2D

# Optional: pymoo for hypervolume
try:
    from pymoo.indicators.hv import HV
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False
    print("[WARN] pymoo not installed — hypervolume will be computed manually.")

RESULTS_DIR = "./results"
FIGURES_DIR = "./report/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Utility: plot styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "figure.dpi":      150,
    "savefig.bbox":    "tight",
})


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    df_all    = pd.read_csv(os.path.join(RESULTS_DIR, "all_trials.csv"))
    df_pareto = pd.read_csv(os.path.join(RESULTS_DIR, "pareto_front.csv"))
    return df_all, df_pareto


# ---------------------------------------------------------------------------
# 1. 3D Pareto front
# ---------------------------------------------------------------------------
def plot_3d_pareto(df_all, df_pareto):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # All trials (faded)
    ax.scatter(df_all["accuracy"], df_all["inference_ms"],
               df_all["n_parameters"] / 1e3,
               c="lightgray", alpha=0.4, s=20, label="Dominated")

    # Pareto front (highlighted)
    sc = ax.scatter(df_pareto["accuracy"], df_pareto["inference_ms"],
                    df_pareto["n_parameters"] / 1e3,
                    c=df_pareto["accuracy"], cmap="viridis",
                    edgecolors="black", s=60, linewidths=0.5,
                    label="Pareto-optimal")

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_zlabel("Parameters (×10³)")
    ax.set_title("3D Pareto Front — Fashion-MNIST CNN")
    fig.colorbar(sc, ax=ax, label="Accuracy", shrink=0.6, pad=0.1)
    ax.legend(loc="upper left")

    plt.savefig(os.path.join(FIGURES_DIR, "pareto_3d.png"))
    plt.close()
    print("  ✓ pareto_3d.png")


# ---------------------------------------------------------------------------
# 2. Pairwise 2D projections
# ---------------------------------------------------------------------------
def plot_pairwise(df_all, df_pareto):
    pairs = [
        ("accuracy",     "inference_ms",  "Accuracy", "Inference Time (ms)"),
        ("accuracy",     "n_parameters",  "Accuracy", "Parameters"),
        ("inference_ms", "n_parameters",  "Inference Time (ms)", "Parameters"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        ax.scatter(df_all[x], df_all[y],
                   c="lightgray", alpha=0.4, s=15, label="Dominated")
        ax.scatter(df_pareto[x], df_pareto[y],
                   c="tab:red", edgecolors="black", s=40,
                   linewidths=0.5, zorder=5, label="Pareto-optimal")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=9)

    fig.suptitle("Pairwise Pareto Projections", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pareto_pairwise.png"))
    plt.close()
    print("  ✓ pareto_pairwise.png")


# ---------------------------------------------------------------------------
# 3. Parallel coordinates
# ---------------------------------------------------------------------------
def plot_parallel_coordinates(df_pareto):
    cols = ["accuracy", "inference_ms", "n_parameters"]
    df_norm = df_pareto[cols].copy()

    # Normalize each column to [0, 1]
    for c in cols:
        mn, mx = df_norm[c].min(), df_norm[c].max()
        if mx - mn > 0:
            df_norm[c] = (df_norm[c] - mn) / (mx - mn)
        else:
            df_norm[c] = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cols))

    for _, row in df_norm.iterrows():
        ax.plot(x, row[cols].values, alpha=0.5, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy\n(higher=better)", "Inference (ms)\n(lower=better)",
                         "Parameters\n(lower=better)"])
    ax.set_ylabel("Normalized Value")
    ax.set_title("Parallel Coordinates — Pareto-Optimal Solutions")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "parallel_coords.png"))
    plt.close()
    print("  ✓ parallel_coords.png")


# ---------------------------------------------------------------------------
# 4. Hypervolume
# ---------------------------------------------------------------------------
def compute_hypervolume(df_pareto):
    """
    Compute hypervolume indicator for the Pareto front.
    Objectives: minimize(-accuracy), minimize(inference_ms), minimize(n_params)
    Reference point: slightly worse than the worst observed point.
    """
    obj = np.column_stack([
        -df_pareto["accuracy"].values,        # minimize
        df_pareto["inference_ms"].values,      # minimize
        df_pareto["n_parameters"].values,      # minimize
    ])

    # Reference point: worst in each objective + margin
    ref = obj.max(axis=0) * 1.1

    if HAS_PYMOO:
        hv_indicator = HV(ref_point=ref)
        hv = hv_indicator(obj)
    else:
        # Simple 2D fallback (use first two objectives)
        hv = _manual_hypervolume_2d(obj[:, :2], ref[:2])

    return hv, ref


def _manual_hypervolume_2d(points, ref):
    """Exact 2-D hypervolume via sweep."""
    pts = points[points[:, 0].argsort()]
    hv = 0.0
    prev_y = ref[1]
    for p in pts:
        if p[1] < prev_y:
            hv += (ref[0] - p[0]) * (prev_y - p[1])
            prev_y = p[1]
    return hv


# ---------------------------------------------------------------------------
# 5. Spacing & spread metrics
# ---------------------------------------------------------------------------
def compute_spacing(df_pareto):
    """
    Spacing metric (Schott, 1995):
    Measures uniformity of the Pareto front distribution.
    S = sqrt( (1/(n-1)) * sum( (d_i - d_mean)^2 ) )
    where d_i = min distance from point i to all other points.
    Lower = more uniform.
    """
    obj = np.column_stack([
        df_pareto["accuracy"].values,
        df_pareto["inference_ms"].values,
        df_pareto["n_parameters"].values,
    ])
    # Normalize objectives to [0,1] for fair distance computation
    mn = obj.min(axis=0)
    mx = obj.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    obj_norm = (obj - mn) / rng

    n = len(obj_norm)
    if n < 2:
        return 0.0

    d = []
    for i in range(n):
        dists = [np.linalg.norm(obj_norm[i] - obj_norm[j])
                 for j in range(n) if j != i]
        d.append(min(dists))
    d = np.array(d)
    d_mean = d.mean()
    spacing = np.sqrt(np.sum((d - d_mean) ** 2) / (n - 1))
    return spacing


def compute_spread(df_pareto):
    """
    Spread / Delta metric: measures extent + uniformity.
    Uses the range of each normalized objective.
    """
    obj = np.column_stack([
        df_pareto["accuracy"].values,
        df_pareto["inference_ms"].values,
        df_pareto["n_parameters"].values,
    ])
    mn = obj.min(axis=0)
    mx = obj.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    # Spread as geometric mean of normalized ranges
    spread = np.prod(rng) ** (1.0 / len(rng))
    return spread


# ---------------------------------------------------------------------------
# 6. Convergence plot (accuracy over trials)
# ---------------------------------------------------------------------------
def plot_convergence(df_all):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = [
        ("accuracy",      "Accuracy (↑)",          "tab:blue"),
        ("inference_ms",   "Inference Time ms (↓)", "tab:orange"),
        ("n_parameters",   "Parameters (↓)",        "tab:green"),
    ]

    for ax, (col, label, color) in zip(axes, metrics):
        ax.scatter(df_all["trial"], df_all[col], s=12, alpha=0.6, c=color)
        # Running best
        if "accuracy" in col:
            running = df_all[col].cummax()
        else:
            running = df_all[col].cummin()
        ax.plot(df_all["trial"], running, c="black", linewidth=1.5,
                label="Running best")
        ax.set_xlabel("Trial")
        ax.set_ylabel(label)
        ax.legend(fontsize=9)

    fig.suptitle("Convergence Over Trials", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "convergence.png"))
    plt.close()
    print("  ✓ convergence.png")


# ---------------------------------------------------------------------------
# 7. Hyperparameter importance (simple bar chart)
# ---------------------------------------------------------------------------
def plot_param_distribution(df_all, df_pareto):
    """Compare hyperparameter distributions: all trials vs Pareto."""
    cat_params = ["optimizer_type", "batch_size", "input_resolution"]

    fig, axes = plt.subplots(1, len(cat_params), figsize=(15, 4.5))
    for ax, param in zip(axes, cat_params):
        all_counts    = df_all[param].value_counts(normalize=True).sort_index()
        pareto_counts = df_pareto[param].value_counts(normalize=True).sort_index()
        idx = all_counts.index.union(pareto_counts.index).sort_values()
        all_vals    = [all_counts.get(k, 0) for k in idx]
        pareto_vals = [pareto_counts.get(k, 0) for k in idx]

        x = np.arange(len(idx))
        w = 0.35
        ax.bar(x - w/2, all_vals, w, label="All Trials", alpha=0.7)
        ax.bar(x + w/2, pareto_vals, w, label="Pareto", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in idx], fontsize=9)
        ax.set_xlabel(param)
        ax.set_ylabel("Proportion")
        ax.legend(fontsize=8)

    fig.suptitle("Hyperparameter Distribution: All vs Pareto", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "param_distribution.png"))
    plt.close()
    print("  ✓ param_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df_all, df_pareto = load_data()
    print(f"Loaded {len(df_all)} trials, {len(df_pareto)} Pareto-optimal\n")

    # ---- Plots ----
    print("Generating plots:")
    plot_3d_pareto(df_all, df_pareto)
    plot_pairwise(df_all, df_pareto)
    plot_parallel_coordinates(df_pareto)
    plot_convergence(df_all)
    plot_param_distribution(df_all, df_pareto)

    # ---- Metrics ----
    print("\nComputing quality metrics:")
    hv, ref = compute_hypervolume(df_pareto)
    spacing = compute_spacing(df_pareto)
    spread  = compute_spread(df_pareto)

    print(f"  Hypervolume     : {hv:,.2f}")
    print(f"  Spacing (↓=better): {spacing:.6f}")
    print(f"  Spread          : {spread:,.2f}")
    print(f"  Reference point : {ref}")
    print(f"  |Pareto front|  : {len(df_pareto)}")

    # ---- Save metrics ----
    metrics = {
        "hypervolume":      float(hv),
        "spacing":          float(spacing),
        "spread":           float(spread),
        "reference_point":  ref.tolist(),
        "n_pareto":         len(df_pareto),
        "n_total_trials":   len(df_all),
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {RESULTS_DIR}/metrics.json")

    # ---- Pareto tabulation ----
    print("\n" + "=" * 80)
    print("PARETO FRONT (sorted by accuracy)")
    print("=" * 80)
    display_cols = ["trial", "accuracy", "inference_ms", "n_parameters",
                    "n_conv_layers", "learning_rate", "batch_size",
                    "dropout_rate", "optimizer_type", "input_resolution"]
    available = [c for c in display_cols if c in df_pareto.columns]
    print(df_pareto[available].sort_values("accuracy", ascending=False)
          .to_string(index=False))

    # Also save as LaTeX table
    latex_cols = ["trial", "accuracy", "inference_ms", "n_parameters"]
    tex = df_pareto[latex_cols].sort_values("accuracy", ascending=False).to_latex(
        index=False,
        float_format="%.4f",
        column_format="cccc",
        caption="Pareto-optimal solutions sorted by accuracy.",
        label="tab:pareto",
    )
    with open(os.path.join(FIGURES_DIR, "pareto_table.tex"), "w") as f:
        f.write(tex)
    print(f"\nLaTeX table saved → {FIGURES_DIR}/pareto_table.tex")

    print("\nAll done! Figures are in report/figures/")


if __name__ == "__main__":
    main()
