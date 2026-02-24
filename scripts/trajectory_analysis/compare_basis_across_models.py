#!/usr/bin/env python3
"""
Compare pairwise distances between tasks in basis-coordinate space across models.

For each model, every task FV is projected onto its SVD basis to get a k-dimensional
coordinate vector. Pairwise cosine distances are computed in this coordinate space.
Then we correlate the two distance matrices over shared tasks to measure how
consistent the basis geometry is between models (e.g., 1B vs 7B).

Usage:
    python scripts/trajectory_analysis/compare_basis_across_models.py \
        --dir_a function_vecs/results/olmo2_1b_compositional_holdout \
        --dir_b function_vecs/results/olmo2_7b_compositional_holdout \
        --label_a "OLMo-2 1B" --label_b "OLMo-2 7B" \
        --output_dir plots/basis_comparison
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import procrustes as scipy_procrustes
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def load_basis_and_coords(result_dir: str) -> Tuple[Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Load basis from an FV results directory and compute basis coordinates
    for all tasks (train + test).

    Returns:
        coords: dict mapping task_name -> coordinate vector (k,)
        task_names: ordered list of task names
        U: the basis matrix (d_model, k)
    """
    result_dir = Path(result_dir)

    # Load basis
    basis = np.load(result_dir / "skill_basis.npz", allow_pickle=True)
    U = basis["U"]             # (d_model, k)
    S = basis["S"]             # (k,)
    Vt = basis["Vt"]           # (k, n_train)
    train_names = list(basis["task_names"])
    mean = basis["mean"]
    has_mean = mean.size > 0

    k = U.shape[1]

    # Reconstruct train FVs: columns of U @ diag(S) @ Vt
    # Each column j of (U @ diag(S) @ Vt) is the train FV for task j
    train_fv_matrix = U @ np.diag(S) @ Vt  # (d_model, n_train)
    if has_mean:
        train_fv_matrix += mean  # un-center

    coords = {}

    # Train task coordinates: project back onto basis
    for i, name in enumerate(train_names):
        fv = train_fv_matrix[:, i]
        if has_mean:
            fv_centered = fv - mean.flatten()
        else:
            fv_centered = fv
        c = U.T @ fv_centered  # (k,)
        coords[name] = c

    # Test task coordinates
    test_dir = result_dir / "test_vectors"
    if test_dir.exists():
        for f in sorted(test_dir.glob("*.npz")):
            data = np.load(f, allow_pickle=True)
            name = str(data["task_name"])
            fv = data["function_vec"]  # (d_model,)
            if fv.shape[0] != U.shape[0]:
                print(f"  Warning: skipping {f.name} — dim {fv.shape[0]} != {U.shape[0]}")
                continue
            if has_mean:
                fv_centered = fv - mean.flatten()
            else:
                fv_centered = fv
            c = U.T @ fv_centered  # (k,)
            coords[name] = c

    return coords, U, S


def compute_pairwise_cosine(coords: Dict[str, np.ndarray], task_order: List[str]) -> np.ndarray:
    """Compute pairwise cosine distance matrix for tasks in the given order."""
    k = next(iter(coords.values())).shape[0]
    n = len(task_order)
    mat = np.zeros((n, k))
    for i, t in enumerate(task_order):
        mat[i] = coords[t]

    # Cosine distance = 1 - cosine_similarity
    dists = squareform(pdist(mat, metric="cosine"))
    return dists


def compute_pairwise_euclidean(coords: Dict[str, np.ndarray], task_order: List[str]) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix (on L2-normalized coords)."""
    k = next(iter(coords.values())).shape[0]
    n = len(task_order)
    mat = np.zeros((n, k))
    for i, t in enumerate(task_order):
        c = coords[t]
        norm = np.linalg.norm(c)
        mat[i] = c / (norm + 1e-10)  # normalize so magnitudes don't dominate

    dists = squareform(pdist(mat, metric="euclidean"))
    return dists


def shorten_name(name: str) -> str:
    """Shorten task names for plot labels."""
    # Remove common prefixes
    for prefix in ["compositional:", "simple_icl:", "textfrct:"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def plot_distance_matrix(dists: np.ndarray, task_names: List[str], title: str, ax=None):
    """Plot a distance matrix as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    short_names = [shorten_name(n) for n in task_names]
    im = ax.imshow(dists, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_correlation_scatter(dists_a: np.ndarray, dists_b: np.ndarray,
                              label_a: str, label_b: str,
                              task_names: List[str], ax=None):
    """Scatter plot of pairwise distances from model A vs model B."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Extract upper triangle (unique pairs)
    n = dists_a.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    x = dists_a[triu_idx]
    y = dists_b[triu_idx]

    # Color by task category
    pair_labels = []
    for i, j in zip(*triu_idx):
        ni, nj = task_names[i], task_names[j]
        both_comp = ni.startswith("compositional") and nj.startswith("compositional")
        both_elem = not ni.startswith("compositional") and not nj.startswith("compositional")
        if both_comp:
            pair_labels.append("comp-comp")
        elif both_elem:
            pair_labels.append("elem-elem")
        else:
            pair_labels.append("cross")

    colors = {"comp-comp": "#e74c3c", "elem-elem": "#3498db", "cross": "#95a5a6"}
    for cat in ["elem-elem", "cross", "comp-comp"]:
        mask = np.array([l == cat for l in pair_labels])
        if mask.sum() == 0:
            continue
        ax.scatter(x[mask], y[mask], alpha=0.4, s=15, c=colors[cat], label=cat, edgecolors="none")

    # Correlation
    r_pearson, p_pearson = pearsonr(x, y)
    r_spearman, p_spearman = spearmanr(x, y)

    # Identity line
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)

    ax.set_xlabel(f"Cosine distance ({label_a})", fontsize=10)
    ax.set_ylabel(f"Cosine distance ({label_b})", fontsize=10)
    ax.set_title(f"Pairwise task distances in basis space\n"
                 f"Pearson r={r_pearson:.3f} (p={p_pearson:.1e}), "
                 f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.1e})",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")

    return r_pearson, r_spearman


def plot_rank_comparison(dists_a: np.ndarray, dists_b: np.ndarray,
                          task_names: List[str], label_a: str, label_b: str, ax=None):
    """For each task, compare its nearest-neighbor ranking across models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    n = len(task_names)
    short_names = [shorten_name(t) for t in task_names]

    # For each task, get its nearest neighbor in each model
    nn_agree = 0
    top3_overlap = []
    for i in range(n):
        # Sort neighbors by distance (exclude self)
        rank_a = np.argsort(dists_a[i])
        rank_b = np.argsort(dists_b[i])
        # Remove self (index i)
        rank_a = rank_a[rank_a != i]
        rank_b = rank_b[rank_b != i]

        if rank_a[0] == rank_b[0]:
            nn_agree += 1

        top3_a = set(rank_a[:3])
        top3_b = set(rank_b[:3])
        top3_overlap.append(len(top3_a & top3_b) / 3.0)

    ax.bar(range(n), top3_overlap, color="#2ecc71", alpha=0.7, edgecolor="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_ylabel("Top-3 neighbor overlap", fontsize=10)
    ax.set_title(f"Nearest-neighbor consistency ({label_a} vs {label_b})\n"
                 f"NN-1 agreement: {nn_agree}/{n} ({100*nn_agree/n:.0f}%), "
                 f"Mean top-3 overlap: {np.mean(top3_overlap):.2f}",
                 fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1/3, color="gray", ls="--", alpha=0.5, label="chance (1/3)")
    ax.legend(fontsize=8)

    return nn_agree / n, np.mean(top3_overlap)


# ======================================================================
# Orthogonal Procrustes Analysis
# ======================================================================

def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray):
    """
    Find the orthogonal matrix Q that minimizes ||XQ - Y||_F.

    Both X,Y are (n, k) after zero-padding to the same k.
    Returns Q, disparity (sum-of-squares residual), and the
    Procrustes distance (disparity normalized by ||Y||^2).

    The algorithm:
        1. Center both matrices (subtract row-means)
        2. Compute SVD of X^T Y = U S V^T
        3. Q* = V U^T  (optimal orthogonal mapping)
        4. disparity = ||X Q* - Y||_F^2
    """
    # Center
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)

    # Scale to unit Frobenius norm (standard Procrustes normalization)
    sx = np.linalg.norm(X_c, "fro")
    sy = np.linalg.norm(Y_c, "fro")
    X_n = X_c / (sx + 1e-15)
    Y_n = Y_c / (sy + 1e-15)

    # Optimal rotation: SVD of X^T Y
    M = X_n.T @ Y_n
    U, S, Vt = np.linalg.svd(M)
    Q = U @ Vt  # orthogonal mapping

    # Ensure proper rotation (det = +1), not reflection
    if np.linalg.det(Q) < 0:
        # Flip sign of last column of U
        U[:, -1] *= -1
        Q = U @ Vt

    # Aligned coordinates
    X_aligned = X_n @ Q

    # Disparity = ||X_aligned - Y_n||_F^2
    disparity = np.sum((X_aligned - Y_n) ** 2)

    # Procrustes distance: disparity is already on unit-norm matrices
    # so it's in [0, 2]. Normalize to [0, 1] by dividing by 2.
    procrustes_dist = disparity / 2.0

    return Q, X_aligned, Y_n, disparity, procrustes_dist


def plot_procrustes_alignment(X_aligned: np.ndarray, Y_norm: np.ndarray,
                                task_names: List[str], label_a: str, label_b: str,
                                output_dir: str):
    """
    Visualize the Procrustes-aligned coordinates.
    Shows first 2 (and first 3) PCs of the aligned spaces overlaid.
    """
    short_names = [shorten_name(t) for t in task_names]

    # --- 2D projection (PC1 vs PC2 of the aligned space) ---
    # Use PCA on concatenated data for a shared projection
    combined = np.vstack([X_aligned, Y_norm])  # (2n, k)
    mean_c = combined.mean(axis=0)
    combined_c = combined - mean_c
    _, _, Vt_pca = np.linalg.svd(combined_c, full_matrices=False)

    X_2d = (X_aligned - mean_c) @ Vt_pca[:2].T
    Y_2d = (Y_norm - mean_c) @ Vt_pca[:2].T

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by category
    is_comp = [t.startswith("compositional") for t in task_names]

    for i, name in enumerate(task_names):
        color = "#e74c3c" if is_comp[i] else "#3498db"
        # Draw arrow from A to B
        ax.annotate("", xy=(Y_2d[i, 0], Y_2d[i, 1]),
                     xytext=(X_2d[i, 0], X_2d[i, 1]),
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=0.8))
        ax.scatter(X_2d[i, 0], X_2d[i, 1], c=color, marker="o", s=30, alpha=0.7, zorder=5)
        ax.scatter(Y_2d[i, 0], Y_2d[i, 1], c=color, marker="s", s=30, alpha=0.7, zorder=5)
        ax.annotate(short_names[i], (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=5, alpha=0.7, ha="center", va="bottom")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=8, label=f"{label_a} (elemental)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label=f"{label_a} (compositional)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db",
               markersize=8, label=f"{label_b} (elemental)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c",
               markersize=8, label=f"{label_b} (compositional)"),
        Line2D([0], [0], color="gray", alpha=0.5, lw=1, label="displacement"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper left")
    ax.set_xlabel("PC1 of aligned space", fontsize=10)
    ax.set_ylabel("PC2 of aligned space", fontsize=10)
    ax.set_title(f"Procrustes-aligned task coordinates\n"
                 f"(circles={label_a}, squares={label_b}, arrows=displacement)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "procrustes_alignment_2d.png"),
                dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/procrustes_alignment_2d.png")
    plt.close(fig)

    # --- Per-task displacement bar chart ---
    displacements = np.linalg.norm(X_aligned - Y_norm, axis=1)  # (n,)
    order = np.argsort(displacements)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bar = ["#e74c3c" if is_comp[i] else "#3498db" for i in order]
    ax.barh(range(len(order)), displacements[order], color=colors_bar, alpha=0.7,
            edgecolor="white")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short_names[i] for i in order], fontsize=6)
    ax.set_xlabel("Displacement after Procrustes alignment", fontsize=10)
    ax.set_title(f"Per-task Procrustes residual ({label_a} → {label_b})\n"
                 f"Mean displacement: {displacements.mean():.4f}", fontsize=11)
    ax.axvline(displacements.mean(), color="gray", ls="--", alpha=0.5, label="mean")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "procrustes_displacements.png"),
                dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/procrustes_displacements.png")
    plt.close(fig)

    return displacements


def main():
    parser = argparse.ArgumentParser(description="Compare basis-space task geometry across models")
    parser.add_argument("--dir_a", required=True, help="FV results dir for model A")
    parser.add_argument("--dir_b", required=True, help="FV results dir for model B")
    parser.add_argument("--label_a", default="Model A", help="Display label for model A")
    parser.add_argument("--label_b", default="Model B", help="Display label for model B")
    parser.add_argument("--output_dir", default="plots/basis_comparison", help="Output directory")
    parser.add_argument("--task_subset", default=None,
                        help="Comma-separated task subset to compare (default: all shared)")
    parser.add_argument("--exclude_pattern", default=None,
                        help="Regex pattern to exclude tasks")
    parser.add_argument("--max_k", type=int, default=None,
                        help="Truncate basis coordinates to this many dims (default: min(k_a, k_b)). "
                             "SVD components are ordered by variance, so truncating keeps the most "
                             "informative dimensions and enables fair comparison across models.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load coordinates
    print(f"Loading {args.label_a} from {args.dir_a}...")
    coords_a, U_a, S_a = load_basis_and_coords(args.dir_a)
    print(f"  {len(coords_a)} tasks, basis dim k={U_a.shape[1]}")

    print(f"Loading {args.label_b} from {args.dir_b}...")
    coords_b, U_b, S_b = load_basis_and_coords(args.dir_b)
    print(f"  {len(coords_b)} tasks, basis dim k={U_b.shape[1]}")

    # Truncate to common dimension for fair comparison
    k_a, k_b = U_a.shape[1], U_b.shape[1]
    max_k = args.max_k if args.max_k is not None else min(k_a, k_b)
    max_k = min(max_k, k_a, k_b)  # can't exceed either model's actual k
    if max_k < k_a or max_k < k_b:
        print(f"\nTruncating coordinates to top-{max_k} basis dims "
              f"(from {k_a} and {k_b}) for fair comparison")
        coords_a = {t: c[:max_k] for t, c in coords_a.items()}
        coords_b = {t: c[:max_k] for t, c in coords_b.items()}

    # Find shared tasks
    shared = sorted(set(coords_a.keys()) & set(coords_b.keys()))

    if args.exclude_pattern:
        import re
        pat = re.compile(args.exclude_pattern)
        before = len(shared)
        shared = [t for t in shared if not pat.search(t)]
        print(f"  Excluded {before - len(shared)} tasks matching '{args.exclude_pattern}'")

    if args.task_subset:
        subset = set(args.task_subset.split(","))
        shared = [t for t in shared if t in subset]

    print(f"\n{len(shared)} shared tasks for comparison")

    only_a = sorted(set(coords_a.keys()) - set(coords_b.keys()))
    only_b = sorted(set(coords_b.keys()) - set(coords_a.keys()))
    if only_a:
        print(f"  Only in {args.label_a} ({len(only_a)}): {', '.join(only_a[:5])}{'...' if len(only_a)>5 else ''}")
    if only_b:
        print(f"  Only in {args.label_b} ({len(only_b)}): {', '.join(only_b[:5])}{'...' if len(only_b)>5 else ''}")

    if len(shared) < 3:
        print("ERROR: Need at least 3 shared tasks for comparison")
        sys.exit(1)

    # Compute pairwise distance matrices
    dists_a = compute_pairwise_cosine(coords_a, shared)
    dists_b = compute_pairwise_cosine(coords_b, shared)

    # ======================================================================
    # Print summary statistics
    # ======================================================================
    triu = np.triu_indices(len(shared), k=1)
    da = dists_a[triu]
    db = dists_b[triu]
    r_p, p_p = pearsonr(da, db)
    r_s, p_s = spearmanr(da, db)

    print(f"\n{'='*60}")
    print(f"PAIRWISE DISTANCE CORRELATION (cosine distance in basis space)")
    print(f"{'='*60}")
    print(f"  Pearson  r = {r_p:.4f}  (p = {p_p:.2e})")
    print(f"  Spearman ρ = {r_s:.4f}  (p = {p_s:.2e})")
    print(f"  N pairs    = {len(da)}")

    # Break down by category
    comp_mask = np.array([t.startswith("compositional") for t in shared])
    elem_mask = ~comp_mask
    n_comp = comp_mask.sum()
    n_elem = elem_mask.sum()

    for subset_name, mask in [("compositional-only", comp_mask), ("elemental-only", elem_mask)]:
        idx = np.where(mask)[0]
        if len(idx) < 3:
            continue
        sub_triu = []
        for ii in range(len(idx)):
            for jj in range(ii+1, len(idx)):
                sub_triu.append((idx[ii], idx[jj]))
        if not sub_triu:
            continue
        rows, cols = zip(*sub_triu)
        sa = dists_a[rows, cols]
        sb = dists_b[rows, cols]
        rp, _ = pearsonr(sa, sb)
        rs, _ = spearmanr(sa, sb)
        print(f"  {subset_name} ({len(idx)} tasks, {len(sa)} pairs):  r={rp:.3f}, ρ={rs:.3f}")

    # Nearest neighbor agreement
    print(f"\nNEAREST-NEIGHBOR CONSISTENCY:")
    n = len(shared)
    nn1_agree = 0
    for i in range(n):
        rank_a = np.argsort(dists_a[i])
        rank_b = np.argsort(dists_b[i])
        rank_a = rank_a[rank_a != i]
        rank_b = rank_b[rank_b != i]
        if rank_a[0] == rank_b[0]:
            nn1_agree += 1
            tag = "✓"
        else:
            tag = "✗"
        print(f"  {tag} {shorten_name(shared[i]):30s}  "
              f"NN({args.label_a})={shorten_name(shared[rank_a[0]]):<25s}  "
              f"NN({args.label_b})={shorten_name(shared[rank_b[0]])}")
    print(f"  NN-1 agreement: {nn1_agree}/{n} ({100*nn1_agree/n:.0f}%)")

    # ======================================================================
    # Orthogonal Procrustes Analysis
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"ORTHOGONAL PROCRUSTES ANALYSIS")
    print(f"{'='*60}")

    # Build coordinate matrices (n_tasks × max_k) — already truncated to same dim above
    k_a = next(iter(coords_a.values())).shape[0]
    k_b = next(iter(coords_b.values())).shape[0]
    assert k_a == k_b, f"coords should be same dim after truncation: {k_a} vs {k_b}"

    X = np.array([coords_a[t] for t in shared])
    Y = np.array([coords_b[t] for t in shared])

    Q, X_aligned, Y_norm, disparity, proc_dist = orthogonal_procrustes(X, Y)

    print(f"  Basis dims (after truncation): {args.label_a}={k_a}, {args.label_b}={k_b}")
    print(f"  Disparity (||XQ - Y||²):  {disparity:.4f}")
    print(f"  Procrustes distance:      {proc_dist:.4f}  (0=identical, 1=orthogonal)")
    print(f"  Procrustes similarity:    {1-proc_dist:.4f}")

    # Per-task displacements
    displacements = np.linalg.norm(X_aligned - Y_norm, axis=1)
    print(f"\n  Per-task displacement (mean={displacements.mean():.4f}, "
          f"std={displacements.std():.4f}):")
    order = np.argsort(displacements)
    print(f"  {'Task':<35s} {'Displacement':>12s}  {'Category':>12s}")
    print(f"  {'-'*60}")
    for idx in order:
        cat = "compositional" if shared[idx].startswith("compositional") else "elemental"
        print(f"  {shorten_name(shared[idx]):<35s} {displacements[idx]:>12.4f}  {cat:>12s}")

    # Category-level summary
    comp_idx = [i for i, t in enumerate(shared) if t.startswith("compositional")]
    elem_idx = [i for i, t in enumerate(shared) if not t.startswith("compositional")]
    if comp_idx:
        print(f"\n  Elemental mean displacement:     {displacements[elem_idx].mean():.4f}")
        print(f"  Compositional mean displacement: {displacements[comp_idx].mean():.4f}")

    # Procrustes plots
    plot_procrustes_alignment(X_aligned, Y_norm, shared, args.label_a, args.label_b, args.output_dir)

    # ======================================================================
    # Plots
    # ======================================================================
    # 1) Side-by-side distance matrices
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plot_distance_matrix(dists_a, shared, f"Cosine distance ({args.label_a})", axes[0])
    plot_distance_matrix(dists_b, shared, f"Cosine distance ({args.label_b})", axes[1])
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "distance_matrices.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output_dir}/distance_matrices.png")
    plt.close(fig)

    # 2) Correlation scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_correlation_scatter(dists_a, dists_b, args.label_a, args.label_b, shared, ax)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "distance_correlation.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output_dir}/distance_correlation.png")
    plt.close(fig)

    # 3) NN consistency bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_rank_comparison(dists_a, dists_b, shared, args.label_a, args.label_b, ax)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "nn_consistency.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output_dir}/nn_consistency.png")
    plt.close(fig)

    # 4) Difference matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    diff = dists_a - dists_b
    vmax = np.abs(diff).max()
    short_names = [shorten_name(n) for n in shared]
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_title(f"Distance difference ({args.label_a} − {args.label_b})", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "distance_difference.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output_dir}/distance_difference.png")
    plt.close(fig)

    # 5) Singular value comparison (how many factors needed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, S, label in [(axes[0], S_a, args.label_a), (axes[1], S_b, args.label_b)]:
        var_explained = (S**2) / (S**2).sum()
        cum_var = np.cumsum(var_explained)
        ax.bar(range(len(S)), var_explained, alpha=0.7, color="#3498db", label="Individual")
        ax.plot(range(len(S)), cum_var, "r-o", markersize=4, label="Cumulative")
        ax.axhline(0.95, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance explained")
        ax.set_title(f"{label} (k={len(S)})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "singular_values.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output_dir}/singular_values.png")
    plt.close(fig)

    print(f"\nDone! All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
