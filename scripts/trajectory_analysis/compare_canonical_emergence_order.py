#!/usr/bin/env python3
"""Pairwise emergence-order comparison across models.

For every pair of models, tasks are ranked by emergence token count.
Unemerged tasks (NaN emergence_tokens_B) are treated as censored — they are
assigned the bottom rank (as if they emerged one step after the last observed
checkpoint).  Spearman ρ and Kendall τ (and their p-values) are reported for
each pair.

By default only tasks evaluated in every specified model are included
(--evaled_all, on by default).

Usage:
    python scripts/trajectory_analysis/compare_canonical_emergence_order.py \\
        --models olmo2_1b:results/emergence_olmo2_1b_fixed_t0.8.csv \\
                 olmo2_7b:results/emergence_olmo2_7b_fixed_t0.8.csv \\
                 amber:results/emergence_amber_fixed_t0.8.csv \\
        --summary results/pairwise_emergence_comparison_summary.csv \\
        --definition fixed_t0.8_censored_evaled_all3
"""

import argparse
import itertools
import pathlib
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def load_emergence(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['task_norm'] = df['task'].str.replace(':', '_', regex=False)
    return df[['task_norm', 'emergence_tokens_B']].copy()


def censored_rank(series: pd.Series) -> pd.Series:
    """Rank with ties → average rank, NaN (unemerged/censored) → bottom rank."""
    return series.rank(method='average', na_option='bottom')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--models', nargs='+', required=True, metavar='NAME:PATH',
        help='Two or more  name:path  pairs, e.g. olmo2_1b:results/emergence_olmo2_1b_fixed_t0.8.csv',
    )
    parser.add_argument('--summary', default='results/pairwise_emergence_comparison_summary.csv',
                        help='Path to append/create summary rows (one row per pair)')
    parser.add_argument('--definition', default=None,
                        help='Label for this run (default: auto-generated)')
    parser.add_argument('--evaled_all', action='store_true', default=True,
                        help='Restrict to tasks with a row in every model file (default: True)')
    args = parser.parse_args()

    # ── Parse name:path pairs ─────────────────────────────────────────────
    model_files = {}
    for token in args.models:
        if ':' not in token:
            parser.error(f"--models entries must be  name:path  (got: {token!r})")
        name, path = token.split(':', 1)
        model_files[name] = pathlib.Path(path)

    names = list(model_files)
    if len(names) < 2:
        parser.error("Need at least two models for pairwise comparison.")

    definition = args.definition or f"{'_'.join(names)}_censored_evaled_all"
    print(f"\n  Models     : {names}")
    print(f"  Definition : {definition}")

    # ── Load all files ────────────────────────────────────────────────────
    dfs = {}
    for name, path in model_files.items():
        dfs[name] = load_emergence(path).rename(columns={'emergence_tokens_B': f'tok_{name}'})
        dfs[name][f'_in_{name}'] = True

    # ── Outer-merge all models ────────────────────────────────────────────
    merged = None
    for name, df in dfs.items():
        merged = df if merged is None else merged.merge(df, on='task_norm', how='outer')

    presence_cols = [f'_in_{n}' for n in names]
    for col in presence_cols:
        merged[col] = merged[col].fillna(False).infer_objects(copy=False)

    # ── Global filter (tasks present in ALL models) ───────────────────────
    if args.evaled_all:
        merged = merged[merged[presence_cols].all(axis=1)].copy().reset_index(drop=True)

    n_total = len(merged)
    print(f"  Tasks after filter : {n_total}\n")

    # ── Pairwise comparisons ──────────────────────────────────────────────
    rows = []
    header_width = 60
    print('=' * header_width)
    for name_a, name_b in itertools.combinations(names, 2):
        # For each pair, work on rows where both models have data
        pair_mask = merged[f'_in_{name_a}'] & merged[f'_in_{name_b}']
        df_pair = merged[pair_mask].copy().reset_index(drop=True)
        n_pair = len(df_pair)

        rank_a = censored_rank(df_pair[f'tok_{name_a}'])
        rank_b = censored_rank(df_pair[f'tok_{name_b}'])

        sp_r, sp_p = spearmanr(rank_a, rank_b)
        kt_r, kt_p = kendalltau(rank_a, rank_b)

        print(f"  {name_a} vs {name_b}  (n={n_pair})")
        print(f"    Spearman ρ = {sp_r:.4f}  (p = {sp_p:.2e})")
        print(f"    Kendall  τ = {kt_r:.4f}  (p = {kt_p:.2e})")
        print()

        rows.append({
            'definition':         definition,
            'model_a':            name_a,
            'model_b':            name_b,
            'n_tasks':            n_pair,
            'spearman_censored':  sp_r,
            'spearman_p':         sp_p,
            'kendall_censored':   kt_r,
            'kendall_p':          kt_p,
        })
    print('=' * header_width)

    # ── Save summary ──────────────────────────────────────────────────────
    summary_path = pathlib.Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    new_rows = pd.DataFrame(rows)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        # Drop any rows with the same definition+pair so we can overwrite cleanly
        key = existing['definition'] == definition
        existing = existing[~key]
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
    new_rows.to_csv(summary_path, index=False)
    print(f"\n  Summary → {summary_path}\n")


if __name__ == '__main__':
    main()
