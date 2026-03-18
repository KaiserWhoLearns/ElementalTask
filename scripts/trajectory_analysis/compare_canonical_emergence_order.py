#!/usr/bin/env python3
"""Compare canonical emergence order between OLMo-2 (1B+7B) and Amber.

For each task, a "canonical" OLMo-2 rank is derived by averaging the rank
of 1B and 7B emergence tokens. This is compared against the Amber rank via
Spearman and Kendall rank correlations. Only tasks evaluated in all three
model runs (evaled_all3) are used by default.

Inputs (emergence CSVs produced by get_emergence_point.py):
  --olmo2_1b   results/emergence_olmo2_1b_fixed_t0.8.csv
  --olmo2_7b   results/emergence_olmo2_7b_fixed_t0.8.csv
  --amber      results/emergence_amber_fixed_t0.8.csv

Outputs:
  results/canonical_order_olmo2_vs_amber_fixed_t0.8.csv   — per-task ranks
  results/canonical_order_olmo2_top20_fixed_t0.8.csv      — best-agreement tasks
  results/canonical_order_olmo2_bottom20_fixed_t0.8.csv   — worst-agreement tasks
  results/canonical_order_comparison_summary.csv          — Spearman/Kendall summary

Usage:
    python scripts/trajectory_analysis/compare_canonical_emergence_order.py \\
        --olmo2_1b results/emergence_olmo2_1b_fixed_t0.8.csv \\
        --olmo2_7b results/emergence_olmo2_7b_fixed_t0.8.csv \\
        --amber    results/emergence_amber_fixed_t0.8.csv \\
        --out_prefix results/canonical_order_olmo2_vs_amber_fixed_t0.8 \\
        --summary  results/canonical_order_comparison_summary.csv \\
        --definition fixed_t0.8_sigma0.8_censored_evaled_all3
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def load_emergence(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize task name: replace ':' with '_' for consistency
    df['task_norm'] = df['task'].str.replace(':', '_', regex=False)
    return df[['task_norm', 'emergence_tokens_B']].copy()


def average_rank(series: pd.Series) -> pd.Series:
    """Rank with ties → average rank, NaN → bottom (last) rank."""
    return series.rank(method='average', na_option='bottom')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--olmo2_1b', required=True, help='1B emergence CSV')
    parser.add_argument('--olmo2_7b', required=True, help='7B emergence CSV')
    parser.add_argument('--amber',    required=True, help='Amber emergence CSV')
    parser.add_argument('--out_prefix', default='results/canonical_order_olmo2_vs_amber',
                        help='Prefix for per-task output CSVs')
    parser.add_argument('--summary', default='results/canonical_order_comparison_summary.csv',
                        help='Path to append/create summary row')
    parser.add_argument('--definition', default='fixed_t0.8_sigma0.8_censored_evaled_all3',
                        help='Label for this run (written to summary)')
    parser.add_argument('--evaled_all3', action='store_true', default=True,
                        help='Restrict to tasks that have a row in all 3 files (regardless of NaN emergence)')
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    df1b  = load_emergence(pathlib.Path(args.olmo2_1b)).rename(columns={'emergence_tokens_B': 'tok_1b'})
    df7b  = load_emergence(pathlib.Path(args.olmo2_7b)).rename(columns={'emergence_tokens_B': 'tok_7b'})
    dfamb = load_emergence(pathlib.Path(args.amber)).rename(columns={'emergence_tokens_B': 'tok_amber'})

    # Track presence in each file before merging
    df1b['_in_1b']  = True
    df7b['_in_7b']  = True
    dfamb['_in_amb'] = True

    # ── Merge (outer) ─────────────────────────────────────────────────────
    df = df1b.merge(df7b, on='task_norm', how='outer').merge(dfamb, on='task_norm', how='outer')
    for col in ('_in_1b', '_in_7b', '_in_amb'):
        df[col] = df[col].fillna(False)

    # ── Filter to evaled_all3: row present in all 3 files ─────────────────
    if args.evaled_all3:
        mask_used = df['_in_1b'] & df['_in_7b'] & df['_in_amb']
    else:
        mask_used = df['_in_1b'] | df['_in_7b'] | df['_in_amb']

    df_used = df[mask_used].copy().reset_index(drop=True)
    n_used = len(df_used)

    # Re-rank within the filtered set; NaN → bottom rank
    df_used['rank_1b']    = average_rank(df_used['tok_1b'])
    df_used['rank_7b']    = average_rank(df_used['tok_7b'])
    df_used['rank_amber'] = average_rank(df_used['tok_amber'])
    # Canonical OLMo-2 rank = mean of 1B and 7B ranks
    df_used['rank_olmo2_canon'] = df_used[['rank_1b', 'rank_7b']].mean(axis=1)
    df_used['rank_gap_abs']     = (df_used['rank_olmo2_canon'] - df_used['rank_amber']).abs()

    # ── Correlations ──────────────────────────────────────────────────────
    sp_r, sp_p = spearmanr(df_used['rank_olmo2_canon'], df_used['rank_amber'])
    kt_r, kt_p = kendalltau(df_used['rank_olmo2_canon'], df_used['rank_amber'])

    print(f"\n{'='*60}")
    print(f"  Canonical OLMo-2 vs Amber emergence order ({n_used} tasks)")
    print(f"{'='*60}")
    print(f"  Spearman ρ = {sp_r:.4f}  (p = {sp_p:.2e})")
    print(f"  Kendall  τ = {kt_r:.4f}  (p = {kt_p:.2e})")

    # ── Output columns ────────────────────────────────────────────────────
    out_cols = ['task_norm', 'rank_olmo2_canon', 'rank_amber', 'rank_gap_abs',
                'tok_1b', 'tok_7b', 'tok_amber']
    df_sorted = df_used[out_cols].sort_values('rank_olmo2_canon').reset_index(drop=True)

    prefix = pathlib.Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    main_out = str(prefix) + '.csv'
    df_sorted.to_csv(main_out, index=False)
    print(f"\n  Per-task ranks  → {main_out}")

    top_n = min(20, len(df_sorted))
    top_out  = str(prefix).replace('_vs_amber', '_top20') + '.csv'
    bot_out  = str(prefix).replace('_vs_amber', '_bottom20') + '.csv'
    df_sorted.nsmallest(top_n, 'rank_gap_abs').to_csv(top_out, index=False)
    df_sorted.nlargest(top_n,  'rank_gap_abs').to_csv(bot_out, index=False)
    print(f"  Top-{top_n} agreement → {top_out}")
    print(f"  Bot-{top_n} agreement → {bot_out}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary_path = pathlib.Path(args.summary)
    row = pd.DataFrame([{
        'definition':                   args.definition,
        'n_tasks_evaled_all3':          n_used,
        'spearman_canonical_olmo2_vs_amber': sp_r,
        'spearman_p':                   sp_p,
        'kendall_canonical_olmo2_vs_amber':  kt_r,
        'kendall_p':                    kt_p,
    }])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing['definition'] != args.definition]  # replace if re-run
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(summary_path, index=False)
    print(f"  Summary         → {summary_path}\n")


if __name__ == '__main__':
    main()
