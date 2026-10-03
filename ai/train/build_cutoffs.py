"""Train recovery cutoffs from the frailty dataset."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai.recovery import fit_tau, hr_drop_60s, MIN_R2


def compute_percentile(value: float, p25: float, p50: float, p75: float, p90: float) -> float:
    """Interpolate a value's percentile given key percentiles.

    Uses piecewise linear interpolation:
    - p0 = 0.5 * p25 (extrapolated)
    - p100 = 2 * p90 (extrapolated)
    """
    p0 = 0.5 * p25
    p100 = 2 * p90

    # Piecewise linear interpolation
    if value <= p0:
        return 0.0
    elif value <= p25:
        return 0 + 25 * (value - p0) / (p25 - p0)
    elif value <= p50:
        return 25 + 25 * (value - p25) / (p50 - p25)
    elif value <= p75:
        return 50 + 25 * (value - p50) / (p75 - p50)
    elif value <= p90:
        return 75 + 15 * (value - p75) / (p90 - p75)
    elif value <= p100:
        return 90 + 10 * (value - p90) / (p100 - p90)
    else:
        return 100.0


def train_cutoffs(episodes_path: Path, output_path: Path, plot_dir: Path) -> dict:
    """Train recovery cutoffs from extracted episodes.

    Args:
        episodes_path: path to frailty_episodes.parquet
        output_path: path to save recovery_cutoffs.json
        plot_dir: path to save diagnostic plots

    Returns:
        Dictionary with trained cutoffs
    """
    # Load episodes
    if not episodes_path.exists():
        # Try to generate mock data
        print(f"Episodes file not found at {episodes_path}")
        print("Generating mock data for testing...")
        from ai.train.mock_data import generate_mock_episodes
        df = generate_mock_episodes(200)
        df.to_parquet(episodes_path, index=False)
    else:
        df = pd.read_parquet(episodes_path)

    print(f"Loaded {len(df)} episodes")

    # Fit tau and compute metrics for each episode
    results = []

    for idx, row in df.iterrows():
        t_s = np.array(row['t_s'])
        hr = np.array(row['hr'])
        baseline = row['baseline']
        hr0 = row['hr0']

        # Fit tau
        fit_result = fit_tau(t_s, hr, baseline, hr0)

        # Compute HR drop at 60s
        drop_60s = hr_drop_60s(t_s, hr, hr0)

        # Store results
        if fit_result.tau_s is not None and fit_result.r2 is not None:
            results.append({
                'patient_id': row['patient_id'],
                'test': row['test'],
                'age': row['age'],
                'meds_flag': row['meds_flag'],
                'tau_s': fit_result.tau_s,
                'r2': fit_result.r2,
                'at_bound': fit_result.at_bound,
                'hr_drop_60s': drop_60s,
                'baseline': baseline,
                'hr0': hr0,
                'rise_bpm': row['rise_bpm']
            })

    results_df = pd.DataFrame(results)
    print(f"\nFit results: {len(results_df)}/{len(df)} episodes")

    # Filter by quality
    quality_mask = (results_df['r2'] >= MIN_R2) & (~results_df['at_bound'])
    filtered_df = results_df[quality_mask].copy()

    print(f"After quality filter (R²≥{MIN_R2}, not at bound): {len(filtered_df)} episodes")
    print(f"  Rejected: {len(results_df) - len(filtered_df)} ({(1 - len(filtered_df)/len(results_df))*100:.1f}%)")

    # Compute cutoffs for each group
    groups = {
        'all': filtered_df,
        'meds': filtered_df[filtered_df['meds_flag']],
        'no_meds': filtered_df[~filtered_df['meds_flag']]
    }

    cutoffs = {
        'version': 1,
        'source': 'Wearable-based signals during physical exercises from patients with frailty after open-heart surgery, PhysioNet v1.0.0, DOI: 10.13026/mp8k-7p27',
        'created': datetime.utcnow().isoformat() + 'Z',
        'groups': {}
    }

    summary_table = []

    for group_name, group_df in groups.items():
        if len(group_df) == 0:
            print(f"\nWarning: Group '{group_name}' has no episodes, skipping...")
            continue

        tau_values = group_df['tau_s'].values
        drop_values = group_df['hr_drop_60s'].dropna().values

        group_stats = {
            'n': len(group_df),
            'tau_p25': float(np.percentile(tau_values, 25)),
            'tau_p50': float(np.percentile(tau_values, 50)),
            'tau_p75': float(np.percentile(tau_values, 75)),
            'tau_p90': float(np.percentile(tau_values, 90)),
            'hr_drop_60s_p10': float(np.percentile(drop_values, 10)) if len(drop_values) > 0 else None,
            'hr_drop_60s_p25': float(np.percentile(drop_values, 25)) if len(drop_values) > 0 else None,
            'hr_drop_60s_p50': float(np.percentile(drop_values, 50)) if len(drop_values) > 0 else None,
            'median_age': float(group_df['age'].median())
        }

        cutoffs['groups'][group_name] = group_stats

        summary_table.append({
            'Group': group_name,
            'N': group_stats['n'],
            'Age (median)': f"{group_stats['median_age']:.0f}",
            'Tau p25': f"{group_stats['tau_p25']:.1f}s",
            'Tau p50': f"{group_stats['tau_p50']:.1f}s",
            'Tau p75': f"{group_stats['tau_p75']:.1f}s",
            'Tau p90': f"{group_stats['tau_p90']:.1f}s",
            'HR drop 60s (p50)': f"{group_stats['hr_drop_60s_p50']:.1f}" if group_stats['hr_drop_60s_p50'] else 'N/A'
        })

    # Save cutoffs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cutoffs, f, indent=2)

    print(f"\nCutoffs saved to {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    summary_df = pd.DataFrame(summary_table)
    print(summary_df.to_string(index=False))
    print("="*80)

    # Generate plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tau histogram with meds overlay
    plt.figure(figsize=(12, 6))

    all_tau = filtered_df['tau_s'].values
    meds_tau = groups['meds']['tau_s'].values if 'meds' in groups else []
    no_meds_tau = groups['no_meds']['tau_s'].values if 'no_meds' in groups else []

    bins = np.linspace(0, min(200, np.percentile(all_tau, 99)), 40)

    plt.hist(no_meds_tau, bins=bins, alpha=0.5, label='No meds', color='blue', density=False)
    plt.hist(meds_tau, bins=bins, alpha=0.5, label='On meds', color='red', density=False)

    # Add vertical lines for p75 and p90
    all_group = cutoffs['groups']['all']
    plt.axvline(all_group['tau_p75'], color='orange', linestyle='--', linewidth=2, label=f"p75 (slow) = {all_group['tau_p75']:.1f}s")
    plt.axvline(all_group['tau_p90'], color='darkred', linestyle='--', linewidth=2, label=f"p90 (very slow) = {all_group['tau_p90']:.1f}s")

    plt.xlabel('Recovery Tau (seconds)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Recovery Time Constants', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    tau_hist_path = plot_dir / 'tau_histogram.png'
    plt.savefig(tau_hist_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {tau_hist_path}")

    # 2. Example fits
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Select examples: 2 fast, 2 median, 2 slow
    tau_sorted = filtered_df.sort_values('tau_s')
    n = len(tau_sorted)

    example_indices = [
        tau_sorted.index[int(n * 0.1)],   # fast
        tau_sorted.index[int(n * 0.2)],   # fast
        tau_sorted.index[int(n * 0.45)],  # median
        tau_sorted.index[int(n * 0.55)],  # median
        tau_sorted.index[int(n * 0.85)],  # slow
        tau_sorted.index[int(n * 0.95)],  # slow
    ]

    for ax_idx, ep_idx in enumerate(example_indices):
        row = df.loc[ep_idx]
        t_s = np.array(row['t_s'])
        hr = np.array(row['hr'])
        baseline = row['baseline']
        hr0 = row['hr0']

        fit_result = fit_tau(t_s, hr, baseline, hr0)

        ax = axes[ax_idx]

        # Plot data
        ax.scatter(t_s, hr, alpha=0.5, s=20, color='blue', label='Data')

        # Plot fitted curve
        if fit_result.tau_s is not None:
            t_fit = np.linspace(0, max(t_s), 100)
            hr_fit = baseline + (hr0 - baseline) * np.exp(-t_fit / fit_result.tau_s)
            ax.plot(t_fit, hr_fit, 'r-', linewidth=2, label='Fit')

        # Plot baseline
        ax.axhline(baseline, color='green', linestyle='--', alpha=0.7, label='Baseline')

        tau_val = fit_result.tau_s if fit_result.tau_s else float('nan')
        r2_val = fit_result.r2 if fit_result.r2 else float('nan')

        ax.set_title(f"τ={tau_val:.1f}s, R²={r2_val:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('HR (bpm)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    example_fits_path = plot_dir / 'example_fits.png'
    plt.savefig(example_fits_path, dpi=150)
    plt.close()
    print(f"Plot saved: {example_fits_path}")

    # 3. Tau vs Age
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['age'], filtered_df['tau_s'], alpha=0.4, s=30)

    # Add trend line
    z = np.polyfit(filtered_df['age'], filtered_df['tau_s'], 1)
    p = np.poly1d(z)
    age_range = np.linspace(filtered_df['age'].min(), filtered_df['age'].max(), 100)
    plt.plot(age_range, p(age_range), "r--", linewidth=2, label=f'Trend: τ = {z[0]:.2f}*age + {z[1]:.1f}')

    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Recovery Tau (seconds)', fontsize=12)
    plt.title('Recovery Speed vs Age', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    tau_age_path = plot_dir / 'tau_vs_age.png'
    plt.savefig(tau_age_path, dpi=150)
    plt.close()
    print(f"Plot saved: {tau_age_path}")

    # 4. Tau by test type
    plt.figure(figsize=(10, 6))

    test_types = filtered_df['test'].unique()
    tau_by_test = [filtered_df[filtered_df['test'] == tt]['tau_s'].values for tt in test_types]

    bp = plt.boxplot(tau_by_test, labels=test_types, patch_artist=True, showmeans=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    plt.ylabel('Recovery Tau (seconds)', fontsize=12)
    plt.xlabel('Test Type', fontsize=12)
    plt.title('Recovery Speed by Exercise Type', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    tau_test_path = plot_dir / 'tau_by_test.png'
    plt.savefig(tau_test_path, dpi=150)
    plt.close()
    print(f"Plot saved: {tau_test_path}")

    return cutoffs


if __name__ == "__main__":
    episodes_path = Path("data/external/frailty_episodes.parquet")
    output_path = Path("ai/model/recovery_cutoffs.json")
    plot_dir = Path("plots/training")

    cutoffs = train_cutoffs(episodes_path, output_path, plot_dir)
    print("\nTraining complete!")
