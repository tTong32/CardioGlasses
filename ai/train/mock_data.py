"""Generate mock recovery episodes for testing when WFDB files aren't available."""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_mock_episodes(n_episodes: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic recovery episodes with realistic characteristics.

    Args:
        n_episodes: number of episodes to generate
        seed: random seed

    Returns:
        DataFrame with same schema as extract.py output
    """
    rng = np.random.RandomState(seed)

    episodes = []

    for i in range(n_episodes):
        # Patient demographics
        age = rng.randint(65, 85)
        meds_flag = rng.rand() < 0.6  # 60% on HR-altering meds
        test_type = rng.choice(['VELO', '6MWT', 'STAIR'])

        # Baseline HR
        baseline = rng.normal(72, 8)
        baseline_std = rng.uniform(2, 6)

        # Exercise characteristics
        moving_s = rng.uniform(30, 180)
        rise_bpm = rng.uniform(15, 50)
        hr0 = baseline + rise_bpm

        # Recovery tau - influenced by age and medications
        # Older patients and those on meds tend to have slower recovery
        base_tau = rng.gamma(3, 15)  # mean ~45s, right-skewed
        age_factor = 1 + (age - 65) / 100  # +20% for 85 vs 65
        meds_factor = 1.3 if meds_flag else 1.0

        tau = base_tau * age_factor * meds_factor
        tau = np.clip(tau, 10, 300)

        # Generate recovery curve
        t_s = np.arange(0, min(180, 5 * tau), 2)  # sample every 2s
        hr_clean = baseline + rise_bpm * np.exp(-t_s / tau)

        # Add realistic noise and occasional dropouts
        hr = hr_clean + rng.normal(0, 2, size=len(t_s))
        dropout_mask = rng.rand(len(t_s)) > 0.05  # 5% dropout
        hr = hr[dropout_mask]
        t_s = t_s[dropout_mask]

        episode = {
            'patient_id': f'mock_{i // 3:03d}',  # ~3 episodes per patient
            'test': test_type,
            'age': age,
            'meds_flag': meds_flag,
            'baseline': baseline,
            'baseline_std': baseline_std,
            'hr0': hr0,
            'rise_bpm': rise_bpm,
            'moving_s': moving_s,
            't_s': t_s.tolist(),
            'hr': hr.tolist()
        }

        episodes.append(episode)

    df = pd.DataFrame(episodes)
    return df


if __name__ == "__main__":
    # Generate and save mock data
    output_path = Path("data/external/frailty_episodes.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_mock_episodes(200)
    df.to_parquet(output_path, index=False)

    print(f"Generated {len(df)} mock episodes")
    print(f"Saved to {output_path}")
    print(f"\nSummary:")
    print(f"  Mean age: {df['age'].mean():.1f}")
    print(f"  On meds: {df['meds_flag'].sum()} ({df['meds_flag'].mean()*100:.0f}%)")
    print(f"  Episodes by test: {df['test'].value_counts().to_dict()}")
