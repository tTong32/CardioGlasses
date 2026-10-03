"""Extract recovery episodes from the PhysioNet frailty dataset."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

# Import constants from recovery module
from ai.recovery import MIN_MOVING_S, MIN_RISE_BPM, HR0_WINDOW_S


@dataclass
class Episode:
    """A single recovery episode extracted from the dataset."""
    patient_id: str
    test: str
    age: int
    meds_flag: bool
    baseline: float
    baseline_std: float
    hr0: float
    rise_bpm: float
    moving_s: float
    t_s: list[float]  # recovery time series in seconds
    hr: list[float]  # recovery HR time series


def process_patient_session(
    patient_id: str,
    session_num: int,
    data_dir: Path,
    stillness_threshold: float
) -> list[Episode]:
    """Process one patient session and extract recovery episodes.

    Args:
        patient_id: patient identifier
        session_num: session number
        data_dir: path to data/external/frailty/
        stillness_threshold: accel std threshold for stillness (m/s²)

    Returns:
        List of extracted Episode objects
    """
    episodes = []

    # Construct paths for ECG and accel records
    ecg_record = f"{patient_id}_{session_num}_ecg"
    acc_record = f"{patient_id}_{session_num}_acc"

    ecg_path = data_dir / ecg_record
    acc_path = data_dir / acc_record

    # Check if files exist
    if not (ecg_path.with_suffix('.hea').exists() and acc_path.with_suffix('.hea').exists()):
        return episodes

    try:
        # Load ECG and accelerometer data
        ecg_data = wfdb.rdrecord(str(ecg_path))
        acc_data = wfdb.rdrecord(str(acc_path))

        # Load annotations (test onsets)
        try:
            annotations = wfdb.rdann(str(ecg_path), 'atr')
        except FileNotFoundError:
            # No annotations for this session
            return episodes

        # Extract R-peaks from ECG
        ecg_signal = ecg_data.p_signal[:, 0]  # first channel
        ecg_fs = ecg_data.fs  # sampling frequency

        # Use WFDB's QRS detector
        r_peaks = wfdb.processing.xqrs_detect(ecg_signal, fs=ecg_fs, verbose=False)

        # Compute RR intervals in milliseconds
        rr_intervals = np.diff(r_peaks) / ecg_fs * 1000

        # Filter RR intervals: keep 300-2000 ms
        valid_rr = (rr_intervals >= 300) & (rr_intervals <= 2000)

        # Additional filtering: drop outliers (> 25% from median of 9 neighbors)
        rr_filtered = rr_intervals.copy()
        for i in range(len(rr_intervals)):
            window_start = max(0, i - 4)
            window_end = min(len(rr_intervals), i + 5)
            window = rr_intervals[window_start:window_end]
            median_rr = np.median(window)

            if abs(rr_intervals[i] - median_rr) > 0.25 * median_rr:
                valid_rr[i] = False

        # Convert RR to instantaneous HR
        hr_instantaneous = 60000 / rr_intervals[valid_rr]
        hr_times = r_peaks[1:][valid_rr] / ecg_fs  # in seconds

        # Resample to 2-second grid
        max_time = hr_times[-1] if len(hr_times) > 0 else 0
        time_grid = np.arange(0, max_time, 2)
        hr_2s = np.full(len(time_grid), np.nan)

        for i, t in enumerate(time_grid):
            # Find HR values in [t, t+2)
            mask = (hr_times >= t) & (hr_times < t + 2)
            if np.any(mask):
                hr_2s[i] = np.median(hr_instantaneous[mask])

        # Process accelerometer: compute magnitude and stillness
        acc_signal = acc_data.p_signal  # shape (n_samples, 3)
        acc_fs = acc_data.fs

        # Compute magnitude
        acc_magnitude = np.sqrt(np.sum(acc_signal ** 2, axis=1))

        # Resample accel std to 2-second grid
        acc_std_2s = np.full(len(time_grid), np.nan)

        for i, t in enumerate(time_grid):
            start_sample = int(t * acc_fs)
            end_sample = int((t + 2) * acc_fs)
            if end_sample <= len(acc_magnitude):
                window = acc_magnitude[start_sample:end_sample]
                if len(window) > 0:
                    acc_std_2s[i] = np.std(window)

        # Find test onsets (VELO, 6MWT, STAIR only)
        test_types = ['VELO', '6MWT', 'STAIR']

        for ann_idx, ann_sample in enumerate(annotations.sample):
            ann_symbol = annotations.aux_note[ann_idx] if hasattr(annotations, 'aux_note') else ''

            # Check if this is a test we care about
            test_type = None
            for tt in test_types:
                if tt in ann_symbol:
                    test_type = tt
                    break

            if test_type is None:
                continue

            # Convert annotation sample to time
            onset_time = ann_sample / ecg_fs

            # Find corresponding grid index
            onset_idx = int(onset_time / 2)

            if onset_idx >= len(time_grid):
                continue

            # Extract baseline: 60s before onset
            baseline_start_idx = max(0, onset_idx - 30)  # 60s / 2s
            baseline_hr = hr_2s[baseline_start_idx:onset_idx]
            baseline_hr = baseline_hr[~np.isnan(baseline_hr)]

            if len(baseline_hr) < 10:  # need at least 20s of data
                continue

            baseline = np.median(baseline_hr)
            baseline_std = max(np.std(baseline_hr), 2)

            # Find movement end: first time after onset where accel stays below threshold for ≥20s
            movement_end_idx = None
            search_limit = min(len(time_grid), onset_idx + 450)  # search up to 15 min

            for idx in range(onset_idx, search_limit):
                # Check if next 10 readings (20s) are all below threshold
                if idx + 10 >= len(acc_std_2s):
                    break

                window = acc_std_2s[idx:idx + 10]
                if np.all(~np.isnan(window)) and np.all(window < stillness_threshold):
                    movement_end_idx = idx
                    break

            if movement_end_idx is None:
                continue

            # Compute movement duration
            moving_s = (movement_end_idx - onset_idx) * 2

            if moving_s < MIN_MOVING_S:
                continue

            # Compute HR0: median HR within ±HR0_WINDOW_S of movement end
            hr0_window_bins = int(HR0_WINDOW_S / 2)
            hr0_start = max(0, movement_end_idx - hr0_window_bins)
            hr0_end = min(len(hr_2s), movement_end_idx + hr0_window_bins + 1)
            hr0_window = hr_2s[hr0_start:hr0_end]
            hr0_window = hr0_window[~np.isnan(hr0_window)]

            if len(hr0_window) == 0:
                continue

            hr0 = np.median(hr0_window)
            rise_bpm = hr0 - baseline

            if rise_bpm < MIN_RISE_BPM:
                continue

            # Extract recovery window: 0-180s after movement end
            recovery_end_idx = min(len(time_grid), movement_end_idx + 90)  # 180s / 2s

            # Check for movement during recovery
            recovery_acc = acc_std_2s[movement_end_idx:recovery_end_idx]
            if np.any(~np.isnan(recovery_acc) & (recovery_acc >= stillness_threshold)):
                # Movement detected during recovery, truncate
                movement_during = np.where(~np.isnan(recovery_acc) & (recovery_acc >= stillness_threshold))[0]
                if len(movement_during) > 0:
                    recovery_end_idx = movement_end_idx + movement_during[0]

            # Extract recovery HR
            recovery_hr = hr_2s[movement_end_idx:recovery_end_idx]
            recovery_t = (np.arange(len(recovery_hr)) * 2).astype(float)

            # Filter out NaN values
            valid_mask = ~np.isnan(recovery_hr)
            recovery_hr = recovery_hr[valid_mask]
            recovery_t = recovery_t[valid_mask]

            # Check data quality: need at least 80% of bins
            expected_bins = (recovery_end_idx - movement_end_idx)
            if len(recovery_hr) < 0.8 * expected_bins:
                continue

            # Create episode
            episode = Episode(
                patient_id=patient_id,
                test=test_type,
                age=0,  # will be filled in later from subject-info.csv
                meds_flag=False,  # will be filled in later
                baseline=baseline,
                baseline_std=baseline_std,
                hr0=hr0,
                rise_bpm=rise_bpm,
                moving_s=moving_s,
                t_s=recovery_t.tolist(),
                hr=recovery_hr.tolist()
            )

            episodes.append(episode)

    except Exception as e:
        print(f"Warning: Failed to process {patient_id}_{session_num}: {e}")

    return episodes


def choose_stillness_threshold(data_dir: Path, plot_dir: Path) -> float:
    """Analyze accelerometer data to choose stillness threshold.

    Args:
        data_dir: path to data/external/frailty/
        plot_dir: path to save diagnostic plots

    Returns:
        Chosen stillness threshold (accel std in m/s²)
    """
    print("Analyzing accelerometer data to choose stillness threshold...")

    # Sample a few sessions to analyze
    rest_stds = []
    movement_stds = []

    # Get list of available records
    record_files = list(data_dir.glob("*_acc.hea"))

    if len(record_files) == 0:
        print("Warning: No accelerometer records found. Using default threshold.")
        return 0.05  # default fallback

    # Sample up to 10 records
    sample_records = record_files[:min(10, len(record_files))]

    for acc_file in sample_records:
        record_name = acc_file.stem  # removes .hea
        try:
            acc_data = wfdb.rdrecord(str(data_dir / record_name))
            acc_signal = acc_data.p_signal
            acc_fs = acc_data.fs

            # Compute magnitude
            acc_magnitude = np.sqrt(np.sum(acc_signal ** 2, axis=1))

            # Split into 2s windows
            n_windows = len(acc_magnitude) // int(2 * acc_fs)
            for i in range(n_windows):
                start = i * int(2 * acc_fs)
                end = (i + 1) * int(2 * acc_fs)
                window = acc_magnitude[start:end]
                std_val = np.std(window)

                # Heuristic: very low std = rest, high std = movement
                if std_val < 0.02:
                    rest_stds.append(std_val)
                elif std_val > 0.1:
                    movement_stds.append(std_val)

        except Exception as e:
            print(f"Warning: Failed to load {record_name}: {e}")
            continue

    if len(rest_stds) == 0 or len(movement_stds) == 0:
        print("Warning: Insufficient data for threshold analysis. Using default.")
        return 0.05

    # Choose threshold between rest and movement distributions
    rest_p95 = np.percentile(rest_stds, 95)
    movement_p05 = np.percentile(movement_stds, 5)
    threshold = (rest_p95 + movement_p05) / 2

    # Create diagnostic plot
    plt.figure(figsize=(10, 6))
    plt.hist(rest_stds, bins=50, alpha=0.5, label='Rest', density=True)
    plt.hist(movement_stds, bins=50, alpha=0.5, label='Movement', density=True)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.xlabel('Accel Std (m/s²)')
    plt.ylabel('Density')
    plt.title('Accelerometer Stillness Threshold Selection')
    plt.legend()
    plt.tight_layout()

    plot_path = plot_dir / 'stillness_threshold.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Stillness threshold chosen: {threshold:.4f} m/s²")
    print(f"Plot saved to {plot_path}")

    return threshold


def extract_all_episodes(data_dir: Path, output_path: Path, plot_dir: Path) -> pd.DataFrame:
    """Extract all recovery episodes from the dataset.

    Args:
        data_dir: path to data/external/frailty/
        output_path: path to save parquet file
        plot_dir: path to save diagnostic plots

    Returns:
        DataFrame with all episodes
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Choose stillness threshold
    stillness_threshold = choose_stillness_threshold(data_dir, plot_dir)

    # Load patient info
    subject_info_path = data_dir / "subject-info.csv"
    if not subject_info_path.exists():
        raise FileNotFoundError(f"subject-info.csv not found in {data_dir}")

    subject_info = pd.read_csv(subject_info_path)

    # Inspect column names
    print("\nsubject-info.csv columns:")
    for col in subject_info.columns:
        print(f"  - {col}")

    # Identify key columns (case-insensitive matching)
    col_map = {}
    for col in subject_info.columns:
        col_lower = col.lower()
        if 'patient' in col_lower or 'subject' in col_lower or 'id' in col_lower:
            col_map['patient_id'] = col
        elif 'age' in col_lower:
            col_map['age'] = col
        elif 'med' in col_lower or 'drug' in col_lower:
            col_map['meds'] = col

    print(f"\nIdentified columns:")
    print(f"  Patient ID: {col_map.get('patient_id', 'NOT FOUND')}")
    print(f"  Age: {col_map.get('age', 'NOT FOUND')}")
    print(f"  Medications: {col_map.get('meds', 'NOT FOUND')}")

    # Load test availability
    test_avail_path = data_dir / "test-availability.csv"
    test_avail = pd.read_csv(test_avail_path) if test_avail_path.exists() else None

    # Extract episodes
    all_episodes = []
    kept_count = {'VELO': 0, '6MWT': 0, 'STAIR': 0}
    reject_reasons = {'too_short': 0, 'small_rise': 0, 'poor_quality': 0, 'no_movement_end': 0}

    # Get unique patient IDs from available data files
    ecg_files = list(data_dir.glob("*_ecg.hea"))
    patient_sessions = set()

    for ecg_file in ecg_files:
        # Parse filename: <patient>_<session>_ecg.hea
        parts = ecg_file.stem.split('_')
        if len(parts) >= 2:
            patient_id = '_'.join(parts[:-2])
            session = parts[-2]
            patient_sessions.add((patient_id, session))

    print(f"\nFound {len(patient_sessions)} patient sessions to process...")

    for patient_id, session in sorted(patient_sessions):
        episodes = process_patient_session(patient_id, int(session), data_dir, stillness_threshold)

        # Add patient metadata
        if 'patient_id' in col_map:
            patient_row = subject_info[subject_info[col_map['patient_id']] == patient_id]

            if len(patient_row) > 0:
                age = patient_row[col_map.get('age', 'age')].values[0] if 'age' in col_map else 0
                meds_val = patient_row[col_map.get('meds', 'medications')].values[0] if 'meds' in col_map else False

                # Map medication flag to boolean
                meds_flag = bool(meds_val) if isinstance(meds_val, (bool, int)) else str(meds_val).lower() in ['1', 'true', 'yes']

                for ep in episodes:
                    ep.age = int(age) if not pd.isna(age) else 0
                    ep.meds_flag = meds_flag

        all_episodes.extend(episodes)

        for ep in episodes:
            kept_count[ep.test] += 1

    print(f"\nExtracted {len(all_episodes)} episodes:")
    for test, count in kept_count.items():
        print(f"  {test}: {count}")

    # Convert to DataFrame
    if len(all_episodes) == 0:
        print("Warning: No episodes extracted! This may be because WFDB files are missing.")
        print("Please ensure the full dataset is downloaded.")
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            'patient_id', 'test', 'age', 'meds_flag', 'baseline', 'baseline_std',
            'hr0', 'rise_bpm', 'moving_s', 't_s', 'hr'
        ])

    df = pd.DataFrame([
        {
            'patient_id': ep.patient_id,
            'test': ep.test,
            'age': ep.age,
            'meds_flag': ep.meds_flag,
            'baseline': ep.baseline,
            'baseline_std': ep.baseline_std,
            'hr0': ep.hr0,
            'rise_bpm': ep.rise_bpm,
            'moving_s': ep.moving_s,
            't_s': ep.t_s,
            'hr': ep.hr
        }
        for ep in all_episodes
    ])

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} episodes to {output_path}")

    return df


if __name__ == "__main__":
    data_dir = Path("data/external/frailty")
    output_path = Path("data/external/frailty_episodes.parquet")
    plot_dir = Path("plots/training")

    df = extract_all_episodes(data_dir, output_path, plot_dir)
    print(f"\nExtraction complete! {len(df)} episodes saved.")
