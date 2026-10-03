# Recovery Model Training Pipeline

This directory contains the training pipeline for the CardioGlasses heart rate recovery model.

## Overview

The recovery model fits exponential decay curves to heart rate data after physical exertion and classifies recovery speed against population percentiles and personal baselines. This implementation trains on data from cardiac rehabilitation patients after open-heart surgery.

## Dataset

**Source:** Wearable-based signals during physical exercises from patients with frailty after open-heart surgery
**Version:** 1.0.0
**DOI:** 10.13026/mp8k-7p27
**PhysioNet:** https://physionet.org/content/wearable-exercise-frailty/1.0.0/

**Contents:**
- 80 patients aged ≥65 post-cardiac surgery
- Polar H10 chest strap data: ECG (130 Hz) + accelerometer (200 Hz)
- Exercise tests: VELO (cycling), 6MWT (walk), STAIR
- Patient metadata: age, gender, medications, frailty scores
- ~2.3 GB total

## Quick Start

### 1. Download Dataset

**Option A: Using wget (recommended)**
```bash
# Install wget if needed
brew install wget  # macOS
# or: apt-get install wget  # Linux

# Download dataset
python3 -m ai.train.download
```

**Option B: Using AWS CLI**
```bash
# Install AWS CLI if needed
brew install awscli  # macOS

# Download dataset (faster, no rate limits)
python3 -m ai.train.download
```

**Option C: Manual download**
Visit https://physionet.org/content/wearable-exercise-frailty/1.0.0/ and download to `data/external/frailty/`

**Option D: Mock data (for testing)**
```bash
# Generate 200 synthetic episodes for development/testing
python3 -m ai.train.mock_data
```

### 2. Train Cutoffs

```bash
# Extract episodes from WFDB records and train cutoffs
python3 -m ai.train.build_cutoffs
```

This will:
1. Run `extract.py` if `frailty_episodes.parquet` doesn't exist
2. Fit tau to all episodes
3. Compute population percentiles (p25, p50, p75, p90) for 3 groups:
   - `all`: all patients
   - `meds`: patients on HR-altering medications
   - `no_meds`: patients not on HR-altering medications
4. Generate diagnostic plots in `plots/training/`
5. Save cutoffs to `ai/model/recovery_cutoffs.json`

### 3. Run Tests

```bash
# Test core fit functions
python3 -m pytest ai/tests/test_recovery.py::TestFitTau -v

# Test live recovery model
python3 -m pytest ai/tests/test_recovery.py::TestRecoveryModel -v

# Run all tests
python3 -m pytest ai/tests/test_recovery.py -v
```

## Pipeline Components

### download.py
Downloads the PhysioNet dataset. Tries AWS S3 first (fastest), falls back to wget, then to urllib for essential CSVs only.

**Output:** `data/external/frailty/` (excluded from git)

### extract.py
Extracts recovery episodes from WFDB records.

**Process:**
1. Load ECG and accelerometer data for each patient session
2. Detect R-peaks using `wfdb.processing.xqrs_detect`
3. Filter RR intervals: 300-2000ms, reject outliers >25% from 9-neighbor median
4. Resample to 2-second grid (median HR per bin)
5. Detect stillness using accelerometer magnitude std
6. For each exercise test (VELO, 6MWT, STAIR):
   - Extract 60s pre-test baseline
   - Find movement end (stillness for ≥20s)
   - Compute HR0 (median within ±3s of stop)
   - Extract 0-180s recovery window
   - Reject if: movement <30s, rise <10 bpm, or <80% valid data

**Key columns identified in `subject-info.csv`:**
- Patient ID: `Patient ID`
- Age: `Age, years`
- HR-altering meds: `Beta blockers` (0/1 flag)

**Output:** `data/external/frailty_episodes.parquet`

### mock_data.py
Generates synthetic recovery episodes for testing when the full dataset isn't available.

**Output:** `data/external/frailty_episodes.parquet` (200 episodes)

### build_cutoffs.py
Trains population cutoffs from extracted episodes.

**Quality filters:**
- R² ≥ 0.5
- Tau not at bounds (5-600s)

**Outputs:**
- `ai/model/recovery_cutoffs.json`: trained percentiles for all groups
- `plots/training/tau_histogram.png`: distribution with p75/p90 markers
- `plots/training/example_fits.png`: 6 example recoveries (2 fast, 2 median, 2 slow)
- `plots/training/tau_vs_age.png`: recovery speed vs patient age
- `plots/training/tau_by_test.png`: box plot by exercise type

## Trained Cutoffs (Mock Data)

Current cutoffs from 200 synthetic episodes:

| Group   | N   | Age (median) | Tau p25 | Tau p50 | Tau p75 | Tau p90 | HR drop 60s (p50) |
|---------|-----|--------------|---------|---------|---------|---------|-------------------|
| all     | 200 | 74           | 34.3s   | 50.9s   | 72.9s   | 103.1s  | 21.5              |
| meds    | 111 | 74           | 37.0s   | 54.2s   | 82.0s   | 109.4s  | 21.1              |
| no_meds | 89  | 73           | 31.2s   | 45.6s   | 65.9s   | 88.4s   | 21.7              |

**Note:** These are from synthetic data. Re-train with real WFDB data for production.

## File Structure

```
ai/train/
├── README.md              # This file
├── __init__.py
├── download.py            # Dataset download script
├── extract.py             # Episode extraction from WFDB
├── mock_data.py           # Synthetic data generator
└── build_cutoffs.py       # Training pipeline

ai/model/
└── recovery_cutoffs.json  # Trained percentiles (generated)

data/external/
├── frailty/               # PhysioNet dataset (gitignored)
│   ├── subject-info.csv
│   ├── test-availability.csv
│   └── [patient WFDB records]
└── frailty_episodes.parquet  # Extracted episodes (gitignored)

plots/training/            # Diagnostic plots (committed)
├── tau_histogram.png
├── example_fits.png
├── tau_vs_age.png
└── tau_by_test.png
```

## Re-training for Production

When real glasses data becomes available:

1. **Collect real recovery episodes:**
   - Use the live `RecoveryModel` class to collect episodes from actual users
   - Export episodes with same schema as `frailty_episodes.parquet`
   - Include patient age, medications, and verified recovery quality

2. **Re-train cutoffs:**
   ```bash
   # Point to your real data
   python3 -m ai.train.build_cutoffs
   ```

3. **Review constants in `ai/recovery.py`:**
   - `MIN_MOVING_S = 30`: May need adjustment for real activity patterns
   - `MIN_RISE_BPM = 10`: Verify this threshold works for your population
   - `MIN_FIT_S = 20`: How soon to start fitting (affects real-time responsiveness)
   - `TAU_BOUNDS = (5, 600)`: Adjust if you see boundary hits in real data
   - `MIN_R2 = 0.5`: Quality threshold - increase for stricter filtering
   - `REF_MIN_R2 = 0.7`: Personal reference quality threshold
   - `RATIO_SLOW = 1.5, RATIO_VERY_SLOW = 2.5`: Personal degradation thresholds
   - `END_NEAR_BASELINE_S = 20`: Episode completion criteria
   - `MAX_EPISODE_S = 300`: Maximum tracking duration

4. **Update medication keywords in `MEDS_KEYWORDS`** if needed for your region/language

## Troubleshooting

**No WFDB files after download:**
- Ensure wget or AWS CLI is installed
- Check PhysioNet website for access restrictions
- Use mock_data.py for development

**Episode extraction fails:**
- Check that `subject-info.csv` and `test-availability.csv` exist
- Verify WFDB record format matches expected structure
- Review error messages for specific patient/session failures

**Low episode counts:**
- Check stillness threshold in `extract.py` (default chosen from data distribution)
- Review quality filters in `build_cutoffs.py`
- Verify exercise test annotations exist in `.atr` files

**Tests failing:**
- Ensure cutoffs file exists: `ai/model/recovery_cutoffs.json`
- Re-run training if cutoffs are from different data
- Check that simulator scenarios match trained percentiles

## Citation

If using this model with the PhysioNet dataset, please cite:

```
Abreu, A., Martins, R., Felgueiras, M., Mendes, M., & Rocha, A. P. (2022).
Wearable-based signals during physical exercises from patients with frailty
after open-heart surgery (version 1.0.0). PhysioNet.
https://doi.org/10.13026/mp8k-7p27
```
