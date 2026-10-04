"""Plot PPG + motion and estimate HR.

Usage:  python plot_ppg.py recording.csv            (whole file)
        python plot_ppg.py recording.csv 15 45      (only 15-45 s)
Shaded regions = IMU says you were moving.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch

FS = 100
GYRO_MOVING_DPS = 15      # rotation above this = moving
ACC_MOVING_G = 0.08       # |accel| deviating from 1 g by this = moving

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "recording.csv")
t = (df["idx"] - df["idx"].iloc[0]) / FS
if len(sys.argv) >= 4:
    keep = (t >= float(sys.argv[2])) & (t <= float(sys.argv[3]))
    df, t = df[keep], t[keep]
t = t.to_numpy()
ir = df["ir"].to_numpy(dtype=float)

# PPG: 0.7-3 Hz bandpass (42-180 bpm), inverted so beats point up
b, a = butter(3, [0.7, 3.0], btype="band", fs=FS)
pulse = -filtfilt(b, a, ir)
peaks, _ = find_peaks(pulse, distance=int(0.5 * FS),
                      prominence=0.5 * np.std(pulse))
ibi = np.diff(peaks) / FS
hr = 60 / np.median(ibi) if len(ibi) else float("nan")
f, P = welch(pulse, fs=FS, nperseg=len(pulse), nfft=2**14)
band = (f >= 0.7) & (f <= 3.0)
hr_spec = 60 * f[band][np.argmax(P[band])]

has_imu = {"ax", "gx"}.issubset(df.columns)
if has_imu:
    acc_mag = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2).to_numpy()
    gyr_mag = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2).to_numpy()
    moving = (gyr_mag > GYRO_MOVING_DPS) | (np.abs(acc_mag - 1) > ACC_MOVING_G)
    # smooth: moving if any sample in the surrounding 0.5 s was moving
    win = int(0.5 * FS)
    moving = np.convolve(moving, np.ones(win), mode="same") > 0

print(f"Samples: {len(df)}  duration: {t[-1] - t[0]:.1f}s")
print(f"IR DC level: {ir.mean():.0f}  (near 262143 = saturated)")
print(f"Beats: {len(peaks)}  median HR: {hr:.1f} bpm  spectral HR: {hr_spec:.1f} bpm")
if has_imu:
    print(f"Moving: {100 * moving.mean():.0f}% of the recording")

n = 3 if has_imu else 2
fig, ax = plt.subplots(n, 1, sharex=True, figsize=(12, 2.6 * n))
ax[0].plot(t, ir, lw=0.8)
ax[0].set_ylabel("IR raw")
ax[1].plot(t, pulse, lw=0.8)
ax[1].plot(t[peaks], pulse[peaks], "rx")
ax[1].set_ylabel("IR bandpassed")
ax[1].set_title(f"Median HR {hr:.1f} bpm  |  spectral {hr_spec:.1f} bpm")
if has_imu:
    ax[2].plot(t, gyr_mag, lw=0.8, label="gyro |w| (deg/s)")
    ax[2].plot(t, 100 * np.abs(acc_mag - 1), lw=0.8,
               label="accel |a|-1g (x100)")
    ax[2].set_ylabel("motion")
    ax[2].legend(loc="upper right")
    for axis in ax:
        axis.fill_between(t, 0, 1, where=moving, color="orange", alpha=0.2,
                          transform=axis.get_xaxis_transform())
ax[-1].set_xlabel("time (s)")
plt.tight_layout()
plt.show()