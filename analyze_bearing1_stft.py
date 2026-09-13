from datetime import datetime
from pathlib import Path
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import stft, get_window

DATA_DIR = Path("2nd_test/2nd_test")
OUTPUT = Path("bearing1_ch1_stft.html")
REPORT_DATA = Path("bearing1_bpfo_results.txt")
FS = 20_000
RPM = 2_000

# Commonly reported ZA-2115 geometry; verify against the bearing drawing if available.
ROLLER_COUNT = 16
ROLLER_DIAMETER_IN = 0.3126
PITCH_DIAMETER_IN = 2.815
CONTACT_ANGLE_DEG = 0.0

files = sorted(DATA_DIR.iterdir())
records = []
for path in files:
    signal = np.loadtxt(path, usecols=0)
    records.append((datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S"), path.name, signal))

fr = RPM / 60.0
ratio = ROLLER_DIAMETER_IN / PITCH_DIAMETER_IN * math.cos(math.radians(CONTACT_ANGLE_DEG))
bpfo = ROLLER_COUNT / 2 * fr * (1 - ratio)
bpfi = ROLLER_COUNT / 2 * fr * (1 + ratio)
ftf = fr / 2 * (1 - ratio)
bsf = PITCH_DIAMETER_IN / (2 * ROLLER_DIAMETER_IN) * fr * (1 - ratio**2)
harmonics = [bpfo * i for i in range(1, 4)]

selected_indices = [0, len(records) // 2, len(records) - 3]
selected = [records[i] for i in selected_indices]
window = get_window("hann", 4096)

fig = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=(
        "Initial STFT", "Initial spectrum",
        "Middle STFT", "Middle spectrum",
        "Pre-end STFT", "Pre-end spectrum",
    ),
    vertical_spacing=0.08,
    horizontal_spacing=0.08,
)

results = []
for row, (timestamp, filename, signal) in enumerate(selected, start=1):
    frequencies, stft_times, values = stft(signal, fs=FS, window=window, nperseg=4096, noverlap=3072, boundary=None)
    magnitude = np.abs(values)
    mask = frequencies <= 1_000
    # Spectral average from a 1-second snapshot, with the DC component removed.
    spectrum = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(signal.size)))
    spectrum_freqs = np.fft.rfftfreq(signal.size, 1 / FS)
    spectrum /= np.sum(np.hanning(signal.size)) / 2

    fig.add_trace(
        go.Heatmap(
            x=stft_times,
            y=frequencies[mask],
            z=20 * np.log10(magnitude[mask] + 1e-9),
            colorscale="Viridis",
            colorbar={"title": "dB", "len": 0.25, "y": 1 - (row - 0.5) / 3},
            zmin=-70,
            zmax=-5,
            showscale=True,
            name=filename,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=spectrum_freqs[1:][spectrum_freqs[1:] <= 1_000],
            y=20 * np.log10(spectrum[1:][spectrum_freqs[1:] <= 1_000] + 1e-9),
            mode="lines",
            name=filename,
            line={"width": 1},
        ),
        row=row,
        col=2,
    )

    # Search for the strongest spectral line in a +/- 8 Hz neighborhood.
    bpfo_observations = []
    for harmonic in harmonics:
        band = (spectrum_freqs >= harmonic - 8) & (spectrum_freqs <= harmonic + 8)
        peak_index = np.argmax(spectrum[band])
        band_indices = np.flatnonzero(band)
        peak_frequency = spectrum_freqs[band_indices[peak_index]]
        peak_amplitude = spectrum[band_indices[peak_index]]
        bpfo_observations.append((harmonic, peak_frequency, peak_amplitude))
        if harmonic <= 1_000:
            fig.add_vline(x=harmonic, line_dash="dash", line_color="red", row=row, col=2)
            fig.add_annotation(x=harmonic, y=1.0, yref=f"y{2 * row}", text=f"{harmonic:.1f} Hz", showarrow=False, font={"size": 9, "color": "red"}, row=row, col=2)
    results.append((timestamp, filename, bpfo_observations))

for row in range(1, 4):
    fig.update_xaxes(title_text="time [s]", row=row, col=1, range=[0, 1])
    fig.update_xaxes(title_text="frequency [Hz]", row=row, col=2, range=[0, 1_000])
    fig.update_yaxes(title_text="frequency [Hz]", row=row, col=1, range=[0, 1_000])
    fig.update_yaxes(title_text="amplitude [dB]", row=row, col=2)
fig.update_layout(
    title="2nd Test - Bearing 1 (Ch 1): STFT and BPFO check",
    height=1_650,
    width=1_500,
    template="plotly_white",
    showlegend=False,
)
fig.write_html(OUTPUT, include_plotlyjs=True, full_html=True)

with REPORT_DATA.open("w", encoding="utf-8") as output:
    output.write(f"rotation_frequency_hz={fr:.8f}\n")
    output.write(f"bpfo_hz={bpfo:.8f}\n")
    output.write(f"bpfi_hz={bpfi:.8f}\n")
    output.write(f"ftf_hz={ftf:.8f}\n")
    output.write(f"bsf_hz={bsf:.8f}\n")
    for timestamp, filename, observations in results:
        output.write(f"{timestamp.isoformat()} {filename}\n")
        for harmonic, peak_frequency, peak_amplitude in observations:
            output.write(f"  theoretical={harmonic:.4f} observed_peak={peak_frequency:.4f} amplitude={peak_amplitude:.8f}\n")

print(f"created {OUTPUT}")
print(f"created {REPORT_DATA}")
print(f"fr={fr:.4f} Hz BPFO={bpfo:.4f} Hz 2x={2*bpfo:.4f} Hz 3x={3*bpfo:.4f} Hz")
for timestamp, filename, observations in results:
    print(timestamp, filename, [(round(theoretical, 2), round(observed, 2), round(amplitude, 6)) for theoretical, observed, amplitude in observations])
