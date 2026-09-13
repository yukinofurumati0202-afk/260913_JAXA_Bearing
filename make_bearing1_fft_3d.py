from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

DATA_DIR = Path("2nd_test/2nd_test")
OUTPUT = Path("bearing1_fft_3d.html")
RESULTS = Path("bearing1_fft_3d_summary.txt")
SAMPLE_RATE = 20_000
DISPLAY_FREQUENCY_POINTS = 1_536

files = sorted(DATA_DIR.iterdir())
times = []
fft_amplitudes = []

for path in files:
    signal = np.loadtxt(path, usecols=0)
    times.append(datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S"))
    signal = signal - signal.mean()
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window)) / (window.sum() / 2)
    fft_amplitudes.append(spectrum)

frequencies = np.fft.rfftfreq(signal.size, 1 / SAMPLE_RATE)
fft_amplitudes = np.asarray(fft_amplitudes)

# The FFT is calculated at every bin. Only the display matrix is reduced so
# the full 0-10 kHz spectrum remains represented without freezing the browser.
display_indices = np.linspace(0, frequencies.size - 1, DISPLAY_FREQUENCY_POINTS, dtype=int)
display_frequencies = frequencies[display_indices]
display_amplitudes = fft_amplitudes[:, display_indices]

fig = go.Figure(
    data=[
        go.Surface(
            x=times,
            y=display_frequencies,
            z=display_amplitudes.T,
            colorscale="Turbo",
            colorbar={"title": "FFT amplitude"},
            hovertemplate=(
                "time=%{x}<br>frequency=%{y:.1f} Hz"
                "<br>amplitude=%{z:.6g}<extra></extra>"
            ),
        )
    ]
)
fig.update_layout(
    title="2nd Test - Bearing 1 (Ch 1) full-file FFT 3D map",
    width=1500,
    height=900,
    template="plotly_white",
    scene={
        "xaxis": {"title": "recording time"},
        "yaxis": {"title": "frequency [Hz]", "range": [0, SAMPLE_RATE / 2]},
        "zaxis": {"title": "FFT amplitude"},
        "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.15}},
    },
    margin={"l": 0, "r": 0, "t": 55, "b": 0},
)
fig.write_html(OUTPUT, include_plotlyjs="cdn", full_html=True)

with RESULTS.open("w", encoding="utf-8") as output:
    output.write(f"files={len(files)}\n")
    output.write(f"samples_per_file={signal.size}\n")
    output.write(f"sampling_rate_hz={SAMPLE_RATE}\n")
    output.write(f"fft_bins_full={frequencies.size}\n")
    output.write(f"frequency_range_hz={frequencies[0]:.6f}-{frequencies[-1]:.6f}\n")
    output.write(f"display_frequency_points={DISPLAY_FREQUENCY_POINTS}\n")
    output.write(f"display_frequency_resolution_approx_hz={np.median(np.diff(display_frequencies)):.6f}\n")
    output.write(f"time_start={times[0].isoformat()}\n")
    output.write(f"time_end={times[-1].isoformat()}\n")
    output.write(f"global_max_amplitude={fft_amplitudes.max():.8f}\n")
    max_file, max_bin = np.unravel_index(np.argmax(fft_amplitudes), fft_amplitudes.shape)
    output.write(f"global_max_time={times[max_file].isoformat()}\n")
    output.write(f"global_max_frequency_hz={frequencies[max_bin]:.6f}\n")

print(f"created {OUTPUT}")
print(f"created {RESULTS}")
print(f"files={len(files)} full_fft_shape={fft_amplitudes.shape}")
print(f"frequency_range={frequencies[0]:.2f}-{frequencies[-1]:.2f} Hz")
print(f"display_shape={display_amplitudes.T.shape}")
