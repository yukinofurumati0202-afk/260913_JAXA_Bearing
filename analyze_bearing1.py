from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = Path("2nd_test/2nd_test")
OUTPUT = Path("bearing1_ch1_analysis.html")
SAMPLE_RATE = 20_000

files = sorted(DATA_DIR.iterdir())
times = []
rms_values = []
peak_values = []
kurtosis_values = []
waveforms = []
waveform_labels = []

selected = {0: "first", len(files) // 2: "middle", len(files) - 3: "pre-end"}
for index, path in enumerate(files):
    signal = np.loadtxt(path, usecols=0)
    times.append(datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S"))
    rms_values.append(np.sqrt(np.mean(signal**2)))
    peak_values.append(np.max(np.abs(signal)))
    standardized = (signal - signal.mean()) / (signal.std() + 1e-12)
    kurtosis_values.append(np.mean(standardized**4))
    if index in selected:
        waveforms.append(signal)
        waveform_labels.append(f"{selected[index]}: {path.name}")

times = np.array(times)
rms_values = np.array(rms_values)
peak_values = np.array(peak_values)
kurtosis_values = np.array(kurtosis_values)

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("RMS amplitude over time", "Absolute peak over time", "Kurtosis over time", "Representative Ch 1 waveforms"),
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)
fig.add_trace(go.Scatter(x=times, y=rms_values, mode="lines", name="RMS", line={"color": "#b13a2e"}), row=1, col=1)
fig.add_trace(go.Scatter(x=times[-2:], y=rms_values[-2:], mode="markers", name="last 2 records", marker={"color": "#1f5f8b", "size": 7}), row=1, col=1)
fig.add_trace(go.Scatter(x=times, y=peak_values, mode="lines", name="absolute peak", line={"color": "#d47b23"}), row=1, col=2)
fig.add_trace(go.Scatter(x=times, y=kurtosis_values, mode="lines", name="kurtosis", line={"color": "#4b7f52"}), row=2, col=1)

seconds = np.arange(waveforms[0].size) / SAMPLE_RATE
colors = ["#6c757d", "#b13a2e", "#1f5f8b"]
for signal, label, color in zip(waveforms, waveform_labels, colors):
    fig.add_trace(go.Scatter(x=seconds, y=signal, mode="lines", name=label, line={"color": color, "width": 1}), row=2, col=2)

fig.update_xaxes(title_text="recording time", row=2, col=1)
fig.update_xaxes(title_text="time within 1-second snapshot [s]", range=[0, 1], row=2, col=2)
fig.update_yaxes(title_text="RMS", row=1, col=1)
fig.update_yaxes(title_text="max |amplitude|", row=1, col=2)
fig.update_yaxes(title_text="kurtosis", row=2, col=1)
fig.update_yaxes(title_text="amplitude", row=2, col=2)
fig.update_layout(title="2nd Test - Bearing 1 (Ch 1) vibration analysis", height=850, width=1400, template="plotly_white", hovermode="x unified")
fig.write_html(OUTPUT, include_plotlyjs=True, full_html=True)
print(f"created {OUTPUT}")
print(f"files={len(files)} rms_first={rms_values[0]:.6f} rms_median={np.median(rms_values):.6f} rms_max={rms_values.max():.6f} rms_last={rms_values[-1]:.6f}")
print(f"max_rms_time={times[np.argmax(rms_values)]}")
