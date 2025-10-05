import numpy as np
import struct
from types import SimpleNamespace
import sys
import os
import io
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import src.TSpec as app  # anta att din kod ligger i app.py


# --- Setup fake args för tester ---
def setup_fake_args():
    from collections import deque

    app.args = SimpleNamespace(
        line=False,
        bar=True,
        color_waterfall=False,
        color_spectrum=False,
        store=None,
        load=None,
        waterfall_height=10,
        bins=app.WIDTH,
        spectrum_height=10,
        spectrum_symbol=".",
        spectrum_symbol_color_background=False,
        max_delta_db=None,
        auto_zoom=False,
        auto_zoom_threshold=10.0,
        auto_zoom_iterations=0,
        freq_min=None,
        freq_max=None,
        colormap="viridis",
        custom_colormap=None,
        line_width=1,
        refresh_rate=None,
        hide_spectrum=False,
        hide_waterfall=False,
        smoothing=0.0,
        decimate=1,
        fft_size=None,
        fft_overlap=0.0,
        window="hann",
        no_normalize=False,
        waterfall_scale="log",
        timestamp=False,
        db_min=120,
        db_max=0,
        agc=None,
        avg_blocks=10,
        maxhold=None,
        byteorder="little",
        feature_avg_offset=10,
        clear_on_new_frame=False,
        rssi=True,
        waterfall_speed=1,
        dtype="float32",
        ignore_missing_meta=True,
        feature_symbol=None,
    )

    if not hasattr(app, "waterfall") or app.waterfall is None:
        app.waterfall = deque(maxlen=app.args.waterfall_height)

    if not hasattr(app, "THRESHOLDS") or app.THRESHOLDS is None:
        app.THRESHOLDS = app.DEFAULT_THRESHOLDS.copy()

    # Reset global state för isolerade tester
    app.prev_interp = None
    app.maxhold_spectrum = None
    app.autozoom_count = 0


# --- Grundläggande tester ---
def test_hex_to_rgb():
    rgb = app.hex_to_rgb("#FF0000")
    assert pytest.approx(rgb) == [1.0, 0.0, 0.0]


def test_clamp_delta():
    old = np.array([0.0, 0.0, 0.0])
    new = np.array([10.0, -10.0, 5.0])
    clamped = app.clamp_delta(new, old, max_delta=2.0)
    assert np.all(clamped == np.array([2.0, -2.0, 2.0]))


def test_get_colormap_rgb_default():
    cmap = app.get_colormap_rgb("viridis", steps=8)
    assert cmap.shape == (8, 3)
    assert np.all((cmap >= 0) & (cmap <= 1))


def test_vertical_spectrum_basic(monkeypatch):
    setup_fake_args()
    power = np.linspace(-80, 0, app.WIDTH)
    freqs = np.linspace(0, 1000, app.WIDTH)
    result = app.vertical_spectrum(power, freqs)
    lines = result.splitlines()
    assert len(lines) == app.args.spectrum_height + 2  # spectrum height + 2 label rows


def test_add_waterfall_symbols(monkeypatch):
    setup_fake_args()
    start_len = len(app.waterfall)
    app.add_waterfall(np.linspace(-80, -50, app.WIDTH))
    assert len(app.waterfall) == start_len + 1


def test_parse_vita49_packet_roundtrip():
    stream_id = b"TESTSTREAMID1234"
    pkt_no = 5
    sr = 48000.0
    cf = 1e6
    payload = b"12345678" * 4
    hdr = struct.pack("<I16sIff", 0x12345678, stream_id, pkt_no, sr, cf)
    data = hdr + payload
    parsed = app.parse_vita49_packet(data)
    sid, pkt, srate, cfreq, pay = parsed
    assert sid == stream_id
    assert pkt == pkt_no
    assert srate == sr
    assert cfreq == cf
    assert pay == payload


def test_process_iq_runs(monkeypatch, capsys):
    setup_fake_args()
    N = 256
    iq = np.exp(2j * np.pi * 0.05 * np.arange(N))
    meta = {"stream_id": "test", "sample_rate": 48000.0, "center_frequency": 0.0}
    app.process_iq(iq, meta)
    out = capsys.readouterr().out
    assert "Stream test" in out
    assert "Spectrum (dB):" in out


# --- Nyare funktionalitet tester ---
def test_process_iq_with_autozoom(monkeypatch, capsys):
    setup_fake_args()
    app.args.auto_zoom = True
    app.args.auto_zoom_iterations = 1
    N = 128
    iq = np.ones(N) + 1j * np.zeros(N)
    meta = {"stream_id": "test", "sample_rate": 48000.0, "center_frequency": 1e6}
    app.autozoom_count = 0
    app.process_iq(iq, meta)
    assert app.autozoom_count == 1
    out = capsys.readouterr().out
    assert "Stream test" in out


def test_process_iq_with_smoothing(monkeypatch, capsys):
    setup_fake_args()
    app.args.smoothing = 0.5
    N = 32
    iq = np.ones(N)
    meta = {"stream_id": "smoothing", "sample_rate": 48000.0, "center_frequency": 0.0}
    # kör två gånger för EMA-effekt
    app.process_iq(iq, meta)
    first_out = capsys.readouterr().out
    app.process_iq(iq, meta)
    second_out = capsys.readouterr().out
    assert "Stream smoothing" in first_out
    assert "Stream smoothing" in second_out


def test_process_iq_with_max_delta(monkeypatch, capsys):
    setup_fake_args()
    app.args.max_delta_db = 1.0
    N = 16
    iq = np.linspace(0, 10, N) + 1j * np.zeros(N)
    meta = {"stream_id": "maxdelta", "sample_rate": 48000.0, "center_frequency": 0.0}
    app.prev_interp = None
    app.process_iq(iq, meta)
    # prev_interp ska uppdateras
    assert app.prev_interp is not None


def test_load_iq_from_file(tmp_path, monkeypatch):
    setup_fake_args()
    # skapa dummy fil
    file_path = tmp_path / "test.iq"
    meta = {
        "num_samples": 4,
        "sample_rate": 1000.0,
        "center_frequency": 1e3,
        "stream_id": "f",
    }
    with open(file_path, "wb") as f:
        f.write((json.dumps(meta) + "\n").encode("utf-8"))
        arr = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]], dtype=np.float32
        )
        f.write(arr.tobytes())
    app.args.load = str(file_path)
    app.args.start_sample = 0
    app.args.duration = None
    # kör load
    app.load_iq_from_file()
    assert len(app.waterfall) > 0
