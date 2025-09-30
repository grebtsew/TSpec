import numpy as np
import struct
import types
import builtins
import io
import sys

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np

import TSpec as app  # anta att din kod ligger i app.py


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
    assert cmap.shape == (8,3)
    assert np.all((cmap >= 0) & (cmap <= 1))

def test_vertical_spectrum_basic(monkeypatch):
    power = np.linspace(-80, 0, app.WIDTH)
    freqs = np.linspace(0, 1000, app.WIDTH)
    monkeypatch.setattr(app.args, "line", False)
    result = app.vertical_spectrum(power, freqs)
    lines = result.splitlines()
    # it should have HEIGHT spectrum rows + 2 label rows
    assert len(lines) == app.args.spectrum_height + 2

def test_add_waterfall_symbols(monkeypatch):
    monkeypatch.setattr(app.args, "color_waterfall", False)
    start_len = len(app.waterfall)
    app.add_waterfall(np.linspace(-80, -50, app.WIDTH))
    assert len(app.waterfall) == start_len + 1

def test_parse_vita49_packet_roundtrip():
    # build fake packet
    stream_id = b"TESTSTREAMID1234"
    pkt_no = 5
    sr = 48000.0
    cf = 1e6
    payload = b"12345678" * 4
    hdr = struct.pack(
        "<I16sIff",
        0x12345678,
        stream_id,
        pkt_no,
        sr,
        cf
    )
    data = hdr + payload
    parsed = app.parse_vita49_packet(data)
    sid, pkt, srate, cfreq, pay = parsed
    assert sid == stream_id
    assert pkt == pkt_no
    assert srate == sr
    assert cfreq == cf
    assert pay == payload

def test_process_iq_runs(monkeypatch, capsys):
    # Fake args to disable file writing etc
    monkeypatch.setattr(app.args, "store", None)
    monkeypatch.setattr(app.args, "refresh_rate", None)
    monkeypatch.setattr(app.args, "line", False)

    N = 256
    iq = np.exp(2j*np.pi*0.05*np.arange(N))  # sinus
    meta = {"stream_id":"test", "sample_rate":48000.0, "center_frequency":0.0}
    app.process_iq(iq, meta)
    out = capsys.readouterr().out
    assert "Stream test" in out
    assert "Spectrum (dB):" in out
