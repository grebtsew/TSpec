#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamar IQ-data från ett SDR via SoapySDR och skickar via UDP som float32 (I/Q-interleaved),
med metadata som skickas vid start, vid förändring och periodiskt.
"""

import SoapySDR
from SoapySDR import *
import numpy as np
import socket
import argparse
import json
import time
import uuid

# --- Argumentparser ---
parser = argparse.ArgumentParser(description="Stream SDR IQ samples via UDP")
parser.add_argument("--driver", type=str, default="rtlsdr", help="SoapySDR driver, e.g. rtlsdr, hackrf, lime")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Mottagarens IP")
parser.add_argument("--port", type=int, default=5005, help="UDP-port")
parser.add_argument("--samplerate", type=float, default=2.048e6, help="Samplerate [Hz]")
parser.add_argument("--center-freq", type=float, default=100e6, help="Center frequency [Hz]")
parser.add_argument("--gain", type=float, default=0, help="RX gain [dB]")
parser.add_argument("--blocksize", type=int, default=4096, help="Antal sampel per UDP-paket")
parser.add_argument("--metadata", type=int, default=1, help="Skicka metadata")
parser.add_argument("--send-rate", type=float, default=None, help="Manuell delay mellan paket (sek)")
parser.add_argument("--metadata-interval", type=float, default=1.0, help="Intervall för att skicka metadata [s]")
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Metadata ---
_last_metadata = None
def send_metadata():
    global _last_metadata
    meta = {
        "stream_id": "sdr_" + uuid.uuid4().hex[:8],
        "center_frequency": args.center_freq,
        "sample_rate": args.samplerate,
        "driver": args.driver,
    }
    if _last_metadata != meta:
        sock.sendto(json.dumps(meta).encode("utf-8"), (args.host, args.port))
        _last_metadata = meta
        print(f"Metadata skickad: {meta}")
    return meta["stream_id"]

# --- Initiera SDR ---
print(f"🔧 Opening SoapySDR device: driver={args.driver}")
sdr = SoapySDR.Device({"driver": args.driver})

# Konfigurera
sdr.setSampleRate(SOAPY_SDR_RX, 0, args.samplerate)
sdr.setFrequency(SOAPY_SDR_RX, 0, args.center_freq)
try:
    sdr.setGain(SOAPY_SDR_RX, 0, args.gain)
except RuntimeError:
    print("⚠️ SDR gain not adjustable or unsupported for this driver.")

print(f"✅ SDR configured: {args.samplerate/1e6:.2f} MS/s @ {args.center_freq/1e6:.2f} MHz, gain={args.gain} dB")

# --- Setup stream ---
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)
buf = np.empty(args.blocksize, np.complex64)

# --- Skicka metadata vid start ---
if args.metadata == 1:
    send_metadata()
last_metadata_time = time.time()

print(f"🚀 Börjar sända IQ-data till {args.host}:{args.port} ... (Ctrl+C för att stoppa)")

# --- Huvudloop ---
try:
    while True:
        # --- Periodisk metadata ---
        now = time.time()
        if args.metadata and (now - last_metadata_time >= args.metadata_interval):
            send_metadata()
            last_metadata_time = now

        # --- Läs SDR och skicka IQ ---
        sr = sdr.readStream(rx_stream, [buf], args.blocksize)
        if sr.ret > 0:
            iq_block = buf[:sr.ret]
            iq_arr = np.zeros((sr.ret, 2), dtype=np.float32)
            iq_arr[:, 0] = np.real(iq_block)
            iq_arr[:, 1] = np.imag(iq_block)
            sock.sendto(iq_arr.tobytes(), (args.host, args.port))
        else:
            print(f"⚠️ readStream returned {sr.ret}")

        if args.send_rate:
            time.sleep(args.send_rate)

except KeyboardInterrupt:
    print("\n🛑 Avslutar...")
finally:
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)
