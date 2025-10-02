#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket
import json
import numpy as np
import argparse
import time
import uuid

parser = argparse.ArgumentParser(
    description="Simulator som skickar raw IQ-float32 via UDP"
)
parser.add_argument("--host", type=str, default="127.0.0.1", help="Mottagarens IP")
parser.add_argument("--port", type=int, default=5005, help="UDP-port")
parser.add_argument("--samplerate", type=float, default=1e6, help="Samplerate i Hz")
parser.add_argument(
    "--center-freq", type=float, default=0.0, help="Center frequency [Hz]"
)
parser.add_argument(
    "--bandwidth", type=float, default=200e3, help="Bredd på simulerad signal [Hz]"
)
parser.add_argument(
    "--blocksize", type=int, default=1024, help="Antal sampel per UDP-paket"
)
parser.add_argument(
    "--send-rate", type=float, default=0.01, help="Tid mellan paket [s]"
)
parser.add_argument("--metadata", type=int, default=1, help="Send metadata")

args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# --- Metadata ---
def send_metadata():
    meta = {
        "stream_id": "sim_raw_" + uuid.uuid4().hex[:8],
        "center_frequency": args.center_freq,
        "sample_rate": args.samplerate,
    }
    sock.sendto(json.dumps(meta).encode("utf-8"), (args.host, args.port))
    print(f"Metadata skickad: {meta}")
    return meta["stream_id"]


# --- Konstant basfrekvens som inte ändras under sessionen ---
F_BASE = 50e3  # exempel: 50 kHz


# --- Generera IQ-data ---
def generate_iq(blocksize, samplerate, bandwidth=200e3):
    t = np.arange(blocksize) / samplerate

    # Bas-signal (konstant)
    base_signal = np.exp(2j * np.pi * F_BASE * t)

    # Slumpmässig signal som hoppar lite
    f_rand_small = np.random.uniform(-bandwidth / 4, bandwidth / 4)
    rand_signal_small = np.exp(2j * np.pi * f_rand_small * t)

    # Större hopp-signal (kan hoppa över nästan hela bandbredden)
    f_rand_large = np.random.uniform(0, bandwidth)
    rand_signal_large = np.exp(2j * np.pi * f_rand_large * t)

    # Lätt brus
    noise = 0.01 * (np.random.randn(blocksize) + 1j * np.random.randn(blocksize))

    # Kombinera signalerna
    iq = base_signal + 0.5 * rand_signal_small + 0.3 * rand_signal_large + noise

    # Packa till float32 I/Q-interleaved
    iq_arr = np.zeros((blocksize, 2), dtype=np.float32)
    iq_arr[:, 0] = np.real(iq)
    iq_arr[:, 1] = np.imag(iq)
    return iq_arr


# --- Huvudloop ---
def main():
    print(
        f"Skickar simulering till {args.host}:{args.port} "
        f"SR={args.samplerate} Hz, block={args.blocksize}"
    )

    if args.metadata == 1:
        stream_id = send_metadata()

    try:
        while True:
            iq_block = generate_iq(
                args.blocksize, args.samplerate, bandwidth=args.bandwidth
            )
            sock.sendto(iq_block.tobytes(), (args.host, args.port))
            time.sleep(args.send_rate)
    except KeyboardInterrupt:
        print("\nAvslutar simulator...")


if __name__ == "__main__":
    main()
