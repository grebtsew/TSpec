#!/usr/bin/env python3
"""
Streamar realistisk syntetisk IQ-data till en fil i float32-format.
Kompatibel med iqfile-läget i din parser (ingen JSON i själva datan).
"""

import numpy as np
import time
import argparse
import json
import sys
import os

SAMPLE_RATE = 48000  # Hz
PACKET_SIZE = 4096  # samples per chunk
NOISE_LEVEL = 0.05  # Gaussian noise std
DELAY = PACKET_SIZE / SAMPLE_RATE  # seconds per chunk

# Några realistiska toner
TONE_FREQS = [2000, 5000, 12000]
_phase = np.zeros(len(TONE_FREQS))  # running phase continuity


def generate_chunk():
    """Generera ett block av realistisk IQ-data."""
    global _phase
    t = np.arange(PACKET_SIZE) / SAMPLE_RATE
    iq = np.zeros(PACKET_SIZE, dtype=complex)

    # Kombination av flera toner med långsam modulation
    for i, f in enumerate(TONE_FREQS):
        amp = 0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * t[0] + time.time() * 0.15 * (i + 1))
        phase_increment = 2 * np.pi * f / SAMPLE_RATE
        phases = _phase[i] + np.arange(PACKET_SIZE) * phase_increment
        _phase[i] = (phases[-1] + phase_increment) % (2 * np.pi)
        iq += amp * np.exp(1j * phases)

    # Lägg till Gaussian-brus
    iq += NOISE_LEVEL * (
        np.random.randn(PACKET_SIZE) + 1j * np.random.randn(PACKET_SIZE)
    )

    # Slumpmässiga “bursts”
    if np.random.rand() < 0.1:
        burst_freq = np.random.uniform(1000, 15000)
        burst_amp = np.random.uniform(0.3, 1.0)
        burst_phase_inc = 2 * np.pi * burst_freq / SAMPLE_RATE
        burst_phase = np.arange(PACKET_SIZE) * burst_phase_inc
        iq += burst_amp * np.exp(1j * burst_phase)

    interleaved = np.column_stack((iq.real, iq.imag)).astype(np.float32)
    return interleaved.tobytes()


def main():
    parser = argparse.ArgumentParser(
        description="Stream realistic synthetic IQ data to file"
    )
    parser.add_argument(
        "--filepath", type=str, default="../simulation.iq", help="Output IQ file path"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds (optional, infinite if omitted)",
    )
    args = parser.parse_args()

    iq_path = args.filepath
    meta_path = iq_path + ".meta"

    # Skriv metadata (en gång)
    meta = {
        "stream_id": "synthetic_stream",
        "center_frequency": 0.0,
        "sample_rate": SAMPLE_RATE,
        "tone_freqs": TONE_FREQS,
        "packet_size": PACKET_SIZE,
    }
    with open(meta_path, "w") as m:
        json.dump(meta, m, indent=2)
    print(f"📝 Metadata sparad i {meta_path}")

    # Öppna IQ-filen för kontinuerlig skrivning
    with open(iq_path, "wb") as out:
        print(f"📡 Streamar IQ-data till {iq_path} ... (Ctrl+C för att stoppa)")
        start = time.time()
        try:
            while True:
                chunk = generate_chunk()
                out.write(chunk)
                out.flush()
                time.sleep(DELAY)
                if args.duration and (time.time() - start) >= args.duration:
                    break
        except KeyboardInterrupt:
            print("\n🛑 Avbrutet av användare.")


if __name__ == "__main__":
    main()
