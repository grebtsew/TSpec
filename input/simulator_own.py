#!/usr/bin/env python3
import numpy as np
import json
import time
import socket
from datetime import datetime
import random
import uuid

MAX_PACKET_SIZE = 8192
DATA_CHUNK_SIZE = MAX_PACKET_SIZE - 32


def generate_broad_signal(t, center_freqs, bw=20e3, points=50):
    """
    Skapar breda signaler runt center_freqs med bandbredd bw.
    Broadcasting används för snabb summering utan np.dot.
    """
    signal = np.zeros_like(t, dtype=complex)
    for f0 in center_freqs:
        freqs = np.linspace(f0 - bw / 2, f0 + bw / 2, points)
        phases = np.random.uniform(0, 2 * np.pi, points)[:, None]  # shape (points,1)
        # Skapa signal med broadcasting: (points, len(t)) * (points,1) → summera över points
        channel_signal = np.sum(
            np.exp(2j * np.pi * freqs[:, None] * t) * np.exp(1j * phases), axis=0
        )
        signal += channel_signal
    signal /= len(center_freqs)
    return signal


def generate_iq_data(sample_rate=1e6, duration=0.05):
    t = np.arange(0, duration, 1 / sample_rate)

    # Smalbandig signal (10 kHz)
    signal1 = np.exp(2j * np.pi * 10e3 * t)

    # Tre breda kanaler (exempel: 20,50,80 kHz)
    center_freqs = [20e3, 50e3, 80e3, 400e3]
    signal2 = generate_broad_signal(t, center_freqs, bw=20e3, points=50)

    # Brus
    noise = 0.05 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))

    return signal1 + 0.5 * signal2 + noise


def generate_metadata(
    stream_id, center_freq, sample_rate, bandwidth, lat, lon, total_packets
):
    return {
        "stream_id": stream_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "center_frequency": center_freq,
        "sample_rate": sample_rate,
        "bandwidth": bandwidth,
        "location": {"latitude": lat, "longitude": lon},
        "packet_count": total_packets,
    }


def send_data(iq_data, metadata, ip="127.0.0.1", port=5005):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(metadata).encode("utf-8"), (ip, port))

    iq_bytes = (
        np.stack((iq_data.real, iq_data.imag), axis=1).astype(np.float32).tobytes()
    )
    stream_id = metadata["stream_id"].encode("utf-8")

    for i in range(0, len(iq_bytes), DATA_CHUNK_SIZE):
        chunk = iq_bytes[i : i + DATA_CHUNK_SIZE]
        header = stream_id[:16].ljust(16, b" ") + (i // DATA_CHUNK_SIZE).to_bytes(
            4, "big"
        )
        sock.sendto(header + chunk, (ip, port))


def main():
    center_freq = 0
    sample_rate = 1000e3
    bandwidth = 200e3
    duration = 0.05

    while True:
        start_time = time.time()

        lat = 59.3293 + random.uniform(-0.0005, 0.0005)
        lon = 18.0686 + random.uniform(-0.0005, 0.0005)

        iq_data = generate_iq_data(sample_rate=sample_rate, duration=duration)
        stream_id = uuid.uuid4().hex[:16]
        total_packets = (len(iq_data) * 2 * 4 + DATA_CHUNK_SIZE - 1) // DATA_CHUNK_SIZE

        metadata = generate_metadata(
            stream_id, center_freq, sample_rate, bandwidth, lat, lon, total_packets
        )
        send_data(iq_data, metadata)

        print(f"Skickade {total_packets} datapaket med stream_id {stream_id}")

        elapsed = time.time() - start_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)


if __name__ == "__main__":
    main()
