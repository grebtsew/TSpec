#!/usr/bin/env python3
import numpy as np
import time
import socket
from datetime import datetime
import random
import uuid
import struct

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MAX_PACKET_SIZE = 8192
PAYLOAD_SIZE = MAX_PACKET_SIZE - 32  # header 32 bytes

def generate_broad_signal(t, center_freqs, bw=20e3, points=50):
    signal = np.zeros_like(t, dtype=complex)
    for f0 in center_freqs:
        freqs = np.linspace(f0 - bw/2, f0 + bw/2, points)
        phases = np.random.uniform(0, 2*np.pi, points)[:, None]
        channel_signal = np.sum(np.exp(2j*np.pi*freqs[:, None]*t) * np.exp(1j*phases), axis=0)
        signal += channel_signal
    signal /= len(center_freqs)
    return signal

def generate_iq_data(sample_rate=1e6, duration=0.05):
    t = np.arange(0, duration, 1/sample_rate)
    signal1 = np.exp(2j * np.pi * 10e3 * t)
    center_freqs = [20e3, 50e3, 80e3, 400e3]
    signal2 = generate_broad_signal(t, center_freqs, bw=20e3, points=50)
    noise = 0.05 * (np.random.randn(len(t)) + 1j*np.random.randn(len(t)))
    return signal1 + 0.5*signal2 + noise

def create_vita49_header(stream_id_bytes, packet_count, sample_rate, center_freq):
    """
    Enkel VITA-49 header, 32 bytes:
    0-15: stream_id (utf-8, padded)
    16-19: packet number
    20-23: sample rate (float32)
    24-27: center freq (float32)
    28-31: reserved
    """
    header = bytearray(32)
    header[:16] = stream_id_bytes[:16].ljust(16, b' ')
    header[16:20] = packet_count.to_bytes(4, 'big')
    header[20:24] = struct.pack(">f", sample_rate)
    header[24:28] = struct.pack(">f", center_freq)
    header[28:32] = b'\x00'*4
    return bytes(header)

def send_vita49_data(iq_data, stream_id, sample_rate=1e6, center_freq=100e6):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    iq_bytes = np.stack((iq_data.real, iq_data.imag), axis=1).astype(np.float32).tobytes()
    total_packets = (len(iq_bytes) + PAYLOAD_SIZE - 1) // PAYLOAD_SIZE
    stream_id_bytes = stream_id.encode("utf-8")
    for pkt_no in range(total_packets):
        start = pkt_no * PAYLOAD_SIZE
        end = start + PAYLOAD_SIZE
        chunk = iq_bytes[start:end]
        header = create_vita49_header(stream_id_bytes, pkt_no, sample_rate, center_freq)
        sock.sendto(header + chunk, (UDP_IP, UDP_PORT))

def main():
    sample_rate = 1e6
    duration = 0.05
    center_freq = 100e6

    while True:
        start_time = time.time()
        iq_data = generate_iq_data(sample_rate=sample_rate, duration=duration)
        stream_id = uuid.uuid4().hex[:16]
        send_vita49_data(iq_data, stream_id, sample_rate=sample_rate, center_freq=center_freq)
        print(f"Skickade stream {stream_id} med {len(iq_data)} IQ-samples")
        elapsed = time.time() - start_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

if __name__ == "__main__":
    main()
