#!/usr/bin/env python3
import socket
import time
import uuid
import struct
import numpy as np
from rtlsdr import RtlSdr

# -----------------------------
# Konfiguration
# -----------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
BUF_SAMPLES = 2048
SAMPLE_RATE = 2e6  # Hz
CENTER_FREQ = 100e6  # Hz
GAIN = "auto"
STREAM_ID = uuid.uuid4().bytes[:16]  # 16-byte unik ID

# -----------------------------
# UDP socket
# -----------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# -----------------------------
# VITA-49 IF Header funktioner
# -----------------------------
def now_vita_timestamp():
    t = time.time()
    sec = int(t)
    frac = int((t - sec) * (1 << 64))
    return sec, frac


def build_vita49_if_packet(iq_samples, pkt_count):
    """Skapar VITA-49 IF-paket med float32 IQ och metadata korrekt 32-byte header"""
    # Float32 I/Q, little-endian
    iq_float32 = np.empty(len(iq_samples) * 2, dtype="<f4")
    iq_float32[0::2] = np.real(iq_samples).astype("<f4")
    iq_float32[1::2] = np.imag(iq_samples).astype("<f4")

    pkt_type = 0x1
    hdr_word0 = 0x40000000 | (pkt_type << 28) | (pkt_count & 0x0FFFFFFF)

    # 32-byte header: word0, stream_id, pkt_count, sample_rate, center_freq
    header = struct.pack(
        "<I16sIff", hdr_word0, STREAM_ID, pkt_count, SAMPLE_RATE, CENTER_FREQ
    )

    return header + iq_float32.tobytes()


# -----------------------------
# RTL-SDR setup
# -----------------------------
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

print(f"RTL-SDR → VITA-49 UDP {UDP_IP}:{UDP_PORT}")
pkt_count = 0

try:
    while True:
        iq = sdr.read_samples(BUF_SAMPLES)
        pkt = build_vita49_if_packet(iq, pkt_count)
        sock.sendto(pkt, (UDP_IP, UDP_PORT))
        pkt_count = (pkt_count + 1) & 0x0F
        time.sleep(BUF_SAMPLES / SAMPLE_RATE)
except KeyboardInterrupt:
    print("Avslutar...")
finally:
    sdr.close()
