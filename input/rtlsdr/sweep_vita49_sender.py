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

FREQ_START = 30e6      # Hz
FREQ_STOP = 1700e6     # Hz
STEP = 2e6             # Hz per steg (≈ instantan bandbredd)
DWELL = 0.05           # sekunder per frekvens (kan ökas till 0.1 för stabilare resultat)

BUF_SAMPLES = 2048
SAMPLE_RATE = 2.4e6    # Hz
GAIN = "auto"
STREAM_ID = uuid.uuid4().bytes[:16]  # 16-byte unik ID

# -----------------------------
# UDP socket
# -----------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------------
# VITA-49 IF Header
# -----------------------------
def build_vita49_if_packet(iq_samples, pkt_count, center_freq):
    """
    Bygger ett förenklat VITA-49 IF-paket med float32 IQ och metadata.
    """
    iq_float32 = np.empty(len(iq_samples) * 2, dtype="<f4")
    iq_float32[0::2] = np.real(iq_samples).astype("<f4")
    iq_float32[1::2] = np.imag(iq_samples).astype("<f4")

    pkt_type = 0x1
    hdr_word0 = 0x40000000 | (pkt_type << 28) | (pkt_count & 0x0FFFFFFF)

    header = struct.pack(
        "<I16sIff", hdr_word0, STREAM_ID, pkt_count, SAMPLE_RATE, center_freq
    )
    return header + iq_float32.tobytes()


# -----------------------------
# RTL-SDR setup
# -----------------------------
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = FREQ_START
sdr.gain = GAIN

freqs = np.arange(FREQ_START, FREQ_STOP, STEP)
pkt_count = 0

print(f"\nRTL-SDR Sweep → VITA-49 UDP {UDP_IP}:{UDP_PORT}")
print(f"Frekvensområde: {FREQ_START/1e6:.1f}–{FREQ_STOP/1e6:.1f} MHz ({len(freqs)} steg)")
print(f"Sveptid per steg: {DWELL:.3f} s\n")

try:
    while True:
        sweep_start = time.time()

        for f in freqs:
            sdr.center_freq = f
            iq = sdr.read_samples(BUF_SAMPLES)
            pkt = build_vita49_if_packet(iq, pkt_count, f)
            sock.sendto(pkt, (UDP_IP, UDP_PORT))
            pkt_count = (pkt_count + 1) & 0x0FFFFFFF
            time.sleep(DWELL)

        sweep_time = time.time() - sweep_start
        print(f"Svep färdigt på {sweep_time:.2f} sekunder ({len(freqs)} steg)")

except KeyboardInterrupt:
    print("Avslutar...")

finally:
    sdr.close()
    sock.close()
