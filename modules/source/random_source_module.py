"""
Synthetic sine-wave IQ generator protocol module.
Used with --input ./modules/source/random_source_module.py

Generates a realistic test IQ signal with:
- a carrier (tone) at an offset from center
- optional noise
- slow amplitude and phase variation for realism
"""

import numpy as np
import time

SAMPLE_RATE = 48000  # Hz
TONE_FREQ = 2000  # Hz offset from center frequency
PACKET_SIZE = 4096  # IQ samples per chunk
NOISE_LEVEL = 0.05  # standard deviation of noise
DELAY = PACKET_SIZE / SAMPLE_RATE  # seconds per chunk

_phase = 0.0  # running phase for continuity


def setup():
    global _phase
    _phase = 0.0
    print(
        f"[protocol] Sine IQ generator active: {TONE_FREQ} Hz tone, {SAMPLE_RATE/1e3:.1f} kHz sample rate"
    )


def get_data():
    global _phase

    # Generate time vector
    t = np.arange(PACKET_SIZE) / SAMPLE_RATE

    # Add a slow amplitude modulation for realism
    amp = 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t[0] + time.time() * 0.2)

    # Generate phase-continuous sine wave
    phase_increment = 2 * np.pi * TONE_FREQ / SAMPLE_RATE
    phases = _phase + np.arange(PACKET_SIZE) * phase_increment
    _phase = (phases[-1] + phase_increment) % (2 * np.pi)

    # Create IQ signal (complex baseband)
    iq = amp * np.exp(1j * phases)

    # Add some Gaussian noise
    iq += NOISE_LEVEL * (
        np.random.randn(PACKET_SIZE) + 1j * np.random.randn(PACKET_SIZE)
    )

    # Convert to interleaved float32 bytes (I0, Q0, I1, Q1, ...)
    interleaved = np.column_stack((iq.real, iq.imag)).astype(np.float32)
    time.sleep(DELAY)
    return interleaved.tobytes()


def cleanup():
    print("[protocol] Sine IQ generator stopped.")
