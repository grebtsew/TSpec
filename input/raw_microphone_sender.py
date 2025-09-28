#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket
import json
import numpy as np
import sounddevice as sd
import argparse

parser = argparse.ArgumentParser(description="Skicka mikrofonljud som raw IQ-float32 via UDP")
parser.add_argument("--host", type=str, default="127.0.0.1",
                    help="Mottagarens IP (där spektrumscriptet körs)")
parser.add_argument("--port", type=int, default=5005,
                    help="UDP-port (samma som --port i spektrumscriptet)")
parser.add_argument("--samplerate", type=int, default=48000,
                    help="Mikrofonens samplingsfrekvens [Hz]")
parser.add_argument("--blocksize", type=int, default=1024,
                    help="Antal sampel per UDP-paket")
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_metadata():
    """
    Skicka ett JSON-meddelande med stream_id och sample_rate
    så att mottagaren kan sätta rätt frekvensskala.
    """
    meta = {
        "stream_id": "mic_raw",
        "center_frequency": 0.0,          # ingen RF-offset
        "sample_rate": float(args.samplerate)
    }
    sock.sendto(json.dumps(meta).encode("utf-8"), (args.host, args.port))
    print(f"Metadata skickad: {meta}")

def callback(indata, frames, time, status):
    if status:
        print("Status:", status, flush=True)
    # indata är (frames x 1) float32, värden -1..1
    mono = indata[:, 0]

    # Skapa IQ-par: I = mono, Q = 0
    iq = np.zeros((frames, 2), dtype=np.float32)
    iq[:, 0] = mono
    # iq[:, 1] är redan noll

    sock.sendto(iq.tobytes(), (args.host, args.port))

def main():
    print(f"Skickar mikrofondata till {args.host}:{args.port} "
          f"samplerate={args.samplerate} block={args.blocksize}")
    send_metadata()
    with sd.InputStream(channels=1,
                        samplerate=args.samplerate,
                        blocksize=args.blocksize,
                        dtype='float32',
                        callback=callback):
        print("Tryck Ctrl+C för att stoppa")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nAvslutar.")

if __name__ == "__main__":
    main()
