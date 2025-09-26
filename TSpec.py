#!/usr/bin/env python3
import socket, json, sys, argparse
import numpy as np
from collections import deque

# python py_terminal.py --thresholds '\-45:|,\-50:-,\-65:.' --waterfall-height 40 --bins 200 --color-waterfall --color-spectrum --colormap inferno --spectrum-symbol " " --spectrum-symbol-color-background
 
# --- Argumentparser ---
parser = argparse.ArgumentParser(description="Terminalbaserat spektrum och waterfall")
parser.add_argument("--thresholds", type=str, help="Trösklar och symboler, t.ex. '-50:|,-72:-,-77:.'")
parser.add_argument("--waterfall-height", type=int, default=40, help="Max antal rader i waterfall")
parser.add_argument("--bins", type=int, default=80, help="Antal frekvenspunkter för display (bredd)")
parser.add_argument("--color-waterfall", action="store_true", help="Visa waterfall i färg istället för symboler")
parser.add_argument("--color-spectrum", action="store_true", help="Visa spektrum i färg")
parser.add_argument("--colormap", type=str, default="viridis", help="Colormap för färg: viridis, magma, plasma, inferno")
parser.add_argument("--spectrum-symbol", type=str, default=".", 
                    help="Symbol som används för färgat spektrum, default '.'")
parser.add_argument("--spectrum-symbol-color-background", action="store_true", help="Visa spektrum symbol bakgrundsfärg")
parser.add_argument("--freq-min", type=float, default=None, help="Lägsta frekvens i spektrum som ska visas (Hz)")
parser.add_argument("--freq-max", type=float, default=None, help="Högsta frekvens i spektrum som ska visas (Hz)")
parser.add_argument("--address", type=str, default="127.0.0.1",
                    help="IP-adress eller värdnamn till radioenheten (t.ex. 192.168.1.50)")
parser.add_argument("--port", type=int, default=5005,
                    help="TCP/UDP-port för anslutning till radioenheten")
parser.add_argument(
    "--format",
    type=str,
    choices=["vita49", "raw", "simulator"],
    default="vita49",
    help="Strömformat/protokoll, t.ex. 'vita49' (VITA-49), 'raw' (råa IQ-sampel), "
         "'simulated' (dataformat corresponding to simulator)"
)


args = parser.parse_args()


if args.freq_min is not None and args.freq_max is not None:
    if args.freq_min >= args.freq_max:
        print(f"Fel: --freq-min ({args.freq_min}) måste vara mindre än --freq-max ({args.freq_max})")
        sys.exit(1)

# --- Defaultvärden ---
DEFAULT_THRESHOLDS = {-50: "|", -72: "-"}

# --- Konvertera argument till dict om angivet ---
if args.thresholds:
    THRESHOLDS = {}
    try:
        pairs = args.thresholds.split(",")
        for p in pairs:
            key, sym = p.split(":")
            key = key.replace("\\", "")
            THRESHOLDS[int(key)] = sym
    except Exception as e:
        print(f"Fel i thresholds-argument, använder default: {e}")
        THRESHOLDS = DEFAULT_THRESHOLDS
else:
    THRESHOLDS = DEFAULT_THRESHOLDS

# --- 24-bit RGB colormap ---
def get_colormap_rgb(name="viridis", steps=64):
    """
    Returnerar en array (steps x 3) med RGB-värden 0..1.
    Inga externa bibliotek krävs.
    """
    name = name.lower()
    if name == "viridis":
        # Viridis approximation (subset av riktiga färger)
        viridis = np.array([
            [0.267004, 0.004874, 0.329415],
            [0.278826, 0.094678, 0.390793],
            [0.282327, 0.165005, 0.430899],
            [0.275191, 0.239885, 0.482299],
            [0.258965, 0.328762, 0.550164],
            [0.253935, 0.438061, 0.601121],
            [0.265991, 0.548853, 0.639049],
            [0.290881, 0.659799, 0.668533],
            [0.317833, 0.759911, 0.690667],
            [0.348249, 0.857299, 0.706673],
            [0.378821, 0.949658, 0.720224],
        ])
        # Interpolera till rätt antal steg
        x = np.linspace(0, 1, viridis.shape[0])
        xnew = np.linspace(0, 1, steps)
        rgb = np.zeros((steps, 3))
        for i in range(3):
            rgb[:, i] = np.interp(xnew, x, viridis[:, i])
        return rgb

    elif name == "magma":
        # Enkel magma-approximation
        magma = np.array([
            [0.001462,0.000466,0.013866],
            [0.063536,0.028426,0.180382],
            [0.144901,0.046292,0.258216],
            [0.313027,0.071655,0.383129],
            [0.532087,0.105983,0.524388],
            [0.784973,0.148263,0.639493],
            [0.993248,0.216618,0.556888],
        ])
        x = np.linspace(0,1,magma.shape[0])
        xnew = np.linspace(0,1,steps)
        rgb = np.zeros((steps,3))
        for i in range(3):
            rgb[:,i] = np.interp(xnew,x,magma[:,i])
        return rgb

    elif name == "plasma":
        plasma = np.array([
            [0.050383,0.029803,0.527975],
            [0.294833,0.074274,0.667240],
            [0.510168,0.128208,0.707174],
            [0.703417,0.182923,0.669366],
            [0.878953,0.270905,0.574048],
            [0.993248,0.515050,0.382914],
        ])
        x = np.linspace(0,1,plasma.shape[0])
        xnew = np.linspace(0,1,steps)
        rgb = np.zeros((steps,3))
        for i in range(3):
            rgb[:,i] = np.interp(xnew,x,plasma[:,i])
        return rgb

    elif name == "inferno":
        inferno = np.array([
            [0.001462,0.000466,0.013866],
            [0.192355,0.063420,0.368055],
            [0.512340,0.130071,0.478355],
            [0.789799,0.278981,0.420268],
            [0.993248,0.550104,0.293768],
        ])
        x = np.linspace(0,1,inferno.shape[0])
        xnew = np.linspace(0,1,steps)
        rgb = np.zeros((steps,3))
        for i in range(3):
            rgb[:,i] = np.interp(xnew,x,inferno[:,i])
        return rgb

    else:
        print(f"Okänd colormap '{name}', använder viridis")
        return get_colormap_rgb("viridis", steps)

COLORMAP_RGB = get_colormap_rgb(args.colormap)

# --- Konstanter ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
BUFFER_SIZE = 8192
WIDTH = args.bins
HEIGHT = 20

stream_buffers = {}
stream_metadata = {}
waterfall = deque(maxlen=args.waterfall_height)

def add_waterfall(power_db):
    row = []
    if args.color_waterfall:
        # Normalisera och mappa till colormap
        min_val, max_val = np.min(power_db), np.max(power_db)
        norm = (power_db - min_val) / (max_val - min_val + 1e-12)
        for n in norm:
            r, g, b = COLORMAP_RGB[int(n*(len(COLORMAP_RGB)-1))]
            row.append(f"\x1b[48;2;{int(r*255)};{int(g*255)};{int(b*255)}m \x1b[0m")
    else:
        # vanliga symboler
        sorted_thresholds = sorted(THRESHOLDS.items())
        for p in power_db:
            symbol = " "
            for thresh, sym in sorted_thresholds:
                if p >= thresh:
                    symbol = sym
            row.append(symbol)
    waterfall.append("".join(row))

def vertical_spectrum(power_db, freqs, f_min=None, f_max=None):

    if f_min is None:
        f_min = freqs[0]
    if f_max is None:
        f_max = freqs[-1]

    min_db = np.min(power_db)
    max_db = np.max(power_db)
    levels = np.clip((power_db - min_db) / (max_db - min_db + 1e-12), 0, 1)
    bars = (levels * (HEIGHT-1)).astype(int)

    rows = []
    symbol = args.spectrum_symbol
    step_db = (max_db - min_db) / (HEIGHT-1)
    for h in range(HEIGHT-1, -1, -1):
        db_label = f"{min_db + h*step_db:5.1f} "
        row = [db_label]
        for i, b in enumerate(bars):
            if b >= h:
                if args.color_spectrum:
                    # Mappa bar-height till colormap-index
                    color_idx = b * len(COLORMAP_RGB) // HEIGHT
                    r, g, bcol = COLORMAP_RGB[color_idx]

                      

                    if args.spectrum_symbol_color_background:
                        row.append(f"\x1b[48;2;{int(r*255)};{int(g*255)};{int(bcol*255)}m{symbol}\x1b[0m")
                    else:
                        row.append(f"\x1b[38;2;{int(r*255)};{int(g*255)};{int(bcol*255)}m{symbol}\x1b[0m")

                else:
                    row.append(f"{symbol}")
            else:
                row.append(" ")
        rows.append("".join(row))

    # Tick-line och labels fortfarande baserade på bins
    tick_line = "      " + "-"*WIDTH
    label_line = [" "] * (WIDTH + 6)
    num_ticks = 6
    tick_freqs = [f_min + i*(f_max-f_min)/(num_ticks-1) for i in range(num_ticks)]

    for f in tick_freqs:
        # Avrunda till närmsta 50 kHz
        f_rounded = round(f/10_000)*10_000
        pos = int((f - f_min) / (f_max - f_min) * (WIDTH-1))
        label = str(int(f_rounded/1e3))  # kHz
        start = 6 + pos - len(label)//2 

        if f == tick_freqs[-1]:
            
            start -= len(str(int(f/1000)))

        # Trimma om start blir negativ
        if start < 0:
            label = label[-start:]
            start = 0

        # Trimma om etiketten går utanför raden
        for i, ch in enumerate(label):
            if start + i < len(label_line):
                label_line[start+i] = ch

    rows.append(tick_line)
    rows.append("".join(label_line) + " kHz")
    return "\n".join(rows)

def print_waterfall():
    print(f"Waterfall (nyast överst, översättning {THRESHOLDS}, max höjd {args.waterfall_height}):")
    for row in reversed(waterfall):
        print("      " + row)

def process_iq(iq_data, meta):
    N = len(iq_data)
    spectrum = np.fft.fft(iq_data)[:N//2]
    freqs = np.fft.fftfreq(N, 1/meta["sample_rate"])[:N//2]
    power_db = 20*np.log10(np.abs(spectrum) + 1e-12)
    power_db -= np.max(power_db)

    # Beräkna peak på hela spektrumet
    peak_idx = np.argmax(power_db)
    peak_freq = freqs[peak_idx]

    # Zooma på display
    f_min = args.freq_min if args.freq_min is not None else freqs[0]
    f_max = args.freq_max if args.freq_max is not None else freqs[-1]

    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs_zoom = freqs[mask]
    power_zoom = power_db[mask]

    # Interpolera till terminalbredd
    bins = np.linspace(freqs_zoom[0], freqs_zoom[-1], WIDTH)
    interp = np.interp(bins, freqs_zoom, power_zoom)

    # Säkerställ att peak alltid visas
    peak_bin = np.searchsorted(bins, peak_freq)
    if 0 <= peak_bin < WIDTH:
        interp[peak_bin] = max(interp[peak_bin], power_db[peak_idx])

    # Resten av displayen som tidigare
    add_waterfall(interp)

    sys.stdout.write("\x1b[2J\x1b[H")
    print(f"Stream {meta['stream_id']}  CF {meta['center_frequency']/1e6:.3f} MHz  "
          f"SR {meta['sample_rate']/1e6:.2f} Msps  Peak: {peak_freq/1e6:.3f} MHz")
    print("Spectrum (dB):")
    print(vertical_spectrum(interp, bins))
    print()
    print_waterfall()
    sys.stdout.flush()

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Lyssnar på UDP {UDP_IP}:{UDP_PORT}")

    while True:
        data, _ = sock.recvfrom(BUFFER_SIZE)
        try:
            meta = json.loads(data.decode("utf-8"))
            stream_id = meta["stream_id"]
            stream_metadata[stream_id] = meta
            stream_buffers[stream_id] = {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            stream_id = data[:16].decode("utf-8").strip()
            pkt_no = int.from_bytes(data[16:20], 'big')
            payload = data[20:]
            if stream_id not in stream_buffers:
                continue
            stream_buffers[stream_id][pkt_no] = payload
            expected = stream_metadata[stream_id]["packet_count"]
            if len(stream_buffers[stream_id]) == expected:
                iq_bytes = b"".join(stream_buffers[stream_id][i] for i in range(expected))
                iq_arr = np.frombuffer(iq_bytes, dtype=np.float32).reshape(-1, 2)
                iq_data = iq_arr[:,0] + 1j*iq_arr[:,1]
                meta = stream_metadata.pop(stream_id)
                stream_buffers.pop(stream_id)
                process_iq(iq_data, meta)

if __name__ == "__main__":
    main()
