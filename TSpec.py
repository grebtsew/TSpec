#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket, json, sys, argparse
import numpy as np
from collections import deque
import struct
import time

last_update = 0.0

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

parser.add_argument(
    "--auto-zoom",
    action="store_true",
    help="Beräkna brusgolv och zooma automatiskt till området där signaler finns"
)
parser.add_argument(
    "--auto-zoom-threshold",
    type=float,
    default=10.0,
    help="Hur många dB över brusgolv som räknas som signal (default 10 dB)"
)

parser.add_argument(
    "--refresh-rate",
    type=float,
    default=None,
    help="Max uppdateringshastighet i Hz (standard None = så hög som möjligt)."
)

parser.add_argument(
    "--max-delta-db",
    type=float,
    default=None,
    help="Maximalt tillåtet hopp i dB per refresh (None = ingen begränsning) (WARNING: detta kräver mer minne!)"
)

prev_interp = None   

parser.add_argument("--address", type=str, default="127.0.0.1",
                    help="IP-adress eller värdnamn till radioenheten (t.ex. 192.168.1.50)")
parser.add_argument("--port", type=int, default=5005,
                    help="TCP/UDP-port för anslutning till radioenheten")
parser.add_argument(
    "--format",
    type=str,
    choices=["vita49", "raw", "simulator"],
    default="vita49",
    help="Strömformat/protokoll, t.ex. 'vita49' (VITA-49), 'raw' (råa IQ-sampel), 'simulator' (simulerade data)"
)

parser.add_argument(
    "--custom-colormap",
    type=str,
    help="Custom colormap: startcolor,stopcolor,steps. Ex: '#0000FF,#FF0000,64'"
)



parser.add_argument("--spectrum-height", type=int, default=20, help="Höjd i rader på spektrumdisplayen")


# --- Nya argument: endera bar eller line, och line-width (vertikal) ---
parser.add_argument("--bar", action="store_true",
                    help="Visa spektrum i stapel-läge (standard).")
parser.add_argument("--line", action="store_true",
                    help="Visa spektrum i linje-läge (konturlinje).")
parser.add_argument("--line-width", type=int, default=1,
                    help="Vertikal tjocklek (antal rader) på linjen i line-läge. 1 = en rad.")

args = parser.parse_args()

if args.freq_min is not None and args.freq_max is not None:
    if args.freq_min >= args.freq_max:
        print(f"Fel: --freq-min ({args.freq_min}) måste vara mindre än --freq-max ({args.freq_max})")
        sys.exit(1)

# Default: bar-läge om inget anges
if not args.bar and not args.line:
    args.bar = True

# --- Defaultvärden ---
DEFAULT_THRESHOLDS = { -80: " ", -72: "-",-50: "|"}

THRESHOLDS = None
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

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)]

def clamp_delta(new_vals, old_vals, max_delta):
    """
    Begränsar skillnaden mellan nytt och gammalt värde till ±max_delta.
    """
    diff = new_vals - old_vals
    diff = np.clip(diff, -max_delta, max_delta)
    return old_vals + diff

# --- 24-bit RGB colormap ---
def get_colormap_rgb(name="viridis", steps=64):
    """
    Returnerar en array (steps x 3) med RGB-värden 0..1.
    Enkel interpolation av ett litet tabellfragment per colormap.
    """
    name = name.lower()
    def interp_table(table):
        x = np.linspace(0,1,table.shape[0])
        xnew = np.linspace(0,1,steps)
        rgb = np.zeros((steps,3))
        for i in range(3):
            rgb[:,i] = np.interp(xnew,x,table[:,i])
        return rgb

    if name == "custom":
        if(args.custom_colormap):
            custom_meta = str(args.custom_colormap).split(",")
            custom_start, custom_stop, _ = custom_meta

            table = np.array([hex_to_rgb(custom_start), hex_to_rgb(custom_stop)])
            return interp_table(table)

        else:
            # Warn and use default color theme if not defined
            name=="viridis"
            print("WARNING: custom colormap not set!")

        

    if name == "viridis":
        table = np.array([
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
        return interp_table(table)

    elif name == "magma":
        table = np.array([
            [0.001462,0.000466,0.013866],
            [0.063536,0.028426,0.180382],
            [0.144901,0.046292,0.258216],
            [0.313027,0.071655,0.383129],
            [0.532087,0.105983,0.524388],
            [0.784973,0.148263,0.639493],
            [0.993248,0.216618,0.556888],
        ])
        return interp_table(table)

    elif name == "plasma":
        table = np.array([
            [0.050383,0.029803,0.527975],
            [0.294833,0.074274,0.667240],
            [0.510168,0.128208,0.707174],
            [0.703417,0.182923,0.669366],
            [0.878953,0.270905,0.574048],
            [0.993248,0.515050,0.382914],
        ])
        return interp_table(table)

    elif name == "inferno":
        table = np.array([
            [0.001462,0.000466,0.013866],
            [0.192355,0.063420,0.368055],
            [0.512340,0.130071,0.478355],
            [0.789799,0.278981,0.420268],
            [0.993248,0.550104,0.293768],
        ])
        return interp_table(table)

    else:
        print(f"Okänd colormap '{name}', använder viridis")
        return get_colormap_rgb("viridis", steps)

COLORMAP_RGB = get_colormap_rgb(args.colormap)

# --- Konstanter ---
BUFFER_SIZE = 8192
WIDTH = args.bins
HEIGHT = 20

stream_buffers = {}
stream_metadata = {}
waterfall = deque(maxlen=args.waterfall_height)

def add_waterfall(power_db, freqs=None):
    """
    power_db: interpolerat till WIDTH
    freqs: frekvensvärden för dessa bins (valfritt, behövs bara om du vill göra thresholds dynamiska)
    """
    row = []
    if args.color_waterfall:
        min_val, max_val = np.min(power_db), np.max(power_db)
        norm = (power_db - min_val) / (max_val - min_val + 1e-12)
        for n in norm:
            r, g, bb = COLORMAP_RGB[int(n*(len(COLORMAP_RGB)-1))]
            row.append(f"\x1b[48;2;{int(r*255)};{int(g*255)};{int(bb*255)}m \x1b[0m")
    else:
        sorted_thresholds = sorted(THRESHOLDS.items())
        for p in power_db:
            symbol = " "
            for thresh, sym in sorted_thresholds:
                if p >= thresh:
                    symbol = sym
            row.append(symbol)
    waterfall.append("".join(row))

def vertical_spectrum(power_db, freqs, f_min=None, f_max=None):
    HEIGHT = args.spectrum_height  # använd argumentet istället för global HEIGHT

    if f_min is None:
        f_min = freqs[0]
    if f_max is None:
        f_max = freqs[-1]

    min_db = np.min(power_db)
    max_db = np.max(power_db)
    denom = (max_db - min_db + 1e-12)
    levels = np.clip((power_db - min_db) / denom, 0, 1)
    bars = (levels * (HEIGHT - 1)).astype(int)

    rows = []
    symbol = args.spectrum_symbol
    step_db = (max_db - min_db) / (HEIGHT - 1)

    w = max(1, int(args.line_width))
    half = w // 2

    for h in range(HEIGHT - 1, -1, -1):
        db_label = f"{min_db + h * step_db:5.1f} "
        row = [db_label]

        if args.line:
            for i, b in enumerate(bars):
                dist = abs(h - b)
                if dist <= half:
                    base_idx = int(levels[i] * (len(COLORMAP_RGB) - 1))
                    r, g, bb = COLORMAP_RGB[base_idx]

                    if half > 0:
                        fade = 1.0 - 0.25 * dist / (half + 1e-12)
                    else:
                        fade = 1.0

                    r_f = int(r * 255 * fade + 255 * (1 - fade))
                    g_f = int(g * 255 * fade + 255 * (1 - fade))
                    b_f = int(bb * 255 * fade + 255 * (1 - fade))

                    if args.color_spectrum:
                        if args.spectrum_symbol_color_background:
                            row.append(f"\x1b[48;2;{r_f};{g_f};{b_f}m{symbol}\x1b[0m")
                        else:
                            row.append(f"\x1b[38;2;{r_f};{g_f};{b_f}m{symbol}\x1b[0m")
                    else:
                        row.append(symbol)
                else:
                    row.append(" ")
        else:
            # vanliga staplar
            for i, b in enumerate(bars):
                if b >= h:
                    if args.color_spectrum:
                        idx = b * len(COLORMAP_RGB) // HEIGHT
                        r, g, bb = COLORMAP_RGB[idx]
                        if args.spectrum_symbol_color_background:
                            row.append(f"\x1b[48;2;{int(r*255)};{int(g*255)};{int(bb*255)}m{symbol}\x1b[0m")
                        else:
                            row.append(f"\x1b[38;2;{int(r*255)};{int(g*255)};{int(bb*255)}m{symbol}\x1b[0m")
                    else:
                        row.append(symbol)
                else:
                    row.append(" ")

        rows.append("".join(row))

    # Tick-linje och etiketter
    tick_line = "      " + "-" * WIDTH
    label_line = [" "] * (WIDTH + 6)
    num_ticks = 6
    tick_freqs = [f_min + i * (f_max - f_min) / (num_ticks - 1) for i in range(num_ticks)]
    for f in tick_freqs:
        if f >= 10_000:
            f_rounded = round(f / 10_000) * 10_000
        else:
            f_rounded = f
        
        pos = int((f - f_min) / (f_max - f_min + 1e-12) * (WIDTH - 1))
        label = str(int(f_rounded / 1e3))
        start = max(0, min(6 + pos - len(label)//2, len(label_line) - len(label)))
        for j, ch in enumerate(label):
            label_line[start + j] = ch

    rows.append(tick_line)
    rows.append("".join(label_line) + " kHz")
    return "\n".join(rows)


def print_waterfall():
    print(f"Waterfall (nyast överst, översättning {THRESHOLDS}, max höjd {args.waterfall_height}):")
    for row in reversed(waterfall):
        print("      " + row)

def process_iq(iq_data, meta):
    
    global prev_interp, last_update, THRESHOLDS
    
    if args.refresh_rate is not None:
        now = time.time()
        min_interval = 1.0 / args.refresh_rate
        if now - last_update < min_interval:
            return          # hoppa över uppdatering
        last_update = now

    N = len(iq_data)
    if N < 2:
        return
    spectrum = np.fft.fft(iq_data)[:N//2]
    freqs = np.fft.fftfreq(N, 1/meta["sample_rate"])[:N//2]
    power_db = 20*np.log10(np.abs(spectrum) + 1e-12)
    power_db -= np.max(power_db)

    
    # Handle autozoom
    if args.auto_zoom:
        # Beräkna brusgolv som t.ex. median + liten marginal
        noise_floor = np.median(power_db)
        
        # Mask för bin där signalen ligger över brus + tröskel
        signal_mask = power_db > (noise_floor + args.auto_zoom_threshold)

        if np.any(signal_mask):
            # Få fram första och sista frekvens med signal
            sig_freqs = freqs[signal_mask]
            auto_min = sig_freqs.min()
            auto_max = sig_freqs.max()

            # Lägg gärna på lite “marginal” i båda ändar, t.ex. 5 %
            span = auto_max - auto_min
            auto_min -= 0.05 * span
            auto_max += 0.05 * span

            f_min, f_max = auto_min, auto_max
        else:
            # Om ingen signal hittas: visa fullbredd
            f_min, f_max = freqs[0], freqs[-1]


        if THRESHOLDS == DEFAULT_THRESHOLDS:
            symbols = list(DEFAULT_THRESHOLDS.values())
            n = len(symbols)
            
            signal_mask = (freqs >= f_min) & (freqs <= f_max)
            power_interval = power_db[signal_mask]

            if len(power_interval) > 0:
                min_val = np.min(power_interval)
                max_val = np.max(power_interval)

                # Dela intervallet i n steg
                step = (max_val - min_val) / n
                new_thresholds = {}
                for i, sym in enumerate(symbols):
                    thresh = min_val + i * step
                    new_thresholds[int(round(thresh))] = sym

                THRESHOLDS = new_thresholds
    else:
        # Använd manuella flaggor eller default
        f_min = args.freq_min if args.freq_min is not None else freqs[0]
        f_max = args.freq_max if args.freq_max is not None else freqs[-1]

    # Beräkna peak på hela spektrumet
    peak_idx = np.argmax(power_db)
    peak_freq = freqs[peak_idx]

    

    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs_zoom = freqs[mask]
    power_zoom = power_db[mask]

    # Om mask blev tom (t.ex. felaktiga freq-min/max) -> fallback
    if freqs_zoom.size == 0:
        freqs_zoom = freqs
        power_zoom = power_db

    # Interpolera till terminalbredd
    bins = np.linspace(freqs_zoom[0], freqs_zoom[-1], WIDTH)
    interp = np.interp(bins, freqs_zoom, power_zoom)

    # Spara och utför clamp
    if args.max_delta_db is not None:
        if prev_interp is None:
            prev_interp = interp.copy()
        else:
            interp = clamp_delta(interp, prev_interp, args.max_delta_db)
            prev_interp = interp.copy()

    # Säkerställ att peak alltid visas
    peak_bin = np.searchsorted(bins, peak_freq)
    if 0 <= peak_bin < WIDTH:
        interp[peak_bin] = max(interp[peak_bin], power_db[peak_idx])

    # Resten av displayen som tidigare
    add_waterfall(interp, bins)

    sys.stdout.write("\x1b[2J\x1b[H")
    mode = "Line" if args.line else "Bar"
    extra = f"  LineWidth: {args.line_width}" if args.line else ""
    print(f"Stream {meta.get('stream_id','-')}  CF {meta.get('center_frequency',0)/1e6:.3f} MHz  "
          f"SR {meta.get('sample_rate',0)/1e6:.2f} Msps  Peak: {peak_freq/1e6:.3f} MHz  Mode: {mode}{extra}")
    print("Spectrum (dB):")
    print(vertical_spectrum(interp, bins))
    print()
    print_waterfall()
    sys.stdout.flush()

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.address, args.port))
    print(f"Lyssnar på UDP {args.address}:{args.port}")
    print(f"Expecting format: {args.format}")
    while True:
        data, _ = sock.recvfrom(BUFFER_SIZE)

        if args.format == "simulator":
            # --- tidigare simulator-läsare ---
            try:
                meta = json.loads(data.decode("utf-8"))
                stream_id = meta["stream_id"]
                stream_metadata[stream_id] = meta
                stream_buffers[stream_id] = {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                try:
                    stream_id = data[:16].decode("utf-8").strip()
                except Exception:
                    continue
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

        elif args.format == "raw":
            # Kolla om paketet är JSON-metadata (börja med '{')
            try:
                txt = data.decode("utf-8")
                if txt.strip().startswith("{"):
                    meta_json = json.loads(txt)
                    stream_metadata["raw_stream"] = meta_json
                    continue   # Vänta på nästa paket med själva IQ
            except UnicodeDecodeError:
                pass

            iq_arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            iq_data = iq_arr[:, 0] + 1j * iq_arr[:, 1]

            # Använd senaste metadata om det finns, annars ett vettigt default
            meta = stream_metadata.get("raw_stream", {
                "stream_id": "raw_stream",
                "center_frequency": 0.0,
                "sample_rate": 48000.0    # fallback
            })

            process_iq(iq_data, meta)

        elif args.format == "vita49":
            # --- Enkel VITA-49 läsare ---
            if len(data) < 32:
                continue  # för kort paket
            header = data[:32]
            payload = data[32:]

            try:
                stream_id = header[:16].decode("utf-8").strip()
                pkt_no = int.from_bytes(header[16:20], 'big')
                sample_rate = struct.unpack(">f", header[20:24])[0]
                center_freq = struct.unpack(">f", header[24:28])[0]
            except Exception as e:
                print("WARNING: threw package {e}")
                continue

            # buffra per stream
            if stream_id not in stream_buffers:
                stream_buffers[stream_id] = {}
            stream_buffers[stream_id][pkt_no] = payload

            # här behöver vi bestämma packet_count, exempelvis första paketet = 1
            expected = max(stream_buffers[stream_id].keys()) + 1  # enklare approximation
            if len(stream_buffers[stream_id]) == expected:
                iq_bytes = b"".join(stream_buffers[stream_id][i] for i in range(expected))
                iq_arr = np.frombuffer(iq_bytes, dtype=np.float32).reshape(-1, 2)
                iq_data = iq_arr[:,0] + 1j*iq_arr[:,1]
                meta = {
                    "stream_id": stream_id,
                    "center_frequency": center_freq,
                    "sample_rate": sample_rate
                }
                # rensa bufferten
                stream_buffers.pop(stream_id)
                process_iq(iq_data, meta)


if __name__ == "__main__":
    main()
