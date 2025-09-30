#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket, json, sys, argparse
import numpy as np
from collections import deque
import struct
import time
import logging

last_update = 0.0

stored_iq = []
stored_meta = None
args = None

# --- Defaultvalues ---
DEFAULT_THRESHOLDS = {-80: " ", -72: "-", -50: "|"}

THRESHOLDS = None


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


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
        x = np.linspace(0, 1, table.shape[0])
        xnew = np.linspace(0, 1, steps)
        rgb = np.zeros((steps, 3))
        for i in range(3):
            rgb[:, i] = np.interp(xnew, x, table[:, i])
        return rgb

    if name == "custom":
        if args.custom_colormap:
            custom_meta = str(args.custom_colormap).split(",")
            custom_start, custom_stop, _ = custom_meta

            table = np.array([hex_to_rgb(custom_start), hex_to_rgb(custom_stop)])
            return interp_table(table)

        else:
            # Warn and use default color theme if not defined
            name == "viridis"
            logging.warning("WARNING: custom colormap not set!")

    if name == "viridis":
        table = np.array(
            [
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
            ]
        )
        return interp_table(table)

    elif name == "magma":
        table = np.array(
            [
                [0.001462, 0.000466, 0.013866],
                [0.063536, 0.028426, 0.180382],
                [0.144901, 0.046292, 0.258216],
                [0.313027, 0.071655, 0.383129],
                [0.532087, 0.105983, 0.524388],
                [0.784973, 0.148263, 0.639493],
                [0.993248, 0.216618, 0.556888],
            ]
        )
        return interp_table(table)

    elif name == "plasma":
        table = np.array(
            [
                [0.050383, 0.029803, 0.527975],
                [0.294833, 0.074274, 0.667240],
                [0.510168, 0.128208, 0.707174],
                [0.703417, 0.182923, 0.669366],
                [0.878953, 0.270905, 0.574048],
                [0.993248, 0.515050, 0.382914],
            ]
        )
        return interp_table(table)

    elif name == "inferno":
        table = np.array(
            [
                [0.001462, 0.000466, 0.013866],
                [0.192355, 0.063420, 0.368055],
                [0.512340, 0.130071, 0.478355],
                [0.789799, 0.278981, 0.420268],
                [0.993248, 0.550104, 0.293768],
            ]
        )
        return interp_table(table)

    else:
        logging.warning(f"Okänd colormap '{name}', använder viridis")
        return get_colormap_rgb("viridis", steps)


COLORMAP_RGB = None

# --- Konstanter ---
BUFFER_SIZE = 65535
WIDTH = 80
HEIGHT = 20

stream_buffers = {}
stream_metadata = {}
waterfall = None


def add_waterfall(power_db, freqs=None):
    """
    power_db: interpolerat till WIDTH
    freqs: frekvensvärden för dessa bins (valfritt, behövs bara om du vill göra thresholds dynamiska)
    """
    row = []
    if args.color_waterfall:
        min_val, max_val = np.min(power_db), np.max(power_db)
        norm = (power_db - min_val) / (max_val - min_val + 1e-12)
        norm = np.nan_to_num(
            norm, nan=0.0, posinf=1.0, neginf=0.0
        )  # convert NaN/inf to valid numbers
        norm = np.clip(norm, 0.0, 1.0)

        for n in norm:

            r, g, bb = COLORMAP_RGB[int(n * (len(COLORMAP_RGB) - 1))]
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
    denom = max_db - min_db + 1e-12
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
                b_left = bars[i - 1] if i > 0 else b
                b_right = bars[i + 1] if i < len(bars) - 1 else b

                # Kontroll: linjen + fyllning mellan bin
                if dist <= half or (b_left <= h <= b) or (b_right <= h <= b):
                    base_idx = int(levels[i] * (len(COLORMAP_RGB) - 1))
                    r, g, bb = COLORMAP_RGB[base_idx]

                    # Fade beräknas utifrån vertikalt avstånd till linjens topp
                    vertical_dist = max(0, b - h)  # avstånd neråt
                    fade = 1.0
                    if half > 0:
                        fade = 1.0 - 0.1 * min(dist, vertical_dist) / (half + 1e-12)
                        fade = max(0.1, fade)  # säkerställ minst lite ljus

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
                            row.append(
                                f"\x1b[48;2;{int(r*255)};{int(g*255)};{int(bb*255)}m{symbol}\x1b[0m"
                            )
                        else:
                            row.append(
                                f"\x1b[38;2;{int(r*255)};{int(g*255)};{int(bb*255)}m{symbol}\x1b[0m"
                            )
                    else:
                        row.append(symbol)
                else:
                    row.append(" ")

        rows.append("".join(row))

    # Tick-linje och etiketter med dynamisk enhet
    tick_line = "      " + "-" * WIDTH
    label_line = [" "] * (WIDTH + 6)
    num_ticks = 6
    tick_freqs = [
        f_min + i * (f_max - f_min) / (num_ticks - 1) for i in range(num_ticks)
    ]

    # Välj enhet dynamiskt
    span = f_max - f_min
    span = max(span, f_max, f_min)
    if span >= 1e6:
        unit = "MHz"
        scale = 1e6
    elif span >= 1e3:
        unit = "kHz"
        scale = 1e3
    else:
        unit = "Hz"
        scale = 1.0

    for f in tick_freqs:
        f_scaled = f / scale
        # Avrunda för läsbarhet
        if unit == "Hz":
            f_rounded = int(round(f_scaled))
        else:
            f_rounded = round(f_scaled, 2)
        label = str(f_rounded)

        pos = int((f - f_min) / (f_max - f_min + 1e-12) * (WIDTH - 1))
        start = max(0, min(6 + pos - len(label) // 2, len(label_line) - len(label)))
        for j, ch in enumerate(label):
            label_line[start + j] = ch

    rows.append(tick_line)
    rows.append("".join(label_line) + f" {unit}")
    return "\n".join(rows)


def print_waterfall():
    print(f"Waterfall (Symbols {THRESHOLDS}, max height {args.waterfall_height}):")
    for row in reversed(waterfall):
        print("      " + row)


def load_iq_from_file():
    path = args.load
    print(f"Läser IQ-data från {path} ...")

    with open(path, "rb") as f:
        while True:
            # Läs metadata (en rad JSON)
            meta_line = f.readline()
            if not meta_line:
                break  # EOF

            try:
                meta_str = meta_line.decode("utf-8", errors="ignore").strip()
                meta = json.loads(meta_str)
            except json.JSONDecodeError:
                print("DEBUG: Kunde inte tolka metadata, hoppar över block")
                continue

            num_samples = meta.get("num_samples")
            if num_samples is None:
                print("DEBUG: num_samples saknas, hoppar över block")
                continue

            # Läs IQ-data
            iq_bytes = f.read(num_samples * 2 * 4)
            if len(iq_bytes) < num_samples * 2 * 4:
                print("DEBUG: EOF innan block var komplett")
                break

            iq_arr = np.frombuffer(iq_bytes, dtype=np.float32).reshape(-1, 2)
            iq_data = iq_arr[:, 0] + 1j * iq_arr[:, 1]

            process_iq(iq_data, meta)
            time.sleep(args.refresh_rate)


def process_iq(iq_data, meta):
    global prev_interp, last_update, THRESHOLDS, stored_iq, stored_meta

    if args.refresh_rate is not None:
        now = time.time()
        min_interval = 1.0 / args.refresh_rate
        if now - last_update < min_interval:
            return  # hoppa över uppdatering
        last_update = now

    N = len(iq_data)
    if N < 2:
        return

    # Spara om --store är aktiverat
    if args.store:

        meta_copy = meta.copy()
        meta_copy["num_samples"] = len(iq_data)

        meta_json = json.dumps(meta_copy)
        store_file.write(meta_json.encode("utf-8") + b"\n")
        # Skriv block direkt till fil
        iq_arr = np.zeros((len(iq_data), 2), dtype=np.float32)
        iq_arr[:, 0] = np.real(iq_data)
        iq_arr[:, 1] = np.imag(iq_data)
        store_file.write(iq_arr.tobytes())

    # FFT och frekvensaxel
    spectrum = np.fft.fftshift(np.fft.fft(iq_data))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / meta["sample_rate"]))

    # Lägg till center frequency från metadata
    freqs += meta["center_frequency"]
    power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
    power_db -= np.max(power_db)

    # Använd manuella flaggor eller default
    f_min = args.freq_min if args.freq_min is not None else freqs[0]
    f_max = args.freq_max if args.freq_max is not None else freqs[-1]

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
    print(
        f"Stream {meta.get('stream_id','-')}  CF {meta.get('center_frequency',0)/1e6:.3f} MHz  "
        f"SR {meta.get('sample_rate',0)/1e6:.2f} Msps  Peak: {peak_freq/1e6:.3f} MHz  Mode: {mode}{extra}"
    )
    if not args.hide_spectrum:
        print("Spectrum (dB):")
        print(vertical_spectrum(interp, bins))
        print()
    if not args.hide_waterfall:
        print_waterfall()
    sys.stdout.flush()


def main():
    logging.info(f"Starting system!")
    logging.info(f"Settings: {args}")

    if args.load:
        load_iq_from_file()
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.address, args.port))

    logging.info(f"Lyssnar på UDP {args.address}:{args.port}")
    logging.info(f"Expecting format: {args.format}")

    print(f"Lyssnar på UDP {args.address}:{args.port}")

    print(f"Expecting format: {args.format}")
    try:
        sock.settimeout(3.0)
        while True:
            try:
                data, _ = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue

            if args.format == "simulator":
                # --- simulator ---
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
                    pkt_no = int.from_bytes(data[16:20], "big")
                    payload = data[20:]
                    if stream_id not in stream_buffers:
                        continue
                    stream_buffers[stream_id][pkt_no] = payload
                    expected = stream_metadata[stream_id]["packet_count"]
                    if len(stream_buffers[stream_id]) == expected:
                        iq_bytes = b"".join(
                            stream_buffers[stream_id][i] for i in range(expected)
                        )
                        iq_arr = np.frombuffer(iq_bytes, dtype=np.float32).reshape(
                            -1, 2
                        )
                        iq_data = iq_arr[:, 0] + 1j * iq_arr[:, 1]
                        meta = stream_metadata.pop(stream_id)
                        stream_buffers.pop(stream_id)

                        process_iq(iq_data, meta)

            elif args.format == "raw":
                # Check for json start
                try:
                    txt = data.decode("utf-8")
                    if txt.strip().startswith("{"):
                        meta_json = json.loads(txt)
                        stream_metadata["raw_stream"] = meta_json
                        continue  # Await next package with IQ data
                except UnicodeDecodeError:
                    pass

                iq_arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
                iq_data = iq_arr[:, 0] + 1j * iq_arr[:, 1]

                # Use latest metadat if exist
                meta = stream_metadata.get(
                    "raw_stream",
                    {
                        "stream_id": "raw_stream",
                        "center_frequency": 0.0,
                        "sample_rate": 48000.0,  # fallback
                    },
                )

                process_iq(iq_data, meta)

            elif args.format == "vita49":
                parsed = parse_vita49_packet(data)
                if parsed is None:
                    continue

                stream_id, pkt_no, sample_rate, center_freq, payload = parsed

                # payload är bara IQ-data, headern är redan separerad
                # Se till att ignorera headern om parse_vita49_packet returnerar hela paketet
                if len(payload) % 8 != 0:
                    print(
                        f"Payload size {len(payload)} not divisible by 8, trimming extra bytes"
                    )
                    payload = payload[: len(payload) // 8 * 8]  # klipp bort extra byte

                # Tolka IQ-data som float32 little-endian interleaved
                iq_arr = np.frombuffer(payload, dtype="<f4").reshape(-1, 2)
                iq_data = iq_arr[:, 0] + 1j * iq_arr[:, 1]

                meta = {
                    "stream_id": stream_id.hex(),
                    "center_frequency": center_freq,
                    "sample_rate": sample_rate,
                }

                process_iq(iq_data, meta)
    except KeyboardInterrupt:
        print("\nAvslutar mottagare...")
    finally:
        sock.close()
        logging.info(f"Closing system!")
        if args.store:
            store_file.close()
            print(f"IQ-data sparad till {args.store}")


def parse_vita49_packet(data: bytes):
    if len(data) < 32:
        return None
    header = data[:32]
    payload = data[32:]

    hdr_word0 = struct.unpack("<I", header[:4])[0]
    stream_id = header[4:20]
    pkt_no = struct.unpack("<I", header[20:24])[0]
    sample_rate = struct.unpack("<f", header[24:28])[0]
    center_freq = struct.unpack("<f", header[28:32])[0]

    return stream_id, pkt_no, sample_rate, center_freq, payload


if __name__ == "__main__":

    # --- Argument parser ---
    parser = argparse.ArgumentParser(
        description="Terminal-based spectrum and waterfall display"
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        help="Thresholds and symbols, e.g., '-50:|,-72:-,-77:.'",
    )
    parser.add_argument(
        "--waterfall-height",
        type=int,
        default=10,
        help="Maximum number of rows in the waterfall",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of frequency points for display (width)",
    )
    parser.add_argument(
        "--color-waterfall",
        action="store_true",
        help="Display waterfall in color instead of symbols",
    )
    parser.add_argument(
        "--color-spectrum", action="store_true", help="Display spectrum in color"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Colormap for color: viridis, magma, plasma, inferno",
    )
    parser.add_argument(
        "--spectrum-symbol",
        type=str,
        default=".",
        help="Symbol used for colored spectrum, default '.'",
    )
    parser.add_argument(
        "--spectrum-symbol-color-background",
        action="store_true",
        help="Display spectrum symbol background color",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=None,
        help="Minimum frequency in spectrum to display (Hz)",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=None,
        help="Maximum frequency in spectrum to display (Hz)",
    )

    parser.add_argument(
        "--store",
        type=str,
        nargs="?",
        const="./output.iq",
        help="Spara inkommande IQ-data till fil (default ./output.iq)",
    )
    parser.add_argument(
        "--load",
        type=str,
        nargs="?",
        const="./output.iq",
        help="Läs IQ-data från fil istället för UDP (default ./output.iq)",
    )

    parser.add_argument(
        "--auto-zoom",
        action="store_true",
        help="Automatically calculate noise floor and zoom to the area where signals exist",
    )
    parser.add_argument(
        "--auto-zoom-threshold",
        type=float,
        default=10.0,
        help="How many dB above noise floor counts as signal (default 10 dB)",
    )

    parser.add_argument("--log", action="store_true", help="Enable logging to file")

    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=None,
        help="Maximum refresh rate in Hz (default None = as fast as possible).",
    )

    parser.add_argument(
        "--max-delta-db",
        type=float,
        default=None,
        help="Maximum allowed jump in dB per refresh (None = no limit) (WARNING: this requires more memory!)",
    )

    parser.add_argument(
        "--address",
        type=str,
        default="127.0.0.1",
        help="IP address or hostname of the radio device (e.g., 192.168.1.50)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="TCP/UDP port for connecting to the radio device",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["vita49", "raw", "simulator"],
        default="vita49",
        help="Stream format/protocol, e.g., 'vita49' (VITA-49), 'raw' (raw IQ samples), 'simulator' (simulated data)",
    )

    parser.add_argument(
        "--custom-colormap",
        type=str,
        help="Custom colormap: startcolor,stopcolor,steps. Example: '#0000FF,#FF0000,64'",
    )

    parser.add_argument(
        "--spectrum-height",
        type=int,
        default=10,
        help="Height in rows of the spectrum display",
    )

    # --- New arguments: either bar or line mode, and line-width (vertical) ---
    parser.add_argument(
        "--bar", action="store_true", help="Display spectrum in bar mode (default)."
    )
    parser.add_argument(
        "--line",
        action="store_true",
        help="Display spectrum in line mode (contour line).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=1,
        help="Vertical thickness (number of rows) of the line in line mode. 1 = one row.",
    )

    parser.add_argument(
        "--hide-spectrum",
        action="store_true",
        help="Do not display the spectrum output",
    )

    parser.add_argument(
        "--hide-waterfall",
        action="store_true",
        help="Do not display the waterfall output",
    )
    prev_interp = None

    args = parser.parse_args()
    WIDTH = args.bins
    COLORMAP_RGB = get_colormap_rgb(args.colormap)

    if args.log:
        logging.basicConfig(
            filename="radio_log.txt",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info("Logging started")

    if args.freq_min is not None and args.freq_max is not None:
        if args.freq_min >= args.freq_max:
            logging.warning(
                f"WARNING: --freq-min ({args.freq_min}) has to be less then --freq-max ({args.freq_max})"
            )
            sys.exit(1)

    if args.thresholds:
        THRESHOLDS = {}
        try:
            pairs = args.thresholds.split(",")
            for p in pairs:
                key, sym = p.split(":")
                key = key.replace("\\", "")
                THRESHOLDS[int(key)] = sym
        except Exception as e:
            logging.warning(f"Fel i thresholds-argument, använder default: {e}")
            THRESHOLDS = DEFAULT_THRESHOLDS
    else:
        THRESHOLDS = DEFAULT_THRESHOLDS
    if args.store:
        store_file = open(args.store, "ab")
        # Skriv metadata först som JSON + newline

    waterfall = deque(maxlen=args.waterfall_height)

    # Default: bar-mode
    if not args.bar and not args.line:
        args.bar = True

    main()
