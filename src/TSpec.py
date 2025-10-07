#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket, json, sys, argparse
import numpy as np
from collections import deque
import struct
import time
import logging
import random

import importlib.util

protocol_handler = None

parse_module = None



last_update = 0.0

stored_iq = []
stored_meta = None
args = None

# --- Defaultvalues ---
DEFAULT_THRESHOLDS = {-80: " ", -72: "-", -50: "|"}
maxhold_spectrum = None
THRESHOLDS = None

autozoom_count = 0

soupy_available = False
try:
    import SoapySDR
    from SoapySDR import *  # noqa
    soupy_available = True
except ImportError:
    print("⚠️ SoapySDR not installed. SoupySDR input disabled.")

sock = None
iqfile = None
sdr = None

def setup_input():
    global sock, protocol_handler, iqfile, sdr
    
    if args.input.endswith(".py"):
        
        module_path = os.path.abspath(args.input)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print(f"Loaded module {args.input}")

        if hasattr(module, "get_data"):
            protocol_handler = module
            if hasattr(module, "setup"):
                module.setup()
            print(f"✅ Loaded protocol module: {module_name}")
        else:
            raise AttributeError(f"⚠️ {module_name} does not define get_data()")

    # --- 2️⃣ UDP-ingång ---
    elif args.input.lower() == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((args.address, args.port))
        sock.settimeout(3.0)
        print(f"✅ Listening on UDP {args.address}:{args.port}")

    # --- 3️⃣ IQ-fil ---
    elif args.input.lower() == "iqfile":
        if not args.filepath:
            raise ValueError("❌ Missing --filepath argument for iqfile input")
        iqfile = open(args.filepath, "rb")
        print(f"✅ Reading IQ data from file: {args.filepath}")

    # --- 4️⃣ SoapySDR (placeholder) ---
    elif args.input.lower() == "soapysdr":
        try:
            if not soupy_available:
                print("❌ SoapySDR input requested but SoapySDR is not installed.")
                sys.exit(1)
            args.driver = getattr(args, "driver", f"driver={args.driver}")
            sdr = SoapySDR.Device(args.driver)
            sdr.setSampleRate(SOAPY_SDR_RX, 0, args.sample_rate)
            sdr.setFrequency(SOAPY_SDR_RX, 0, args.center_frequency)
            rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            sdr.activateStream(rxStream)
            print("✅ SoapySDR device active")
        except ImportError:
            raise ImportError("❌ SoapySDR not installed, run: pip install SoapySDR")


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


waterfall_counter = 0  # global


def add_waterfall(power_db, freqs=None):
    global waterfall_counter, COLORMAP_RGB
    waterfall_counter += 1
    if waterfall_counter < args.waterfall_speed:
        return  # hoppa över den här uppdateringen
    waterfall_counter = 0  # återställ

    row = []
    if args.color_waterfall:
        min_val, max_val = np.min(power_db), np.max(power_db)
        norm = (power_db - min_val) / (max_val - min_val + 1e-12)
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        norm = np.clip(norm, 0.0, 1.0)

        # Apply gamma correction
        gamma = args.waterfall_gamma if args.waterfall_gamma is not None else 1.0
        norm = norm**gamma

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


def vertical_spectrum(power_db, freqs, f_min=None, f_max=None, feature_flags=None):
    global COLORMAP_RGB
    HEIGHT = args.spectrum_height  # använd argumentet istället för global HEIGHT

    if f_min is None:
        f_min = freqs[0]
    if f_max is None:
        f_max = freqs[-1]

    min_db = args.db_min if args.db_min is not None else np.min(power_db)
    max_db = args.db_max if args.db_max is not None else np.max(power_db)
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

                if dist <= half or (b_left <= h <= b) or (b_right <= h <= b):
                    base_idx = int(levels[i] * (len(COLORMAP_RGB) - 1))
                    r, g, bb = COLORMAP_RGB[base_idx]

                    vertical_dist = max(0, b - h)
                    fade = 1.0
                    if half > 0:
                        fade = 1.0 - 0.1 * min(dist, vertical_dist) / (half + 1e-12)
                        fade = max(0.1, fade)

                    r_f = int(r * 255 * fade + 255 * (1 - fade))
                    g_f = int(g * 255 * fade + 255 * (1 - fade))
                    b_f = int(bb * 255 * fade + 255 * (1 - fade))

                    draw_symbol = symbol
                    # Lägg till feature-symbol på toppen av stapeln
                    if args.feature_symbol is not None:
                        if feature_flags is not None and feature_flags[i] and h == b:
                            draw_symbol = args.feature_symbol
                            if getattr(args, "feature_color", None):
                                try:
                                    r_fc, g_fc, b_fc = [
                                        int(x) for x in args.feature_color.split(",")
                                    ]
                                except:
                                    r_fc, g_fc, b_fc = 255, 255, 255
                                # Skriv över alla andra färger
                                if args.spectrum_symbol_color_background:
                                    draw_symbol = f"\x1b[48;2;{r_fc};{g_fc};{b_fc}m{draw_symbol}\x1b[0m"
                                else:
                                    draw_symbol = f"\x1b[38;2;{r_fc};{g_fc};{b_fc}m{draw_symbol}\x1b[0m"

                    if args.color_spectrum:
                        if args.spectrum_symbol_color_background:
                            row.append(
                                f"\x1b[48;2;{r_f};{g_f};{b_f}m{draw_symbol}\x1b[0m"
                            )
                        else:
                            row.append(
                                f"\x1b[38;2;{r_f};{g_f};{b_f}m{draw_symbol}\x1b[0m"
                            )
                    else:
                        row.append(draw_symbol)
                else:
                    row.append(" ")

        else:
            for i, b in enumerate(bars):
                draw_symbol = symbol
                if b >= h:
                    if args.color_spectrum:
                        idx = b * len(COLORMAP_RGB) // HEIGHT
                        r, g, bb = COLORMAP_RGB[idx]
                        if args.spectrum_symbol_color_background:
                            draw_symbol = f"\x1b[48;2;{int(r*255)};{int(g*255)};{int(bb*255)}m{draw_symbol}\x1b[0m"
                        else:
                            draw_symbol = f"\x1b[38;2;{int(r*255)};{int(g*255)};{int(bb*255)}m{draw_symbol}\x1b[0m"

                # Lägg till feature-symbol på toppen av stapeln
                if args.feature_symbol is not None:
                    if feature_flags is not None and feature_flags[i] and h == b:
                        draw_symbol = args.feature_symbol
                        if getattr(args, "feature_color", None):
                            try:
                                r_fc, g_fc, b_fc = [
                                    int(x) for x in args.feature_color.split(",")
                                ]
                            except:
                                r_fc, g_fc, b_fc = 255, 255, 255
                            # Skriv över alla andra färger
                            if args.spectrum_symbol_color_background:
                                draw_symbol = f"\x1b[48;2;{r_fc};{g_fc};{b_fc}m{draw_symbol}\x1b[0m"
                            else:
                                draw_symbol = f"\x1b[38;2;{r_fc};{g_fc};{b_fc}m{draw_symbol}\x1b[0m"

                row.append(
                    draw_symbol
                    if b >= h
                    or (feature_flags is not None and feature_flags[i] and h == b)
                    else " "
                )

        rows.append("".join(row))

    # Tick-linje och etiketter med dynamisk enhet
    tick_line = "      " + "-" * WIDTH
    label_line = [" "] * (WIDTH + 6)
    num_ticks = 6
    tick_freqs = [
        f_min + i * (f_max - f_min) / (num_ticks - 1) for i in range(num_ticks)
    ]

    span = f_max - f_min
    span = max(span, f_max, f_min)
    if span >= 1e9:  # > 1 GHz
        scale = 1e9
        unit = "GHz"
    elif span >= 1e6:
        unit = "MHz"
        scale = 1e6
    elif span >= 1e3:
        unit = "kHz"
        scale = 1e3
    elif span >= 10:
        unit = "Hz"
        scale = 1.0
    elif span >= 1e-3:
        scale = 1e-3
        unit = "mHz"
    else:
        scale = 1e-6
        unit = "µHz"
        
    # Kontrollera om tick-värdena blir > 999 (max 3 siffror)
    max_val = max(abs(f_min), abs(f_max)) / scale
    if max_val > 999:
        factor = 10 ** (int(np.log10(max_val)) - 2)  # t.ex. 10^3 -> 10^(3-2)=10
        scale *= factor
        unit = f"{unit}*{factor:.0e}"  # t.ex. "MHz*1e3"

    # Tick-loop fungerar precis som tidigare
    for f in tick_freqs:
        f_scaled = f / scale
        if unit.startswith("Hz") and "*" not in unit:
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

    start_sample = args.start_sample if args.start_sample else 0
    duration_sec = args.duration if args.duration is not None else None

    with open(path, "rb") as f:
        total_samples_read = 0
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
                if args.ignore_missing_meta:
                    continue  # hoppa över blocket tyst
                else:
                    print("DEBUG: num_samples saknas, hoppar över block")
                    continue

            # Hoppa över samples före start-sample
            if total_samples_read + num_samples <= start_sample:
                # Skippa hela blocket
                f.seek(num_samples * 2 * 4, 1)
                total_samples_read += num_samples
                continue

            # Om start ligger mitt i blocket, justera
            block_start_idx = max(0, start_sample - total_samples_read)
            block_end_idx = num_samples  # standard: hela blocket
            if duration_sec is not None:
                # Beräkna max antal samples som får läsas
                max_samples = int(duration_sec * meta["sample_rate"])
                if block_start_idx + max_samples < block_end_idx:
                    block_end_idx = block_start_idx + max_samples

            samples_to_read = block_end_idx - block_start_idx
            if samples_to_read <= 0:
                break

            # Läs hela blocket först
            iq_bytes = f.read(num_samples * 2 * 4)
            if len(iq_bytes) < num_samples * 2 * 4:
                print("DEBUG: EOF innan block var komplett")
                break

            dtype_map = {
                "float32": np.dtype("float32"),
                "int16": np.dtype("int16"),
                "int8": np.dtype("int8"),
            }
            dtype = dtype_map[args.dtype]

            # Apply byte order
            if args.byteorder == "big":
                dtype = dtype.newbyteorder(">")
            else:
                dtype = dtype.newbyteorder("<")

            iq_arr = np.frombuffer(iq_bytes, dtype=dtype).reshape(-1, 2)

            # Om input är int8/int16, normalisera till -1.0..1.0
            if dtype in [np.int16, np.int8]:
                max_val = float(np.iinfo(dtype).max)
                iq_arr = iq_arr.astype(np.float32) / max_val

            iq_data = (
                iq_arr[block_start_idx:block_end_idx, 0]
                + 1j * iq_arr[block_start_idx:block_end_idx, 1]
            )

            process_iq(iq_data, meta)
            total_samples_read += num_samples

            # Om duration uppnådd, avbryt
            if (
                duration_sec is not None
                and total_samples_read - start_sample >= max_samples
            ):
                break

            if args.refresh_rate:
                time.sleep(args.refresh_rate)


def process_iq(iq_data, meta):
    global prev_interp, last_update, THRESHOLDS, stored_iq, stored_meta, autozoom_count

    if args.refresh_rate is not None:
        now = time.time()
        min_interval = 1.0 / args.refresh_rate
        if now - last_update < min_interval:
            return
        last_update = now

    N = len(iq_data)
    if N < 2:
        return

    # Bestäm FFT- och blockstorlek samt overlap (säkra värden)
    fft_size = int(args.fft_size) if args.fft_size else N
    block_size = int(args.block_size) if getattr(args, "block_size", None) else fft_size
    overlap = (
        float(args.fft_overlap)
        if getattr(args, "fft_overlap", 0.0) is not None
        else 0.0
    )
    overlap = min(max(overlap, 0.0), 0.99)  # clamp 0..0.99
    hop_size = max(1, int(block_size * (1.0 - overlap)))

    # Spara om --store är aktiverat (oförändrat)
    if args.store:
        meta_copy = meta.copy()
        meta_copy["num_samples"] = len(iq_data)
        meta_json = json.dumps(meta_copy)
        store_file.write(meta_json.encode("utf-8") + b"\n")
        iq_arr = np.zeros((len(iq_data), 2), dtype=np.float32)
        iq_arr[:, 0] = np.real(iq_data)
        iq_arr[:, 1] = np.imag(iq_data)
        store_file.write(iq_arr.tobytes())

    # Decimera om begärt
    if args.decimate > 1:
        iq_data = iq_data[:: args.decimate]
        meta["sample_rate"] /= args.decimate

    # Om input är kortare än block_size: padda med nollor så vi kör åtminstone en FFT
    if len(iq_data) < block_size:
        pad_len = block_size - len(iq_data)
        iq_data = np.pad(iq_data, (0, pad_len), "constant")

    # Loop över block med overlap
    for start in range(0, len(iq_data) - block_size + 1, hop_size):
        segment = iq_data[start : start + block_size]

        # Fönsterfunktion (behåller din logik)
        win_len = min(len(segment), fft_size)
        if args.window == "hann":
            win = np.hanning(win_len)
        elif args.window == "hamming":
            win = np.hamming(win_len)
        elif args.window == "blackman":
            win = np.blackman(win_len)
        else:
            win = np.ones(win_len)

        # Förbered block för FFT (trim/padda till fft_size om nödvändigt)
        block_for_fft = segment[:fft_size].copy()
        if len(block_for_fft) < win_len:
            # om fft_size < win_len (ovanligt) - trim; annars pad
            block_for_fft = np.pad(
                block_for_fft, (0, win_len - len(block_for_fft)), "constant"
            )
        block_for_fft[:win_len] *= win

        # FFT + frekvensaxel
        spectrum = np.fft.fftshift(np.fft.fft(block_for_fft, n=fft_size))
        freq_offset = getattr(args, "freq_offset", 0.0)
        freqs = (
            np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / meta["sample_rate"]))
            + meta.get("center_frequency", 0.0)
            + freq_offset
        )

        handle_key_press(freqs)

        # Power (lin/log) + normalisering
        if args.waterfall_scale == "linear":
            power_db = np.abs(spectrum)
        else:
            power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

        if not args.no_normalize:
            power_db = power_db - np.max(power_db)

        if args.db_min is not None:
            power_db = np.maximum(power_db, args.db_min)  # allt under db-min klipps
        if args.db_max is not None:
            power_db = np.minimum(power_db, args.db_max)  # allt över db-max klipps

        if args.agc:
            if not hasattr(process_iq, "agc_level"):
                process_iq.agc_level = np.max(power_db)

            # Exponential smoothing of gain level
            alpha = args.agc_speed
            process_iq.agc_level = (1 - alpha) * process_iq.agc_level + alpha * np.max(
                power_db
            )

            # Normalize power_db relative to AGC level
            power_db -= process_iq.agc_level

        # 4️⃣ Clip to db-min / db-max
        if args.db_min is not None:
            power_db = np.maximum(power_db, args.db_min)
        if args.db_max is not None:
            power_db = np.minimum(power_db, args.db_max)
        # f_min / f_max (behöver freqs här, därför inuti loopen)
        f_min = args.freq_min if args.freq_min is not None else freqs[0]
        f_max = args.freq_max if args.freq_max is not None else freqs[-1]

        # Autozoom (behållen logik, räknas per analyserat block)
        if args.auto_zoom and (
            args.auto_zoom_iterations == 0 or autozoom_count < args.auto_zoom_iterations
        ):
            autozoom_count += 1
            noise_floor = np.median(power_db)
            signal_mask = power_db > (noise_floor + args.auto_zoom_threshold)
            if np.any(signal_mask):
                sig_freqs = freqs[signal_mask]
                auto_min = sig_freqs.min()
                auto_max = sig_freqs.max()
                span = auto_max - auto_min
                args.freq_min = auto_min - 0.05 * span
                args.freq_max = auto_max + 0.05 * span
            else:
                args.freq_min, args.freq_max = freqs[0], freqs[-1]

            if THRESHOLDS == DEFAULT_THRESHOLDS:
                symbols = list(DEFAULT_THRESHOLDS.values())
                n = len(symbols)
                signal_mask = (freqs >= f_min) & (freqs <= f_max)
                power_interval = power_db[signal_mask]
                if len(power_interval) > 0:
                    min_val = np.min(power_interval)
                    max_val = np.max(power_interval)
                    step = (max_val - min_val) / n
                    new_thresholds = {}
                    for i, sym in enumerate(symbols):
                        thresh = min_val + i * step
                        new_thresholds[int(round(thresh))] = sym
                    THRESHOLDS = new_thresholds

        # Peak och zoomad vy
        peak_idx = np.argmax(power_db)
        peak_freq = freqs[peak_idx]
        mask = (freqs >= f_min) & (freqs <= f_max)
        freqs_zoom = freqs[mask]
        power_zoom = power_db[mask]
        if freqs_zoom.size == 0:
            freqs_zoom = freqs
            power_zoom = power_db

        if args.freq_min is None:
            args.freq_min = freqs[0]
        if args.freq_max is None:
            args.freq_max = freqs[-1]

        # Skapa bins för display baserat på det nuvarande fönstret
        bins = np.linspace(args.freq_min, args.freq_max, WIDTH)

        # Interpolera spektrum till bins, fyll utanför med lågt värde (db_min)
        interp = np.interp(
            bins,
            freqs,
            power_db,
            left=args.db_min if args.db_min is not None else -100,
            right=args.db_min if args.db_min is not None else -100,
        )

        # --- Avg-blocks ---
        if args.avg_blocks > 1:
            if not hasattr(process_iq, "block_history"):
                from collections import deque

                process_iq.block_history = deque(maxlen=args.avg_blocks)
            process_iq.block_history.append(interp.copy())
            interp = np.mean(np.array(process_iq.block_history), axis=0)

        # --- Efter avg_blocks-logiken ---
        if not hasattr(process_iq, "interp_prev"):
            process_iq.interp_prev = interp.copy()

        # delta fallprocent per tick
        slew = args.peak_preserve  # t.ex. 0.1 = 10% per tick

        # skillnad mellan föregående och nuvarande
        diff = interp - process_iq.interp_prev

        # om signalen går upp → visa direkt
        output_interp = np.where(diff > 0, interp, process_iq.interp_prev)

        # om signalen går ner → minska med procent
        falling = diff < 0
        output_interp[falling] = process_iq.interp_prev[falling] + diff[falling] * slew

        # spara som tidigare
        process_iq.interp_prev = output_interp.copy()
        interp = output_interp

        # Clamp / max-delta
        if args.max_delta_db is not None:
            if prev_interp is None:
                prev_interp = interp.copy()
            else:
                interp = clamp_delta(interp, prev_interp, args.max_delta_db)
                prev_interp = interp.copy()

        # Smoothing (EMA) — bevarad
        if args.smoothing > 0:
            if (
                not hasattr(process_iq, "smoothed_interp")
                or process_iq.smoothed_interp.shape != interp.shape
            ):
                process_iq.smoothed_interp = interp.copy()
            else:
                alpha = float(args.smoothing)
                process_iq.smoothed_interp = (
                    1.0 - alpha
                ) * process_iq.smoothed_interp + alpha * interp
            interp = process_iq.smoothed_interp.copy()

        if args.maxhold:
            global maxhold_spectrum
            if maxhold_spectrum is None or maxhold_spectrum.shape != interp.shape:
                maxhold_spectrum = interp.copy()
            else:
                maxhold_spectrum = np.maximum(maxhold_spectrum, interp)
            interp = maxhold_spectrum

        # Visa peak i interpolationen# Hitta peak inom det zoomade fönstret
        peak_idx = np.argmax(power_zoom)
        peak_freq = freqs_zoom[peak_idx]
        peak_bin = int(
            (peak_freq - args.freq_min) / (args.freq_max - args.freq_min) * (WIDTH - 1)
        )

        if 0 <= peak_bin < WIDTH:
            interp[peak_bin] = max(interp[peak_bin], power_zoom[peak_idx])

        feature_flags = np.zeros_like(interp, dtype=bool)

        # Medelvärde + tröskel
        mean_val = np.mean(interp)
        threshold = mean_val + args.feature_avg_offset

        # Markera bins som peakar över tröskeln
        for i in range(1, len(interp) - 1):
            if (
                interp[i] > interp[i - 1]
                and interp[i] > interp[i + 1]
                and interp[i] > threshold
            ):
                feature_flags[i] = True

        # Waterfall + utskrift (oförändrat)
        add_waterfall(interp, bins)
        if args.clear_on_new_frame:
            sys.stdout.write("\x1b[2J\x1b[H")
        else:
            sys.stdout.write("\x1b[H")

        mode = "Line" if args.line else "Bar"
        extra = f"  LineWidth: {args.line_width}" if args.line else ""

        print(
            f"Stream {meta.get('stream_id','-')}  CF {meta.get('center_frequency',0)/1e6:.3f} MHz  "
            f"SR {meta.get('sample_rate',0)/1e6:.2f} Msps  Peak: {peak_freq/1e6:.3f} MHz  Mode: {mode}{extra}"
        )
        if args.timestamp:
            print(time.strftime("[%Y-%m-%d %H:%M:%S]"), end=" ")

        if args.rssi:
            # Konvertera från dB till linjär skala för medelvärde
            linear_power = 10 ** (interp / 10.0)
            rssi_val = 10 * np.log10(np.mean(linear_power) + 1e-12)
            print(f"RSSI: {rssi_val:.2f} dB ")
        if not args.hide_spectrum:
            print("Spectrum (dB):")
            print(vertical_spectrum(interp, bins, feature_flags=feature_flags))

        if not args.hide_waterfall:
            print_waterfall()
        sys.stdout.flush()


def main():
    logging.info(f"Starting system!")
    logging.info(f"Settings: {args}")

    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()

    if args.load:
        load_iq_from_file()
        return

    logging.info(f"Format {args.format}")
    logging.info(f"Parser {args.input}")
    
    print(f"Format {args.format}")
    print(f"Parser {args.input}")
    
    print(f"Default: {args.address}:{args.port}")

    print(f"Expecting format: {args.format}")
    
    setup_input()

    try:
        while True:
            try:
                if protocol_handler:
                    # Custom protocol provides its own data-fetching logic
                    data = protocol_handler.get_data()
                    
                else:
                    data, _ = sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Protocol error: {e}")
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
                        if args.ignore_missing_meta:
                            continue
                        else:
                            print(f"DEBUG: saknas metadata för stream {stream_id}")
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

            elif args.format == "parse_module":
                if not parse_module:
                    print(
                        "⚠️ No parse module loaded. Use --parse-module ./path/to/parser.py"
                    )
                    continue

                try:
                    result = parse_module.parse_data(data)
                    if not result:
                        continue

                    iq_data, meta = result
                    if iq_data is None:
                        continue

                    process_iq(iq_data, meta or {})
                except Exception as e:
                    print(f"Parse module error: {e}")
                    continue

    except KeyboardInterrupt:
        print("\nAvslutar mottagare...")
    finally:
        sock.close()
        logging.info(f"Closing system!")
        if args.store:
            store_file.close()
            print(f"IQ-data sparad till {args.store}")


import sys, os, select


def get_key():
    """Returnerar en tangent om en är nedtryckt, annars None (ingen blockering)."""
    if os.name == "nt":
        import msvcrt

        key = None
        while msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8", errors="ignore")
        return key
    else:  # Linux / macOS
        import tty, termios

        key = None
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        while dr:
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                key = sys.stdin.read(1)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            dr, _, _ = select.select([sys.stdin], [], [], 0)
        return key


def handle_key_press(freqs):
    global maxhold_spectrum, prev_interp, autozoom_count, COLORMAP_RGB
    key = get_key()

    if key:
        # Initiera frekvensfönster om None
        if args.freq_min is None or args.freq_max is None:
            args.freq_min = freqs[0]
            args.freq_max = freqs[-1]
        
        span = args.freq_max - args.freq_min
        mid = (args.freq_max + args.freq_min) / 2

        if key == "a":  # vänster
            shift = span * 0.1
            args.freq_min -= shift
            args.freq_max -= shift
        elif key == "d":  # höger
            shift = span * 0.1
            args.freq_min += shift
            args.freq_max += shift
        elif key == "w":  # höj dB-max
            args.db_max = args.db_max + 5
            args.db_min = args.db_min + 5
        elif key == "s":  # sänk dB-min
            args.db_max = args.db_max - 5
            args.db_min = args.db_min - 5
        elif key == "+":  # zooma in
            span *= 0.9
            args.freq_min = mid - span / 2
            args.freq_max = mid + span / 2
        elif key == "-":  # zooma ut
            span *= 1.1
            args.freq_min = mid - span / 2
            args.freq_max = mid + span / 2
        elif key == "f":  # autozoom en gång
            args.auto_zoom = True
            args.auto_zoom_iterations += 1

        elif key == "x":  # random_color
            if args.color_spectrum or args.color_waterfall:
                args.colormap = "custom"
                args.custom_colormap = f"{random_hex_color()},{random_hex_color()},64"

                COLORMAP_RGB = get_colormap_rgb(args.colormap)

        elif key == "c":  # clear
            # clear old
            sys.stdout.write("\x1b[2J\x1b[H")

        elif key == "i":  # increase line width
            args.line_width += 1
        elif key == "o":  # decrease line width
            args.line_width -= 1

        elif key == "r":  # reset
            args.freq_min = None
            args.freq_max = None
            args.db_min = -80
            args.db_max = 0
            maxhold_spectrum = None
            prev_interp = None

        elif key in "0123456789":
            if key == "0":
                args.refresh_rate = None
                print("\nRefresh rate: unlimited")
            else:
                args.refresh_rate = int(key)
                print(f"\nRefresh rate: {args.refresh_rate} fps")


def random_hex_color():
    return "#{:06X}".format(random.randint(0, 0xFFFFFF))


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


COLORMAP_RGB = None
if __name__ == "__main__":
    # --- Argument parser ---
    parser = argparse.ArgumentParser(
        description="Terminal-based spectrum and waterfall display"
    )
    parser.add_argument(
        "--rssi",
        action="store_true",
        help="Display received signal strength (RSSI) in dB instead of full spectrum",
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
        help="UDP port for connecting to the radio device",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["vita49", "raw", "simulator", "parsemodule"],
        default="vita49",
        help="Stream format/protocol, e.g., 'parsemodule' filepath to parsing module, 'vita49' (VITA-49), 'raw' (raw IQ samples), 'simulator' (simulated data)",
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

    # --- FFT & signalbehandling ---

    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        choices=["hann", "hamming", "blackman", "rectangular"],
        help="FFT window function",
    )

    parser.add_argument(
        "--freq-offset",
        type=int,
        default=0,
        help="Frequency Offset",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize spectrum to 0 dB max",
    )

    # --- Waterfall ---
    parser.add_argument(
        "--waterfall-scale",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="Scale type for waterfall (log or linear)",
    )

    # --- Smoothing ---
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Apply exponential moving average smoothing (0.0 = off, e.g., 0.2 = 20%)",
    )

    parser.add_argument(
        "--peak-preserve",
        type=float,
        default=0.1,
        help="Behåll toppar (max) när signalen ökar, slewdown value float.",
    )

    # --- Tidsstämpling ---
    parser.add_argument(
        "--timestamp", action="store_true", help="Print timestamp with each frame"
    )

    parser.add_argument(
        "--agc",
        action="store_true",
        help="Enable automatic gain control to normalize spectrum display",
    )

    parser.add_argument(
        "--agc-speed",
        type=float,
        default=0.1,
        help="AGC speed / smoothing factor (0.0 = slow, 1.0 = instant). Default 0.1",
    )

    # --- Decimation ---
    parser.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="Use only every Nth sample to reduce sample rate (default=1 = no decimation)",
    )

    parser.add_argument(
        "--ignore-missing-meta",
        action="store_true",
        help="Ignore IQ blocks that are missing metadata instead of raising errors",
    )

    parser.add_argument(
        "--fft-size",
        type=int,
        default=None,
        help="FFT size (default = length of input block)",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="udp",
        help="Input source: 'udp', 'iqfile', 'soupySDR' or path to a Python protocol module.",
    )
    parser.add_argument("--file", type=str, help="Path to IQ file if using 'iqfile'.")

    parser.add_argument(
        "--fft-overlap",
        type=float,
        default=0.0,
        help="Fractional overlap between FFT blocks (0.0 = no overlap, 0.5 = 50%)",
    )
    # --- Filkontroll ---
    parser.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="Start reading file at this sample index (default=0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="How many seconds of data to process from file (default=None = until EOF)",
    )

    parser.add_argument(
        "--waterfall-speed",
        type=int,
        default=1,
        help="Number of waterfall updates to skip per frame (higher = faster scroll)",
    )

    # --- Autozoom ---
    parser.add_argument(
        "--auto-zoom-iterations",
        type=int,
        default=0,
        help="Number of iterations to perform autozooming (0 = infinite)",
    )

    parser.add_argument(
        "--clear-on-new-frame",
        type=bool,
        default=False,
        help="Determine if to clear terminal each frame. This might cause flimmer.",
    )

    parser.add_argument(
        "--waterfall-gamma",
        type=float,
        default=1.0,
        help="Gamma correction for waterfall colors (default=1.0, linear)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "int16", "int8"],
        default="float32",
        help="Data type of incoming IQ samples (float32, int16, int8). Default = float32",
    )

    parser.add_argument(
        "--byteorder",
        type=str,
        choices=["little", "big"],
        default="little",
        help="Byte order of incoming IQ samples (little or big endian). Default = little",
    )

    parser.add_argument(
        "--avg-blocks",
        type=int,
        default=1,
        help="Average over N blocks (smoother display, default=1 = off)",
    )

    parser.add_argument(
        "--maxhold",
        action="store_true",
        help="Show maximum value per bin across frames instead of current spectrum",
    )

    parser.add_argument(
        "--feature-symbol",
        type=str,
        default=None,
        help="Symbol used to mark extracted features (e.g., peak) in the spectrum",
    )

    parser.add_argument(
        "--animate-symbols",
        action="store_true",
        help="Aktivera symbolanimation (övergång mellan symboler)",
    )
    parser.add_argument(
        "--animate-length",
        type=int,
        default=5,
        help="Antal steg för symbolövergång innan slutlig symbol",
    )


    parser.add_argument(
        "--feature-color",
        type=str,
        default=None,
        help="Color for extracted feature (format: 'R,G,B' 0-255), overrides symbol color if set",
    )

    parser.add_argument(
        "--feature-avg-offset",
        type=int,
        default=5,
        help="Offset for feature extraction detection.",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Block size (number of samples per FFT). Default = use fft-size or input length",
    )

    parser.add_argument(
        "--parse-module",
        type=str,
        help="Path to a custom Python parser module implementing parse_data(data) -> (iq_data, meta)",
    )

    parser.add_argument(
        "--driver", type=str, default="rtlsdr", help="Name of soupySDR driver to use."
    )
    

    parser.add_argument(
        "--db-min", type=float, default=-80, help="Minimum dB level for display"
    )
    parser.add_argument(
        "--db-max", type=float, default=0, help="Maximum dB level for display"
    )

    prev_interp = None

    args = parser.parse_args()
    WIDTH = args.bins
    COLORMAP_RGB = get_colormap_rgb(args.colormap)


    if args.parse_module:
        module_path = os.path.abspath(args.parse_module)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "parse_data"):
            parse_module = module
            print(f"✅ Loaded parse module: {module_name}")
        else:
            print(
                f"⚠️ Parse module {module_name} missing required function parse_data(data)"
            )


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
