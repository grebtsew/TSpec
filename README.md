# TSpec
TSpec is a minimalist, high-performance terminal-based spectrum and waterfall display designed to run on any system, even without a GUI. It works seamlessly on older terminals and low-powered hardware, making it ideal for embedded systems or remote servers. With support for multiple stream formats and radio systems, TSpec ensures maximum flexibility while keeping resource usage minimal. TSpec also provides a wide range of features—see [features](#flags) for more details. Whether you need a quick visualization on a headless server or a lightweight monitoring tool, TSpec delivers fast, reliable, and portable spectrum analysis wherever you need it.



![demogif](./docs/demo.gif)

![CI](https://github.com/grebtsew/TSpec/actions/workflows/ci.yml/badge.svg)
![license](https://img.shields.io/github/license/grebtsew/TSpec)
![size](https://img.shields.io/github/repo-size/grebtsew/TSpec)
![commit](https://img.shields.io/github/last-commit/grebtsew/TSpec)



# Run (for simple usage)

Download executable [here](https://github.com/grebtsew/TSpec/releases) from latest release and get going. I recommend placing the application in system programs a using enviroment variables.

Windows:
```bash
TSpec.exe [OPTIONS]
```

Linux:
```bash
TSpec [OPTIONS]
```

# Examples 
These examples uses windows executable:

Minimal features:
```bash
TSpec.exe 
```
Setting symbols and sizes, using format raw
```bash
TSpec.exe --thresholds '\-45:|,\-50:-,\-65:.' --waterfall-height 20 --bins 80 --spectrum-height 10 --format raw --freq-min 0 --freq-max 500000  --db-min -120 --db-max 0
```

Using default color themes, calculate rssi and timestamp:
```bash
TSpec.exe --color-waterfall --color-spectrum --colormap inferno --spectrum-symbol " " --spectrum-symbol-color-background --rssi --timestamp
```

Using custom color themes and line graph with set refresh rate:
```bash
TSpec.exe --color-waterfall --color-waterfall --color-spectrum --colormap inferno --spectrum-symbol " " --spectrum-symbol-color-background --format raw  --line-width 3 --line --format raw   --colormap custom --custom-colormap "#000000,#aa0000,64" --refresh-rate 10
```

# Controls

(control_demo)[./docs/control_demo.gif]

## Run (for developers, python and docker users)
Run in python use:
```bash
python TSpec.py [OPTIONS]
```
Or in windows use the release .exe file, a corresponding file in release for linux users exist.

Run in docker using:
```bash
docker-compose up
```


## Installation (for developers or python users )

1. Clone the repository:
```bash
git clone https://github.com/grebtsew/TSpec
cd ./TSpec
```
2. Install required:
```bash
pip install -r ./requirements.txt
```

3. Make sure the script is executable:
```bash
chmod +x TSpec.py
```

# Features

The table below describes all available command-line flags and their corresponding features.


| Argument                             | Description                                                         | Default / Notes |
| ------------------------------------ | ------------------------------------------------------------------- | --------------- |
| `--address`                          | IP address of the radio device                                      | `127.0.0.1`     |
| `--port`                             | TCP/UDP port                                                        | `5005`          |
| `--format`                           | Stream format (`vita49`, `raw`, `simulator`)                        | `vita49`        |
| `--bins`                             | Number of frequency bins                                            | `80`            |
| `--waterfall-height`                 | Maximum waterfall rows                                              | `10`            |
| `--color-waterfall`                  | Enable color in waterfall display                                   | -               |
| `--color-spectrum`                   | Enable color in spectrum display                                    | -               |
| `--colormap`                         | Colormap to use (`viridis`, `magma`, `plasma`, `inferno`, `custom`) | `viridis`       |
| `--custom-colormap`                  | Custom colormap: startcolor,stopcolor,steps                         | -               |
| `--spectrum-height`                  | Number of rows for spectrum                                         | `10`            |
| `--spectrum-symbol`                  | Symbol used for spectrum bars or line                               | `.`             |
| `--spectrum-symbol-color-background` | Display spectrum symbol with colored background                     | -               |
| `--bar`                              | Display spectrum in bar mode                                        | Default         |
| `--line`                             | Display spectrum in line mode (contour line)                        | -               |
| `--line-width`                       | Vertical thickness of line in line mode                             | `1`             |
| `--freq-min`                         | Minimum frequency to display (Hz)                                   | -               |
| `--freq-max`                         | Maximum frequency to display (Hz)                                   | -               |
| `--auto-zoom`                        | Automatically zoom to signal region                                 | -               |
| `--auto-zoom-threshold`              | dB above noise floor considered signal                              | `10`            |
| `--auto-zoom-iterations`             | Number of autozoom iterations (0 = infinite)                        | `0`             |
| `--store`                            | Save incoming IQ data to a file                                     | -               |
| `--load`                             | Load IQ data from a file instead of UDP                             | -               |
| `--hide-spectrum`                    | Do not display spectrum output                                      | -               |
| `--hide-waterfall`                   | Do not display waterfall output                                     | -               |
| `--refresh-rate`                     | Maximum refresh rate in Hz                                          | None            |
| `--max-delta-db`                     | Maximum allowed jump in dB per refresh                              | None            |
| `--smoothing`                        | Apply exponential moving average smoothing (0.0 = off)              | `0.0`           |
| `--timestamp`                        | Print timestamp with each frame                                     | -               |
| `--decimate`                         | Use only every Nth sample to reduce sample rate                     | `1`             |
| `--fft-size`                         | FFT size (default = length of input block)                          | None            |
| `--window`                           | FFT window function (`hann`, `hamming`, `blackman`, `rectangular`)  | `hann`          |
| `--no-normalize`                     | Don't normalize spectrum to 0 dB max                                | -               |
| `--waterfall-scale`                  | Scale type for waterfall (`log` or `linear`)                        | `log`           |
| `--waterfall-speed`                  | Number of waterfall updates to skip per frame                       |                 |
| `--clear-on-new-frame`               | Determine if to clear terminal each frame. This might cause flimmer |                 |
| `--freq-offset`                      | Frequency Offset                                                    |  0               |
| `--block-size`                       | Block size (number of samples per FFT). Default = use fft-size or input length    | 0                |
| `--db-min`                           | Minimum dB level for display (clip). If not set, auto or normalize is used       | None             |
| `--db-max`                           | Maximum dB level for display (clip). If not set, auto or normalize is used       | None             |
| `--waterfall-gamma`                   | Gamma correction for waterfall colors (default 1.0 = linear)                     | 1.0              |
| `--dtype`                             | Data type of incoming IQ samples (float32, int16, int8). Default = float32       | float32          |
| `--avg-blocks` | Average over N blocks for smoother display (1 = off) | 1 |
| `--maxhold` | Show maximum value per bin across frames instead of current spectrum | False |
| `--rssi` | Display RSSI (Received Signal Strength Indicator) of the shown spectrum instead of per-bin values | False |
| `--ignore-missing-meta` | Ignore IQ blocks that are missing metadata instead of raising errors | False |
| `--feature-symbol` | Symbol used to mark extracted features (e.g., peak) in the spectrum. | * |
| `--feature-color` | Color for extracted feature (format: 'R,G,B' 0-255), overrides symbol color if set. | None |
| `--feature-avg-offset` | Offset for feature extraction detection, threshold avg + this offset. | 5 |





# Features

- Terminal-based **spectrum and waterfall display** with ASCII symbols or colored visualization.
- Support for colormaps: `viridis`, `magma`, `plasma`, `inferno`, or custom colormaps.
- Customizable thresholds and symbols for the waterfall.
- Auto-zoom functionality that focuses on signals above the noise floor.
- Ability to save incoming IQ data to a file and load IQ data from files for replay.
- Adjustable display settings including spectrum height, waterfall height, symbol choice, and line/bar modes.
- Supports maximum refresh rates and clamping of rapid dB changes for smoother visualizations.

# Input

The input folder contains the following input-examples:
- rtl-sdr example using minimized vita49
- microphone data as raw
- simulator data as raw
- simulator data as simulator (own) format

# Dependencies

The only required dependency except for python3 is numpy for performing the FFT and handling data arrays more effectively. For testing and examples, some more libraries are needed!

# Testing
Vital and core functionality are tested with pytest. To run tests yourself enter `Testing`-folder and run:
```cmd
pytest .
```

# Format
This repository uses python `black` to keep code format.

# License
The project uses MIT License, read more [here](./License).


# License
The project uses MIT License, read more [here](./License).


# Disclaimer

Large quantities of this code and documentation is generated using ChatGPT-5 mini 2025.



@Grebtsew 2025
