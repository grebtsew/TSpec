# TSpec


![CI](https://github.com/<USERNAME>/<REPO>/actions/workflows/python-tests.yml/badge.svg)

![demogif](./docs/demo.gif)

TSpec is a minimalist, high-performance terminal-based spectrum and waterfall display designed to run on any system, even without a GUI. It works seamlessly on older terminals and low-powered hardware, making it ideal for embedded systems or remote servers. With support for multiple stream formats and radio systems, TSpec ensures maximum flexibility while keeping resource usage minimal. Whether you need a quick visualization on a headless server or a lightweight monitoring tool, TSpec delivers fast, reliable, and portable spectrum analysis wherever you need it.


## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_folder>
```
2. Install required:
```bash
pip install -r ./requirements.txt
```

3. Make sure the script is executable:
```bash
chmod +x TSpec.py
```

## Run
Run in python use:
```bash
python TSpec.py [OPTIONS]
```
Or in windows use the release .exe file, corresponding for linux users.

Run in docker using:
```bash
docker-compose up
```

## Flags

The table below describe all available flags.


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
| `--clear-on-new-frame`                | Determine if to clear terminal each frame. This might cause flimmer                       |                 |




## Features

- Terminal-based **spectrum and waterfall display** with ASCII symbols or colored visualization.
- Support for colormaps: `viridis`, `magma`, `plasma`, `inferno`, or custom colormaps.
- Customizable thresholds and symbols for the waterfall.
- Auto-zoom functionality that focuses on signals above the noise floor.
- Ability to save incoming IQ data to a file and load IQ data from files for replay.
- Adjustable display settings including spectrum height, waterfall height, symbol choice, and line/bar modes.
- Supports maximum refresh rates and clamping of rapid dB changes for smoother visualizations.

## Examples

The example folder contain these examples:
- rtl-sdr example using minimized vita49
- microphone data as raw
- simulator data as raw
- simulator data as simulator (own) format

## Tests

-- todo


## License


## Changelog


## Disclaimer

Large quantities of this code and documentation is generated using ChatGPT-5 mini 2025.



@Grebtsew 2025




# TODO
badges
create demo gif
update readme


install

changelog
create release
to english