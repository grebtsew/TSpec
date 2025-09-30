#!/usr/bin/env python3
import sys
import numpy as np
from rtlsdr import RtlSdr
from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- RTL-SDR inställningar ---
CENTER_FREQ = 100e6  # Hz
SAMPLE_RATE = 1e6  # Hz
BUF_SAMPLES = 4096
GAIN = "auto"


class SDRPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTL-SDR Spectrum & Waterfall")
        self.resize(1000, 600)

        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax_spec = self.fig.add_subplot(211)
        self.ax_waterfall = self.fig.add_subplot(212)

        self.setCentralWidget(self.canvas)

        self.waterfall_height = 200
        self.waterfall = np.zeros((self.waterfall_height, BUF_SAMPLES // 2))
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # uppdatera var 50 ms

        self.sdr = RtlSdr()
        self.sdr.sample_rate = SAMPLE_RATE
        self.sdr.center_freq = CENTER_FREQ
        self.sdr.gain = GAIN

    def update_plot(self):
        iq = self.sdr.read_samples(BUF_SAMPLES)
        spectrum = np.fft.fftshift(np.fft.fft(iq)[: BUF_SAMPLES // 2])
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

        # update waterfall
        self.waterfall = np.roll(self.waterfall, -1, axis=0)
        self.waterfall[-1, :] = power_db

        # plot spectrum
        self.ax_spec.clear()
        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / SAMPLE_RATE))[
            : BUF_SAMPLES // 2
        ]
        self.ax_spec.plot(freqs / 1e6, power_db)
        self.ax_spec.set_ylabel("dB")
        self.ax_spec.set_title(f"Spectrum @ {CENTER_FREQ/1e6:.3f} MHz")

        # plot waterfall
        self.ax_waterfall.clear()
        self.ax_waterfall.imshow(
            self.waterfall,
            aspect="auto",
            origin="lower",
            extent=[freqs[0] / 1e6, freqs[-1] / 1e6, 0, self.waterfall_height],
            cmap="viridis",
        )
        self.ax_waterfall.set_ylabel("Time")
        self.ax_waterfall.set_xlabel("Frequency [MHz]")

        self.canvas.draw()

    def closeEvent(self, event):
        self.sdr.close()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SDRPlot()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
