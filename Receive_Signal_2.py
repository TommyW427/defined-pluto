#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: BPSK Pluto Receiver with M&M Clock Recovery
# GNU Radio version: 3.10.12.0
#
# ════════════════════════════════════════════════════════════════
#  RECEIVER SIGNAL CHAIN
# ════════════════════════════════════════════════════════════════
#
# This receiver undoes what the transmitter did, in reverse order:
#
#   PlutoSDR Antenna
#        │
#        ▼
#   ┌──────────────────────┐
#   │  PlutoSDR Source      │  Receives RF at 915 MHz, outputs
#   │  (soapy_plutosdr)     │  complex baseband IQ at 250 kHz
#   └──────────┬───────────┘
#              │  250,000 complex samples/sec
#              ▼
#   ┌──────────────────────┐
#   │  AGC                  │  Automatic Gain Control.
#   │  (analog.agc_cc)      │  Normalizes the signal amplitude to
#   │                       │  a reference level of 1.0 regardless
#   │                       │  of how strong/weak the received signal
#   │                       │  is.  This is needed because the Costas
#   │                       │  loop and clock recovery expect a
#   │                       │  consistent signal level.
#   │                       │  Rate=1e-4 means it adapts slowly
#   │                       │  (averaged over ~10,000 samples).
#   └──────────┬───────────┘
#              │  still 250,000 samples/sec, but amplitude ≈ 1.0
#              ▼
#   ┌──────────────────────┐
#   │  Costas Loop          │  Frequency and phase synchronization.
#   │  (digital.costas_     │
#   │   loop_cc, order=2)   │  Problem: the transmitter and receiver
#   │                       │  oscillators are not perfectly matched.
#   │                       │  Even a tiny frequency offset (say 50 Hz)
#   │                       │  causes the received constellation to
#   │                       │  spin continuously.  For BPSK, the two
#   │                       │  points ±1 would rotate in a circle
#   │                       │  instead of sitting still on the real axis.
#   │                       │
#   │                       │  The Costas loop is a PLL (phase-locked
#   │                       │  loop) that tracks and removes this
#   │                       │  rotation.  Order=2 means it's designed
#   │                       │  for BPSK (2 constellation points).
#   │                       │  It multiplies the signal by e^{-jθ(t)}
#   │                       │  where θ(t) tracks the phase/freq offset.
#   │                       │
#   │                       │  loop_bw controls how fast it adapts:
#   │                       │  too fast → noisy tracking
#   │                       │  too slow → can't follow freq offset
#   │                       │  0.0628 rad (≈ 2π/100) is a common start.
#   └──────────┬───────────┘
#              │  still 250,000 samples/sec, but phase is locked
#              ▼
#   ┌──────────────────────┐
#   │  Symbol Sync          │  Clock recovery using Mueller & Mueller
#   │  (digital.symbol_     │  Timing Error Detector (TED).
#   │   sync_cc)            │
#   │                       │  Problem: we have 25 samples per symbol
#   │  TED: Mueller &       │  (sps=25), but we don't know which of
#   │       Mueller         │  those 25 samples is the "best" one to
#   │                       │  sample at (the decision point).
#   │                       │
#   │                       │  Mueller & Mueller TED works by looking
#   │                       │  at the current symbol, the previous
#   │                       │  symbol, and the midpoint between them:
#   │                       │
#   │                       │    e(n) = Re{ x̂(n-1)·y(n) - x̂(n)·y(n-1) }
#   │                       │
#   │                       │  where x̂ is the hard decision and y is
#   │                       │  the received sample.  This error drives
#   │                       │  a loop that adjusts the sampling instant.
#   │                       │
#   │                       │  The block also resamples the signal using
#   │                       │  an MMSE interpolating filter (8-tap) so
#   │                       │  it can sample at fractional positions
#   │                       │  between the input samples.
#   │                       │
#   │                       │  Input:  25 samples per symbol
#   │                       │  Output: 1 sample per symbol (osps=1)
#   └──────────┬───────────┘
#              │  10,000 symbols/sec (1 complex value per symbol)
#              │  These are the y_t values from the paper:
#              │    y_t = h · x_t + z_t
#              ▼
#   ┌──────────────────────┐
#   │  File Sink            │  Saves the synchronized, downsampled
#   │  (blocks.file_sink)   │  symbols to disk as complex64.
#   │                       │  Each value is one symbol-rate sample
#   │                       │  ready for MMSE detection or DEFINED.
#   └──────────────────────┘
#
# The output file contains one complex value per symbol period.
# You do NOT need to downsample again — the clock recovery block
# already outputs at 1 sample/symbol (osps=1).
#

from PyQt5 import Qt
from PyQt5 import QtCore
from gnuradio import qtgui
from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import soapy
import sip
import threading


# ── Output file for synchronized symbols ────────────────────────
# Each complex64 value in this file is one symbol-rate sample.
# These are ready to pass directly into the MMSE or DEFINED pipeline.
SYMBOL_OUTPUT_FILE = "received_symbols.bin"

# Also save raw IQ (before clock recovery) for debugging
RAW_IQ_OUTPUT_FILE = "received_iq_raw.bin"
# ────────────────────────────────────────────────────────────────


class Receive_Signal(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "BPSK Pluto Receiver", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("BPSK Pluto Receiver - M&M Clock Recovery")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)

        # ── Qt GUI boilerplate ──────────────────────────────────
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "Receive_Signal")
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################

        # Sample rate: must match the transmitter (1 MHz)
        self.samp_rate = samp_rate = 1000000

        # Center frequency: must match the transmitter (915 MHz)
        self.center_freq = center_freq = 915e6

        # Symbol rate: must match the transmitter (40 kHz)
        # The transmitter sends each BPSK symbol for sps=25 samples.
        self.symbol_rate = symbol_rate = 40000

        # Samples per symbol: how many raw IQ samples make up one symbol.
        # The transmitter's repeat block uses this same value.
        # sps = 1,000,000 / 40,000 = 25
        self.sps = sps = int(samp_rate // symbol_rate)

        # Receiver gain in dB.  Adjustable via GUI slider.
        # Higher gain = more sensitivity but also more noise/distortion.
        # Start at 40 dB and adjust based on what the constellation looks like.
        self.rx_gain = rx_gain = 40

        # Costas loop bandwidth (radians per sample).
        # Controls how fast the frequency/phase tracking adapts.
        # 2*pi/100 ≈ 0.0628 is a typical starting value.
        # Increase if the frequency offset is large or changing fast.
        # Decrease for cleaner tracking at the cost of slower lock.
        self.costas_bw = costas_bw = 0.0628

        # Mueller & Mueller clock recovery loop bandwidth.
        # Controls how fast the symbol timing adapts.
        # Typical range: 0.01 to 0.1
        # Smaller = more stable but slower to lock
        # Larger = faster lock but noisier timing
        self.clock_bw = clock_bw = 0.045

        ##################################################
        # Blocks
        ##################################################

        # ── GUI: RX gain slider ─────────────────────────────────
        # Lets you adjust gain while running to find the sweet spot
        self._rx_gain_range = qtgui.Range(0, 73, 1, rx_gain, 200)
        self._rx_gain_win = qtgui.RangeWidget(
            self._rx_gain_range, self.set_rx_gain, "RX Gain (dB)",
            "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.top_grid_layout.addWidget(self._rx_gain_win, 0, 0, 1, 1)

        # ── GUI: Costas loop BW slider ──────────────────────────
        self._costas_bw_range = qtgui.Range(0.001, 0.2, 0.001, costas_bw, 200)
        self._costas_bw_win = qtgui.RangeWidget(
            self._costas_bw_range, self.set_costas_bw, "Costas Loop BW",
            "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.top_grid_layout.addWidget(self._costas_bw_win, 0, 1, 1, 1)

        # ── GUI: Clock recovery BW slider ───────────────────────
        self._clock_bw_range = qtgui.Range(0.001, 0.2, 0.001, clock_bw, 200)
        self._clock_bw_win = qtgui.RangeWidget(
            self._clock_bw_range, self.set_clock_bw, "Clock Recovery BW",
            "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.top_grid_layout.addWidget(self._clock_bw_win, 0, 2, 1, 1)

        # ── PlutoSDR Source ─────────────────────────────────────
        # Receives RF signal at center_freq, downconverts to baseband,
        # and outputs complex IQ samples at samp_rate.
        dev = 'driver=plutosdr'
        stream_args = ''
        tune_args = ['']
        settings_list = ['']

        self.soapy_plutosdr_source_0 = soapy.source(
            dev, "fc32", 1, "",
            stream_args, tune_args, settings_list
        )
        self.soapy_plutosdr_source_0.set_sample_rate(0, samp_rate)
        self.soapy_plutosdr_source_0.set_bandwidth(0, samp_rate)
        self.soapy_plutosdr_source_0.set_gain_mode(0, False)
        self.soapy_plutosdr_source_0.set_frequency(0, center_freq)
        self.soapy_plutosdr_source_0.set_gain(0, min(max(rx_gain, 0.0), 73.0))

        # ── AGC (Automatic Gain Control) ────────────────────────
        # Normalizes signal amplitude to reference=1.0.
        #
        # Parameters:
        #   rate=1e-4 : adaptation speed.  1e-4 means the gain
        #               adjusts slowly, averaging over ~10,000 samples.
        #               This prevents the AGC from tracking individual
        #               symbol transitions (which would destroy the signal).
        #   reference=1.0 : target output amplitude
        #   gain=1.0 : initial gain before adaptation starts
        #   max_gain=65536 : upper limit on the AGC gain (safety limit)
        self.analog_agc_xx_0 = analog.agc_cc((1e-4), 1.0, 1.0, 65536)

        # ── Costas Loop (Frequency + Phase Synchronization) ─────
        # Removes frequency offset and phase offset between the
        # transmitter and receiver oscillators.
        #
        # For BPSK, order=2.  The loop tracks the carrier phase
        # and multiplies the input by e^{-jθ(t)} to derotate it.
        #
        # After the Costas loop, the BPSK constellation should sit
        # on the real axis at approximately ±1 (for a normalized signal).
        #
        # Parameters:
        #   w=costas_bw : loop bandwidth in radians per sample
        #   order=2 : BPSK has 2 constellation points
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc(
            costas_bw,  # loop bandwidth
            2,          # order = 2 for BPSK
            False       # use_snr = False (don't use SNR-based BW adjustment)
        )

        # ── Symbol Sync with Mueller & Mueller TED ──────────────
        # Performs clock recovery: finds the optimal sampling instant
        # within each symbol period and outputs exactly 1 sample
        # per symbol.
        #
        # The Mueller & Mueller Timing Error Detector (TED) computes:
        #   e(n) = Re{ x̂(n-1)·y(n) - x̂(n)·y(n-1) }
        # where x̂ is the hard decision (nearest constellation point)
        # and y is the interpolated sample.  The error e(n) is positive
        # if we're sampling too late, negative if too early.
        #
        # This error feeds a second-order loop filter that adjusts
        # a numerically controlled oscillator (NCO) to shift the
        # sampling phase.  The MMSE 8-tap interpolating filter lets
        # us sample at fractional positions between input samples.
        #
        # Parameters:
        #   TED_MUELLER_AND_MULLER : which timing error detector to use
        #   sps : expected input samples per symbol (25)
        #   loop_bw : how fast the timing loop adapts
        #   damping_factor=1.0 : critically damped loop (standard choice)
        #   ted_gain=1.0 : TED output scaling
        #   max_deviation=1.5 : max fractional timing adjustment per symbol
        #                       (1.5 means the loop can shift ±1.5 samples
        #                       per symbol period to track timing drift)
        #   osps=1 : output samples per symbol (we want 1)
        #   constellation : BPSK constellation for the hard slicer
        #                   inside the M&M TED (it needs to make hard
        #                   decisions to compute the timing error)
        #   resamp_type : interpolation filter type
        #                 IR_MMSE_8TAP = minimum mean square error
        #                 8-tap interpolator (good quality)
        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_MUELLER_AND_MULLER,  # timing error detector type
            sps,                             # input samples per symbol
            clock_bw,                        # loop bandwidth
            1.0,                             # damping factor
            1.0,                             # TED gain
            1.5,                             # max deviation (samples)
            1,                               # output samples per symbol
            digital.constellation_bpsk().base(),  # constellation for slicer
            digital.IR_MMSE_8TAP,            # interpolation filter type
            128,                             # n_filters (polyphase arms)
            []                               # taps (empty = use default)
        )

        # ── File Sink: synchronized symbols ─────────────────────
        # Saves the output of the clock recovery block.
        # Each value in this file is ONE complex sample representing
        # ONE symbol period.  These are the y_t values from the paper:
        #   y_t = h · x_t + z_t
        # ready for MMSE estimation or DEFINED processing.
        #
        # File format: raw complex64 (pairs of float32: I, Q, I, Q, ...)
        # To load in Python:
        #   symbols = np.fromfile("received_symbols.bin", dtype=np.complex64)
        self.blocks_file_sink_symbols = blocks.file_sink(
            gr.sizeof_gr_complex * 1,  # item size: one complex value
            SYMBOL_OUTPUT_FILE,        # output file path
            False                      # append=False (overwrite)
        )
        self.blocks_file_sink_symbols.set_unbuffered(False)

        # ── File Sink: raw IQ (for debugging) ───────────────────
        # Saves the raw IQ before any synchronization, in case you
        # need to inspect or reprocess it offline.
        self.blocks_file_sink_raw = blocks.file_sink(
            gr.sizeof_gr_complex * 1,
            RAW_IQ_OUTPUT_FILE,
            False
        )
        self.blocks_file_sink_raw.set_unbuffered(False)

        # ── GUI: Constellation display (after clock recovery) ───
        # Shows the received constellation after all synchronization.
        # For a good BPSK signal, you should see two tight clusters
        # near -1 and +1 on the real axis.
        # If they're spinning: Costas loop hasn't locked (increase BW).
        # If they're smeared: clock recovery hasn't locked (adjust BW).
        # If they're scattered everywhere: no signal or SNR too low.
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024,                      # number of points to display
            "After Clock Recovery",    # title
            1,                         # number of inputs
            None                       # parent widget
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(
            qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, ""
        )
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)
        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_win, 1, 0, 1, 2)

        # ── GUI: Time domain display (after clock recovery) ─────
        # Shows the real part of the synchronized symbols over time.
        # For BPSK, this should look like a clean rectangular wave
        # jumping between approximately -1 and +1.
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            200,                       # number of points to display
            symbol_rate,               # sample rate for x-axis scaling
            "Synchronized Symbols",    # title
            1,                         # number of inputs
            None                       # parent widget
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")
        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(
            qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, ""
        )
        self.qtgui_time_sink_x_0.enable_autoscale(True)
        self.qtgui_time_sink_x_0.enable_grid(True)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)
        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_0_win, 1, 2, 1, 1)

        # ── GUI: Frequency display (before sync, for diagnostics) ──
        # Shows the RF spectrum to verify the signal is present
        # and centered at the right frequency.
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024,                      # FFT size
            window.WIN_BLACKMAN_hARRIS, # window function
            center_freq,               # center frequency for display
            samp_rate,                 # sample rate
            "Received Spectrum",       # title
            1,                         # number of inputs
            None                       # parent widget
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_0.set_y_label('', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(True)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)
        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 2, 0, 1, 3)

        ##################################################
        # Connections
        ##################################################

        # PlutoSDR → raw file sink (save unprocessed IQ for debugging)
        self.connect((self.soapy_plutosdr_source_0, 0),
                     (self.blocks_file_sink_raw, 0))

        # PlutoSDR → spectrum display
        self.connect((self.soapy_plutosdr_source_0, 0),
                     (self.qtgui_freq_sink_x_0, 0))

        # PlutoSDR → AGC → Costas loop → Clock recovery
        self.connect((self.soapy_plutosdr_source_0, 0),
                     (self.analog_agc_xx_0, 0))
        self.connect((self.analog_agc_xx_0, 0),
                     (self.digital_costas_loop_cc_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 0),
                     (self.digital_symbol_sync_xx_0, 0))

        # Clock recovery → symbol file sink
        self.connect((self.digital_symbol_sync_xx_0, 0),
                     (self.blocks_file_sink_symbols, 0))

        # Clock recovery → constellation display
        self.connect((self.digital_symbol_sync_xx_0, 0),
                     (self.qtgui_const_sink_x_0, 0))

        # Clock recovery → time domain display
        self.connect((self.digital_symbol_sync_xx_0, 0),
                     (self.qtgui_time_sink_x_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "Receive_Signal")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    # ── Variable getters and setters ────────────────────────────
    # These are called by the GUI when you move a slider.

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.soapy_plutosdr_source_0.set_gain(0, min(max(rx_gain, 0.0), 73.0))

    def get_costas_bw(self):
        return self.costas_bw

    def set_costas_bw(self, costas_bw):
        self.costas_bw = costas_bw
        self.digital_costas_loop_cc_0.set_loop_bandwidth(costas_bw)

    def get_clock_bw(self):
        return self.clock_bw

    def set_clock_bw(self, clock_bw):
        self.clock_bw = clock_bw
        self.digital_symbol_sync_xx_0.set_loop_bandwidth(clock_bw)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.sps = int(self.samp_rate // self.symbol_rate)
        self.soapy_plutosdr_source_0.set_sample_rate(0, self.samp_rate)
        self.soapy_plutosdr_source_0.set_bandwidth(0, self.samp_rate)

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.soapy_plutosdr_source_0.set_frequency(0, self.center_freq)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.center_freq, self.samp_rate)

    def get_symbol_rate(self):
        return self.symbol_rate

    def set_symbol_rate(self, symbol_rate):
        self.symbol_rate = symbol_rate
        self.sps = int(self.samp_rate // self.symbol_rate)


def main(top_block_cls=Receive_Signal, options=None):
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Timer keeps the Python event loop alive so Ctrl+C works
    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == '__main__':
    main()
