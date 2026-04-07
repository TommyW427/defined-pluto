#!/usr/bin/env python3
"""
Frame Configuration
===================

Configuration dataclass for frame structure parameters.
"""

from dataclasses import dataclass


@dataclass
class FrameConfig:
    """Frame structure configuration"""
    preamble_bits: int = 2000
    pilot_bits: int = 64
    data_bits: int = 64
    modulation: str = "BPSK"
