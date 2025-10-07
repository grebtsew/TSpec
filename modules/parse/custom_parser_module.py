# --format parse-module
# --parse-module ./modules/parse/custom_parser_module.py
"""
Template for a custom SDR data parser module.
"""

import numpy as np


def parse_data(data):
    """
    Required:
        - Accepts raw bytes (from network, file, etc.)
        - Returns tuple (iq_data, meta)
    """
    # Parse or decode your data here
    # Return None, meta if you just received metadata

    iq_arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
    iq_data = iq_arr[:, 0] + 1j * iq_arr[:, 1]
    meta = {
        "sample_rate": 48000,
        "center_frequency": 0,
        "stream_id": "custom",
    }
    return iq_data, meta
