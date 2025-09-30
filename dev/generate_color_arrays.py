#!/usr/bin/env python3


"""
PythonScript to generate new color schemes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ANSI 24-bit färg escape
def rgb_bg(r, g, b):
    """Return ANSI escape för bakgrundsfärg 24-bit"""
    r255, g255, b255 = int(r * 255), int(g * 255), int(b * 255)
    return f"\x1b[48;2;{r255};{g255};{b255}m \x1b[0m"


def print_colormap(name="viridis", steps=64):
    cmap = cm.get_cmap(name, steps)
    colors = cmap(np.linspace(0, 1, steps))[:, :3]  # RGB
    line = "".join([rgb_bg(r, g, b) for r, g, b in colors])
    print(f"{name}:")
    print(line)
    print("\n")


if __name__ == "__main__":
    for cmap_name in ["viridis", "magma", "plasma", "inferno"]:
        print_colormap(cmap_name)
