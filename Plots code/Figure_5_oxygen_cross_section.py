import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl


plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'stix',
})


def read_curtain_data(filename: str, n_depth: int, n_time: int):
    """Read HYDRUS profile output data."""
    raw = []
    try:
        with open(filename, "r") as f:
            f.readline()
            f.readline()
            for _ in range(n_time):
                for _ in range(n_depth):
                    items = f.readline().split()
                    if not items:
                        continue
                    depth_cm, val = float(items[1]), float(items[2])
                    raw.append((depth_cm, val))
                f.readline()
    except FileNotFoundError:
        print(f"Error: File not found at {filename}, generating dummy data.")
        return np.linspace(0, -30, n_depth), np.random.rand(n_depth, n_time)

    raw = np.asarray(raw)
    depth_m = -raw[:n_depth, 0] / 100.0
    matrix = raw[:, 1].reshape(n_time, n_depth).T

    if depth_m[0] < depth_m[-1]:
        depth_m, matrix = depth_m[::-1], matrix[::-1, :]

    return depth_m, matrix


def add_soil_legend(fig, ax, layer_interfaces_m, textures=None):
    """Add a soil-layer legend using the actual layer thicknesses."""
    layer_interfaces_m = np.asarray(layer_interfaces_m, dtype=float)
    n_layers = len(layer_interfaces_m) - 1

    if textures is None:
        textures = ["Sand", "Loamy Sand", "Sand", "Loam", "Sandy Loam", "Sand"]

    if len(textures) != n_layers:
        raise ValueError("textures length must equal len(layer_interfaces_m)-1")

    fig.canvas.draw()
    pos = ax.get_position()

    offset_left = 0.145
    inset_left = pos.x0 - offset_left
    if inset_left < 0.005:
        inset_left = 0.005

    inset_width = 0.04
    inset = fig.add_axes([inset_left, pos.y0, inset_width, pos.height])
    inset.set_xlim(0, 1)
    inset.set_ylim(layer_interfaces_m.min(), layer_interfaces_m.max())
    inset.axis("off")

    texture_to_gray = {
        "Sand": "#4c4c4c",
        "Loamy Sand": "#a6a6a6",
        "Loam": "#2e2e2e",
        "Sandy Loam": "#7a7a7a"
    }

    gray = [texture_to_gray.get(t, "#777777") for t in textures]

    for i in range(n_layers):
        top = layer_interfaces_m[i]
        bottom = layer_interfaces_m[i + 1]
        height = top - bottom

        inset.add_patch(
            Rectangle((0, bottom), 1, height, facecolor=gray[i], edgecolor="white")
        )

        inset.text(
            0.5, bottom + height / 2, str(i + 1),
            ha="center", va="center",
            color="white", fontsize=16
        )


def plot_three_oxygen_profiles():
    n_depth, n_time = 329, 121
    dt_h = 6
    dt_day = dt_h / 24.0
    t_day = np.arange(n_time) * dt_day

    oxy_paths = [
        "/Users/zhaozhe/Desktop/Project/Hydrus-2D/4d-6d/CS 4d-6d conc.txt",
        "/Users/zhaozhe/Desktop/Project/Hydrus-2D/6d-4d/CS 6d-4d conc.txt",
        "/Users/zhaozhe/Desktop/Project/Hydrus-2D/9d-1d/CS 9d-1d conc.txt"
    ]

    layer_interfaces_m = [0, -8, -11.5, -17.5, -20.5, -26, -30]
    textures = ["Sand", "Loamy Sand", "Sand", "Loam", "Sandy Loam", "Sand"]

    letters = "abc"

    fig, axs = plt.subplots(
        3, 1, figsize=(12, 13), sharex=True,
        gridspec_kw={"hspace": 0.1}
    )

    plt.subplots_adjust(left=0.16, right=0.90, top=0.96, bottom=0.08)

    for idx, path in enumerate(oxy_paths):
        depth_oxy, oxy = read_curtain_data(path, n_depth, n_time)
        ax = axs[idx]

        im = ax.pcolormesh(t_day, depth_oxy, oxy, shading="gouraud", cmap='viridis')

        ax.set_ylabel("Depth (m)", labelpad=10)
        ax.set_yticks([0, -10, -20, -30])
        ax.set_ylim(-30, 0)

        ax.text(
            0.98, 0.9, f"{letters[idx]})",
            transform=ax.transAxes, color='white',
            fontsize=24, ha='right', va='center'
        )

        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)

        cb.set_label("Oxygen (g/cm$^3$)", labelpad=10, fontsize=20)

        cb.formatter = ScalarFormatter(useMathText=True)
        cb.formatter.set_scientific(True)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('left')
        cb.update_ticks()
        cb.ax.yaxis.get_offset_text().set_fontsize(16)

        add_soil_legend(fig, ax, layer_interfaces_m, textures=textures)

    axs[-1].set_xlabel("Time (d)", fontsize=20, labelpad=10)
    axs[-1].set_xlim(0, 30)

    outfile = "/Users/zhaozhe/Desktop/Project/Thesis/oxygen_three_profiles_revised_v4_compact_nobold.png"
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    print(f"Figure saved to {outfile}")
    plt.show()


if __name__ == "__main__":
    plot_three_oxygen_profiles()