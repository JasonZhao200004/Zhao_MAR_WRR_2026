import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter


# Global plotting settings
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 15,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.1,
    'xtick.major.width': 1.1,
    'ytick.major.width': 1.1,
    'xtick.major.size': 4.5,
    'ytick.major.size': 4.5,
})


def read_curtain_data(filename: str, n_depth: int, n_time: int):
    """Read HYDRUS curtain-plot output data."""
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
        depth_m = depth_m[::-1]
        matrix = matrix[::-1, :]

    return depth_m, matrix


def compute_decay_rate(theta: np.ndarray):
    """Calculate the moisture-dependent first-order respiration rate."""
    theta = np.maximum(theta, 0)

    sf = np.where(
        theta <= 0.06, 0.1,
        np.where(
            theta < 0.15, 10 * theta - 0.5,
            np.where(
                theta <= 0.25, 1.0,
                np.where(theta < 0.37, -5.8333333 * theta + 2.4583333, 0.3)
            )
        )
    )

    return sf * 0.06


def make_blue_white_red(N: int = 256) -> ListedColormap:
    """Create a blue-white-red colormap centered around white."""
    half = N // 2

    b2w = np.column_stack([
        np.linspace(0, 1, half),
        np.linspace(0, 1, half),
        np.ones(half)
    ])

    w2r = np.column_stack([
        np.ones(half),
        np.linspace(1, 0, half),
        np.linspace(1, 0, half)
    ])

    if N % 2:
        cmap = np.vstack([b2w, np.array([[1, 1, 1]]), w2r])
    else:
        cmap = np.vstack([b2w, w2r])

    return ListedColormap(cmap)


def add_soil_legend(fig, ax, layer_interfaces_m, textures=None):
    """
    Add a soil-layer bar on the left side of each panel.

    Layer thicknesses are plotted in proportion to their actual thicknesses.
    """
    layer_interfaces_m = np.asarray(layer_interfaces_m, dtype=float)
    n_layers = len(layer_interfaces_m) - 1

    if textures is None:
        textures = ["Sand", "Loamy Sand", "Sand", "Loam", "Sandy Loam", "Sand"]

    if len(textures) != n_layers:
        raise ValueError("textures length must equal len(layer_interfaces_m) - 1")

    pos = ax.get_position()

    offset_left = 0.125
    inset_left = pos.x0 - offset_left
    if inset_left < 0.01:
        inset_left = 0.01

    inset_width = 0.030
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

    gray_levels = [texture_to_gray.get(t, "#777777") for t in textures]

    for i in range(n_layers):
        top = layer_interfaces_m[i]
        bottom = layer_interfaces_m[i + 1]
        height = top - bottom

        inset.add_patch(
            Rectangle(
                (0, bottom), 1, height,
                facecolor=gray_levels[i],
                edgecolor="white",
                linewidth=0.8
            )
        )

        inset.text(
            0.5,
            bottom + height / 2,
            str(i + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=13
        )


def add_wet_drain_bar(ax, pattern, y0=1.095, height=0.19):
    """Add a wetting-draining schedule bar above the first panel."""
    ax_bar = ax.inset_axes([0.0, y0, 1.0, height], transform=ax.transAxes)
    ax_bar.set_xlim(ax.get_xlim())
    ax_bar.set_ylim(0, 1)
    ax_bar.axis("off")

    colours = {
        "wet": "#2ca02c",
        "drain": "#4b0082"
    }

    x_start = 0.0

    for label, dur_h in pattern:
        dur_day = dur_h / 24.0

        ax_bar.add_patch(
            Rectangle(
                (x_start, 0),
                dur_day,
                1,
                facecolor=colours[label],
                edgecolor="white",
                linewidth=0.9
            )
        )

        if dur_day >= 2.2:
            text_label = label
            fs = 15
        elif dur_day >= 1.0:
            text_label = "W" if label == "wet" else "D"
            fs = 13
        else:
            text_label = None

        if text_label is not None:
            ax_bar.text(
                x_start + dur_day / 2,
                0.5,
                text_label,
                ha="center",
                va="center",
                color="white",
                fontsize=fs
            )

        x_start += dur_day


def plot_curtain_with_legends():
    # Model output dimensions and time step
    n_depth, n_time = 329, 121
    dt_h = 6
    dt_day = dt_h / 24.0
    t_day = np.arange(n_time) * dt_day

    # HYDRUS output folder for the 4-day wetting and 6-day draining scenario
    base_dir = "/Users/zhaozhe/Desktop/Project/Hydrus-2D/4d-6d/"

    depth_oxy, oxy = read_curtain_data(
        base_dir + "CS 4d-6d conc.txt",
        n_depth,
        n_time
    )

    depth_vel, vel = read_curtain_data(
        base_dir + "CS 4d-6d v.txt",
        n_depth,
        n_time
    )

    depth_wat, wat = read_curtain_data(
        base_dir + "CS 4d-6d th.txt",
        n_depth,
        n_time
    )

    wat_perc = wat / 0.38 * 100.0
    decay = compute_decay_rate(wat)
    vel_m = vel / 100.0
    dz = abs(depth_wat[1] - depth_wat[0])

    with np.errstate(divide='ignore', invalid='ignore'):
        da_log = np.log10((decay * dz) / (vel_m + 1e-10) + 1e-12)

    # Soil layer boundaries and texture classes
    layer_interfaces_m = [0, -8, -11.5, -17.5, -20.5, -26, -30]
    textures = ["Sand", "Loamy Sand", "Sand", "Loam", "Sandy Loam", "Sand"]

    fig, axs = plt.subplots(
        5,
        1,
        figsize=(11.6, 14.8),
        sharex=True,
        gridspec_kw={"hspace": 0.26}
    )

    plt.subplots_adjust(left=0.19, right=0.89, top=0.93, bottom=0.08)

    cfg = [
        (oxy, "Oxygen (g/cm$^3$)", "viridis", None),
        (wat_perc, "Water saturation (%)", "viridis", None),
        (vel, "Pore water velocity (cm/h)", "viridis", None),
        (decay, "Respiration rate (1/h)", "viridis", None),
        (da_log, "log(Damköhler) (-)", "bwr", make_blue_white_red(256)),
    ]

    letters = "abcde"

    for idx, (mat, clabel, cmap_name, custom_cmap) in enumerate(cfg):
        ax = axs[idx]
        z = depth_oxy
        final_cmap = custom_cmap if custom_cmap else cmap_name

        if idx == 4:
            lim = np.max(np.abs(da_log))
            norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)

            im = ax.pcolormesh(
                t_day,
                z,
                mat,
                shading="gouraud",
                cmap=final_cmap,
                norm=norm
            )

        else:
            im = ax.pcolormesh(
                t_day,
                z,
                mat,
                shading="gouraud",
                cmap=final_cmap
            )

        ax.set_ylabel("Depth (m)", labelpad=8)
        ax.set_yticks([0, -10, -20, -30])
        ax.set_ylim(-30, 0)
        ax.tick_params(direction="out")

        ax.text(
            0.985,
            0.91,
            f"{letters[idx]})",
            transform=ax.transAxes,
            color="white",
            fontsize=20,
            ha="right",
            va="center"
        )

        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3.0%", pad=0.06)
        cb = fig.colorbar(im, cax=cax)

        cb.set_label(clabel, labelpad=10, fontsize=17)
        cb.ax.tick_params(labelsize=15, width=1.0, length=3.5)

        cb.formatter = ScalarFormatter(useMathText=True)

        if idx == 1:
            cb.formatter.set_scientific(False)
        else:
            cb.formatter.set_scientific(True)
            cb.formatter.set_powerlimits((0, 0))

        cb.ax.yaxis.set_offset_position("left")
        cb.update_ticks()
        cb.ax.yaxis.get_offset_text().set_fontsize(13)

        offset_text = cb.ax.yaxis.get_offset_text()
        offset_text.set_verticalalignment("bottom")
        offset_text.set_position((0, 1.01))

        add_soil_legend(fig, ax, layer_interfaces_m, textures=textures)

    axs[-1].set_xlabel("Time (d)", fontsize=18, labelpad=8)
    axs[-1].set_xlim(0, 30)
    axs[-1].set_xticks([0, 5, 10, 15, 20, 25, 30])

    pattern = [("wet", 96), ("drain", 144)] * 3
    add_wet_drain_bar(axs[0], pattern, y0=1.095, height=0.19)

    outfile_png = (
        "/Users/zhaozhe/Desktop/Project/Thesis/"
        "CS_4d-6d_Figure_WRR_final_abbrev.png"
    )

    outfile_pdf = (
        "/Users/zhaozhe/Desktop/Project/Thesis/"
        "CS_4d-6d_Figure_WRR_final_abbrev.pdf"
    )

    fig.savefig(outfile_png, dpi=600, bbox_inches="tight")
    fig.savefig(outfile_pdf, bbox_inches="tight")

    print(f"Figure saved to:\n{outfile_png}\n{outfile_pdf}")

    plt.show()


if __name__ == "__main__":
    plot_curtain_with_legends()