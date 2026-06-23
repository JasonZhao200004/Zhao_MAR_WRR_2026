import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, ScalarFormatter

# ==========================================
# 1. Read HYDRUS all-time cross-section output
# ==========================================
def read_hydrus_cross_section_all_times(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n_times = int(lines[0])   # first line = number of time blocks
    idx = 1

    depths_all = []
    conc_all = []

    for t in range(n_times):
        n_depth = int(lines[idx])
        idx += 1

        rows = []
        for _ in range(n_depth):
            parts = lines[idx].split()
            node_id = int(parts[0])
            depth_cm = float(parts[1])
            conc = float(parts[2])
            rows.append((node_id, depth_cm, conc))
            idx += 1

        rows = np.array(rows)
        depths_all.append(rows[:, 1])
        conc_all.append(rows[:, 2])

    depth_cm = depths_all[0]
    conc_all = np.array(conc_all)

    return depth_cm, conc_all


# ==========================================
# 2. File path
# ==========================================
file_path = "/Users/zhaozhe/Desktop/Project/paper figures/dw2_Henrys_constant_NO_FLOW_bacteria_1storder_la_less_than_lw_ftheta_2nd_Concentration_all_Time-Layers_Cross-Section_No_1.txt"

# Read data
depth_cm, conc_all = read_hydrus_cross_section_all_times(file_path)

# Convert depth from cm to m, downward negative
z_m = -depth_cm / 100.0

n_times = conc_all.shape[0]
n_depth = conc_all.shape[1]

print(f"Total number of profiles in file = {n_times}")
print(f"Number of depth nodes = {n_depth}")


# ==========================================
# 3. Define total simulation time and day conversion
# ==========================================
total_hours = 2880.0
hours_per_day = 24.0
hours_per_step = 120.0   # 5 days = 120 hours

total_days = total_hours / hours_per_day

print(f"Total simulation time = {total_hours:.1f} h")
print(f"Equivalent total days = {total_days:.1f} days")
print(f"Time interval between profiles = {hours_per_step:.1f} h = {hours_per_step / hours_per_day:.1f} days")

# Assume profiles are saved every 5 days
time_hours = np.arange(n_times) * hours_per_step
time_days = time_hours / hours_per_day

# Keep only Day 5 to Day 120
valid_mask = (time_hours > 0) & (time_hours <= total_hours)

selected_indices = np.where(valid_mask)[0]
selected_day_labels = time_days[valid_mask].astype(int)

print("Selected day -> profile index:")
for d, idx in zip(selected_day_labels, selected_indices):
    print(f"Day {d:3d} -> profile {idx + 1:2d} (time = {time_hours[idx]:.1f} h)")


# ==========================================
# 4. Find positive values for log x-axis
# ==========================================
all_positive_vals = conc_all[conc_all > 0]

if len(all_positive_vals) == 0:
    raise ValueError("No positive concentration values found. Cannot plot on log scale.")

cmin = np.min(all_positive_vals)
cmax = np.max(all_positive_vals)

xmin_log = 10 ** np.floor(np.log10(cmin))
xmax_log = 10 ** np.ceil(np.log10(cmax))

print(f"Global positive concentration range = {cmin:.3e} to {cmax:.3e}")
print(f"Auto log x-axis limits = {xmin_log:.3e} to {xmax_log:.3e}")

# Linear x-axis limits
xmin_linear = 0.0
xmax_linear = 8.5e-6

# Keep all data visible if maximum concentration is slightly larger than 8.5e-6
if cmax > xmax_linear:
    xmax_linear = cmax * 1.03

print(f"Linear x-axis limits = {xmin_linear:.3e} to {xmax_linear:.3e}")


# ==========================================
# 5. Plot style
# ==========================================
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 24,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
})


# ==========================================
# 6. Save directory
# ==========================================
save_dir = "/Users/zhaozhe/Desktop/Project/paper figures"
os.makedirs(save_dir, exist_ok=True)


# ==========================================
# 7. Function to draw profile figure
# ==========================================
def plot_oxygen_profiles(xscale="log"):
    fig, ax = plt.subplots(figsize=(7.2, 7.8))

    # Soil layer boundaries [m]
    layer_boundaries = [0, -8, -11.5, -17.5, -20.5, -26, -30]

    # Shade alternating layers
    shade_layers = [
        (0, -8),
        (-11.5, -17.5),
        (-20.5, -26)
    ]

    for z_top, z_bot in shade_layers:
        ax.axhspan(z_bot, z_top, color="0.88", alpha=0.8, zorder=0)

    # Horizontal lines at internal layer boundaries
    for z in layer_boundaries[1:-1]:
        ax.axhline(z, color="0.78", linewidth=0.9, zorder=0)

    # --------------------------------------------------
    # Plot profiles: Day 5 to Day 120, every 5 days
    # --------------------------------------------------
    plot_items = list(zip(selected_indices, selected_day_labels))

    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_items)))

    for i, (idx, day_label) in enumerate(plot_items):
        c_plot = conc_all[idx, :]

        if xscale == "log":
            # Log axis cannot plot zero or negative values
            c_plot_safe = np.where(c_plot > 0, c_plot, np.nan)
        else:
            # Linear axis can show zero, but remove negative values if any
            c_plot_safe = np.where(c_plot >= 0, c_plot, np.nan)

        ax.plot(
            c_plot_safe,
            z_m,
            linewidth=2.0,
            color=colors[i],
            label=f"Day {day_label}"
        )

    # ==========================================
    # Axis formatting
    # ==========================================
    ax.set_ylim(-30, 0)
    ax.set_xlabel(r"$c_{O_2}$ (g/cm$^3$)")
    ax.set_ylabel(r"$z$ (m)")
    ax.set_yticks([0, -5, -10, -15, -20, -25, -30])

    if xscale == "log":
        ax.set_xscale("log")
        ax.set_xlim(xmin_log, xmax_log)

        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

        save_path_png = os.path.join(
            save_dir,
            "FigureS5_day5_to_120_every5days_log.png"
        )
        save_path_pdf = os.path.join(
            save_dir,
            "FigureS5_day5_to_120_every5days_log.pdf"
        )

    elif xscale == "linear":
        ax.set_xscale("linear")
        ax.set_xlim(xmin_linear, xmax_linear)

        # Ticks: 0, 2, 4, 6, 8 × 10^-6
        ax.set_xticks([0, 2e-6, 4e-6, 6e-6, 8e-6])

        sf = ScalarFormatter(useMathText=True)
        sf.set_powerlimits((-6, -6))
        ax.xaxis.set_major_formatter(sf)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))

        save_path_png = os.path.join(
            save_dir,
            "FigureS5_day5_to_120_every5days_linear.png"
        )
        save_path_pdf = os.path.join(
            save_dir,
            "FigureS5_day5_to_120_every5days_linear.pdf"
        )

    else:
        raise ValueError("xscale must be either 'log' or 'linear'.")

    # Grid
    ax.grid(True, which="major", linestyle="-", alpha=0.25)

    if xscale == "log":
        ax.grid(True, which="minor", linestyle=":", alpha=0.18)

    # Legend without title
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(save_path_png, dpi=600, bbox_inches="tight")
    plt.savefig(save_path_pdf, bbox_inches="tight")

    print(f"Saved {xscale} figure to:\n{save_path_png}\n{save_path_pdf}")

    plt.show()


# ==========================================
# 8. Draw both figures
# ==========================================
plot_oxygen_profiles(xscale="log")
plot_oxygen_profiles(xscale="linear")