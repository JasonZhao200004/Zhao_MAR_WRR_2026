import os
import re
import numpy as np
import matplotlib.pyplot as plt
import imageio
import matplotlib.tri as mtri


# Physical parameter
# Velocity is in cm/h, and mesh coordinates in Meshtria_copy.txt are in cm.
DECAY_LAMBDA = 0.0002  # [1/h] first-order decay coefficient


# Global font settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
})


def read_velocity_data(filename):
    """
    Read the HYDRUS velocity output file and return a list of dictionaries
    containing time, x-component velocity, y-component velocity, and velocity magnitude.
    """
    velocity_blocks = []
    current_block = {}
    current_component = None
    component_lines = []
    prev_time = None

    def parse_component(lines):
        vals = []
        for ln in lines:
            if ln.strip():
                vals.extend([float(x) for x in ln.split()])
        return np.array(vals)

    header_re = re.compile(
        r"Time\s*=\s*([\d\.Ee+-]+)\s+Velocity\s*-\s*(first|second)\s*component",
        re.IGNORECASE
    )

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue

        m = header_re.match(ln)
        if m:
            if current_component and component_lines:
                arr = parse_component(component_lines)
                if current_component == 'vx':
                    current_block['vx'] = arr
                else:
                    current_block['vy'] = arr
                component_lines = []

            time_val = float(m.group(1))
            comp = m.group(2).lower()

            if comp == 'first':
                if 'vx' in current_block and 'vy' in current_block:
                    current_block['v'] = np.sqrt(
                        current_block['vx']**2 + current_block['vy']**2
                    )
                    current_block['time'] = prev_time
                    velocity_blocks.append(current_block)
                    current_block = {}

                prev_time = time_val
                current_component = 'vx'
            else:
                current_component = 'vy'

            continue

        if current_component:
            component_lines.append(ln)

    if current_component and component_lines:
        arr = parse_component(component_lines)
        if current_component == 'vx':
            current_block['vx'] = arr
        else:
            current_block['vy'] = arr

    if 'vx' in current_block and 'vy' in current_block:
        current_block['v'] = np.sqrt(
            current_block['vx']**2 + current_block['vy']**2
        )
        current_block['time'] = prev_time
        velocity_blocks.append(current_block)

    return velocity_blocks


def compute_node_dx_cm(triang, x_cm, y_cm):
    """
    Estimate a characteristic length scale for each mesh node.

    The characteristic length is calculated as the mean length of all triangle
    edges connected to the node.
    """
    n_points = x_cm.size
    edge_lengths = [[] for _ in range(n_points)]

    for tri in triang.triangles:
        i, j, k = tri
        for a, b in ((i, j), (j, k), (k, i)):
            dx = x_cm[a] - x_cm[b]
            dy = y_cm[a] - y_cm[b]
            length = np.hypot(dx, dy)
            edge_lengths[a].append(length)
            edge_lengths[b].append(length)

    dx_node = np.zeros(n_points)
    for idx in range(n_points):
        if edge_lengths[idx]:
            dx_node[idx] = np.mean(edge_lengths[idx])
        else:
            dx_node[idx] = 1e-6

    return dx_node


def create_velocity_and_logda_gif(
    velocity_blocks,
    grid_file="Meshtria_copy.txt",
    vel_gif_name="Velocity.gif",
    logda_gif_name="logDa.gif",
    base_width=8,
    eps=1e-12,
    da_threshold=1.0
):
    """
    Generate two GIF files:
    1. Velocity magnitude field.
    2. log10(Damköhler number) field with the Da threshold contour.

    The local Damköhler number is defined as:
        Da_i = lambda * (dx_i / |v_i|)
    """
    grid = np.loadtxt(grid_file)
    x_cm = grid[:, 1]
    y_cm = grid[:, 2]

    x = x_cm / 100.0
    y = y_cm / 100.0
    triang = mtri.Triangulation(x, y)

    dx_node_cm = compute_node_dx_cm(triang, x_cm, y_cm)
    print(
        f"Node dx range: {dx_node_cm.min():.3f} cm "
        f"to {dx_node_cm.max():.3f} cm"
    )

    dx_plot = x.max() - x.min()
    dy_plot = y.max() - y.min()
    aspect = dy_plot / dx_plot

    logv_list = []
    logda_list = []
    times = []

    for blk in velocity_blocks:
        v = blk['v']
        t = blk['time']

        if v.size != x.size:
            print(
                f"Skipping t={t} h: velocity nodes ({v.size}) "
                f"do not match mesh nodes ({x.size})"
            )
            continue

        logv = np.log10(v + eps)
        logv_list.append(logv)
        times.append(t)

        v_abs = np.abs(v) + eps
        tau_adv = dx_node_cm / v_abs
        da = DECAY_LAMBDA * tau_adv
        log_da = np.log10(da + eps)
        logda_list.append(log_da)

    if not logv_list:
        print("No valid time steps available for plotting.")
        return

    logv_arr = np.array(logv_list)
    logda_arr = np.array(logda_list)
    times = np.array(times)

    vmin = logv_arr.min()
    vmax = logv_arr.max()
    print(f"Velocity GIF log10(v) range: {vmin:.3f} to {vmax:.3f}")

    logda_min = logda_arr.min()
    logda_max = logda_arr.max()
    print(f"logDa GIF log10(Da) range: {logda_min:.3f} to {logda_max:.3f}")

    logda_min_plot = logda_min
    logda_max_plot = logda_max

    x1 = np.array([0, 0, 43, 43]) / 100.0
    z1 = -np.array([0, 1600, 1600, 0]) / 100.0
    x2 = np.array([0, 0, 60, 60]) / 100.0
    z2 = -np.array([0, 360, 360, 0]) / 100.0

    vel_frames = []
    for idx, (logv, t) in enumerate(zip(logv_arr, times)):
        fig, ax = plt.subplots(figsize=(base_width, base_width * aspect))
        ax.set_aspect('equal', adjustable='box')

        cf = ax.tricontourf(
            triang,
            logv,
            levels=51,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
        )
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("log10(Velocity) (cm/h)")
        cbar.ax.tick_params(labelsize=14)

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"log10(Velocity) at {t:.0f} h")

        ax.fill(x1, z1, fc='gray', alpha=1)
        ax.fill(x2, z2, fc='gray', alpha=1)

        frame_name = f"vel_frame_{idx:03d}.png"
        fig.savefig(frame_name, dpi=100)
        plt.close(fig)

        vel_frames.append(frame_name)
        print(f"Velocity frame saved: {frame_name} (t={t} h)")

    with imageio.get_writer(vel_gif_name, mode='I', duration=0.5) as writer:
        for frame_name in vel_frames:
            writer.append_data(imageio.v2.imread(frame_name))

    print(f"Velocity GIF saved: {vel_gif_name}")

    for frame_name in vel_frames:
        os.remove(frame_name)

    da_frames = []
    logda_level = np.log10(da_threshold)

    for idx, (log_da, t) in enumerate(zip(logda_arr, times)):
        fig, ax = plt.subplots(figsize=(base_width, base_width * aspect))
        ax.set_aspect('equal', adjustable='box')

        cf = ax.tricontourf(
            triang,
            log_da,
            levels=51,
            cmap='plasma',
            vmin=logda_min_plot,
            vmax=logda_max_plot
        )
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("log10(Damköhler number) (-)")
        cbar.ax.tick_params(labelsize=14)

        cs = ax.tricontour(
            triang,
            log_da,
            levels=[logda_level],
            colors='red',
            linewidths=1.5
        )
        cs.collections[0].set_label(f"Da = {da_threshold:g}")
        ax.legend(loc='upper right')

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"log10(Damköhler) at {t:.0f} h")

        ax.fill(x1, z1, fc='gray', alpha=1)
        ax.fill(x2, z2, fc='gray', alpha=1)

        frame_name = f"logda_frame_{idx:03d}.png"
        fig.savefig(frame_name, dpi=100)
        plt.close(fig)

        da_frames.append(frame_name)
        print(f"logDa frame saved: {frame_name} (t={t} h)")

    with imageio.get_writer(logda_gif_name, mode='I', duration=0.5) as writer:
        for frame_name in da_frames:
            writer.append_data(imageio.v2.imread(frame_name))

    print(f"logDa GIF saved: {logda_gif_name}")

    for frame_name in da_frames:
        os.remove(frame_name)

    print("Temporary frame files removed.")


if __name__ == "__main__":
    velocity_filename = "/Users/zhaozhe/Desktop/Project/Thesis/V.txt"

    blocks = read_velocity_data(velocity_filename)
    print(f"Number of velocity time steps read: {len(blocks)}")

    create_velocity_and_logda_gif(
        blocks,
        grid_file="/Users/zhaozhe/Desktop/Project/Hydrus-2D/Meshtria_copy.txt",
        vel_gif_name="/Users/zhaozhe/Desktop/Project/Thesis/Velocity.gif",
        logda_gif_name="/Users/zhaozhe/Desktop/Project/Thesis/logDa.gif",
        base_width=8,
        eps=1e-12,
        da_threshold=1.0
    )