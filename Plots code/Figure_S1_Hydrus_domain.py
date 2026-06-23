import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import matplotlib.patches as patches
from matplotlib.ticker import FixedLocator, MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==========================================
# 1. Plot style
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# ==========================================
# 2. Read HYDRUS Meshtria.txt
# ==========================================
def read_meshtria_strict(filename):
    nodes = []
    elements = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    header_parts = lines[0].strip().split()
    try:
        num_nodes = int(header_parts[1])
        num_elements = int(header_parts[3])
    except (IndexError, ValueError):
        raise ValueError("Cannot parse node/element counts from the first line of Meshtria.txt")

    print(f"Header parsed: {num_nodes} nodes, {num_elements} elements")

    current_line_idx = 1

    # Read nodes
    count_n = 0
    while count_n < num_nodes and current_line_idx < len(lines):
        parts = lines[current_line_idx].strip().split()
        current_line_idx += 1
        if not parts:
            continue
        if len(parts) >= 3:
            try:
                x = float(parts[1])
                z = float(parts[2])
                nodes.append([x, z])
                count_n += 1
            except ValueError:
                continue

    # Read elements
    count_e = 0
    while count_e < num_elements and current_line_idx < len(lines):
        parts = lines[current_line_idx].strip().split()
        current_line_idx += 1
        if not parts:
            continue
        if len(parts) >= 4:
            try:
                n1 = int(parts[1]) - 1
                n2 = int(parts[2]) - 1
                n3 = int(parts[3]) - 1
                elements.append([n1, n2, n3])
                count_e += 1
            except ValueError:
                continue

    return np.array(nodes), np.array(elements)

# ==========================================
# 3. Main plot
# ==========================================
try:
    file_path = "/Users/zhaozhe/Desktop/Project/Meshtria.txt"

    print(f"Reading {file_path} ...")
    nodes_cm, elements = read_meshtria_strict(file_path)
    print(f"Successfully read: {len(nodes_cm)} nodes, {len(elements)} elements")

    # Convert cm -> m
    nodes_m = nodes_cm / 100.0

    # ==========================================
    # 4. Domain settings
    # ==========================================
    x_min, x_max = 0.0, 12.0
    z_top, z_bottom = 0.0, -30.0

    # ==========================================
    # 5. Layer definition by thickness
    # ==========================================
    layer_boundaries = [0.0, -8.0, -11.5, -17.5, -20.5, -26.0, -30.0]

    layer_names = [
        "Layer 1: Sand 1",
        "Layer 2: Loamy Sand",
        "Layer 3: Sand 2",
        "Layer 4: Loam",
        "Layer 5: Sandy Loam",
        "Layer 6: Sand 3"
    ]

    layer_colors = [
        '#dddddd',
        '#cdcdcd',
        '#b8b8b8',
        '#989898',
        '#707070',
        '#505050'
    ]

    # ==========================================
    # 6. Create figure
    # ==========================================
    fig, ax = plt.subplots(figsize=(7.4, 10.8))

    # ==========================================
    # 7. Draw layer background by thickness
    # ==========================================
    for i in range(len(layer_names)):
        z_upper = layer_boundaries[i]
        z_lower = layer_boundaries[i + 1]
        ax.axhspan(
            z_lower,
            z_upper,
            facecolor=layer_colors[i],
            edgecolor='none',
            zorder=1
        )

    # ==========================================
    # 8. Draw HYDRUS mesh
    # ==========================================
    triangulation = mtri.Triangulation(nodes_m[:, 0], nodes_m[:, 1], elements)

    ax.triplot(
        triangulation,
        color='black',
        linewidth=0.23,
        alpha=0.30,
        zorder=2
    )

    # ==========================================
    # 9. Draw only internal layer boundaries
    # ==========================================
    for z in layer_boundaries[1:-1]:
        ax.hlines(
            z,
            x_min,
            x_max,
            colors='black',
            linewidth=1.25,
            alpha=0.95,
            zorder=3
        )

    # ==========================================
    # 10. Dry well polygon (white interior + hatch)
    # ==========================================
    well_points = np.array([
        [0.0,   0.0],
        [0.6,   0.0],
        [0.6,  -3.0],
        [0.43, -3.0],
        [0.43, -16.0],
        [0.0,  -16.0]
    ])

    # White fill to mask background
    poly_well_fill = patches.Polygon(
        well_points,
        closed=True,
        linewidth=0.0,
        edgecolor='none',
        facecolor='white',
        zorder=7
    )
    ax.add_patch(poly_well_fill)

    # Black outline + hatch
    poly_well_hatch = patches.Polygon(
        well_points,
        closed=True,
        linewidth=2.0,
        edgecolor='black',
        facecolor='none',
        hatch='//',
        zorder=8
    )
    ax.add_patch(poly_well_hatch)

    ax.annotate(
        'Dry Well',
        xy=(0.43, -6.2),
        xytext=(4.0, -5.9),
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            lw=1.8,
            connectionstyle="arc3,rad=-0.10"
        ),
        color='black',
        fontsize=12,
        va='center',
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="black",
            alpha=0.95,
            lw=0.8
        ),
        zorder=12
    )

    # ==========================================
    # 11. Boundary conditions
    # ==========================================
    bc1_color = '#008000'   # green
    bc2_color = '#0000FF'   # blue
    bc3_color = '#D60000'   # red
    bc4_color = 'black'     # remaining boundaries

    bc_lw = 4.0
    bc4_lw = 2.8

    # Boundary 1 (green): (0.6,0) -> (12,0)
    ax.plot(
        [0.6, 12.0], [0.0, 0.0],
        color=bc1_color,
        lw=bc_lw,
        solid_capstyle='butt',
        zorder=20,
        clip_on=False
    )

    # Boundary 2 (blue):
    ax.plot(
        [0.43, 0.43], [-3.0, -16.0],
        color=bc2_color,
        lw=bc_lw,
        solid_capstyle='butt',
        zorder=20
    )
    ax.plot(
        [0.43, 0.0], [-16.0, -16.0],
        color=bc2_color,
        lw=bc_lw,
        solid_capstyle='butt',
        zorder=20
    )

    # Boundary 3 (red): (0,-30) -> (12,-30)
    ax.plot(
        [0.0, 12.0], [-30.0, -30.0],
        color=bc3_color,
        lw=bc_lw,
        solid_capstyle='butt',
        zorder=20,
        clip_on=False
    )

    # Boundary 4 (black): remaining boundaries
    ax.plot(
        [12.0, 12.0], [0.0, -30.0],
        color=bc4_color, lw=bc4_lw, solid_capstyle='butt', zorder=18
    )
    ax.plot(
        [0.0, 0.0], [-16.0, -30.0],
        color=bc4_color, lw=bc4_lw, solid_capstyle='butt', zorder=18
    )
    ax.plot(
        [0.0, 0.6], [0.0, 0.0],
        color=bc4_color, lw=bc4_lw, solid_capstyle='butt', zorder=18
    )
    ax.plot(
        [0.6, 0.6], [0.0, -3.0],
        color=bc4_color, lw=bc4_lw, solid_capstyle='butt', zorder=18
    )
    ax.plot(
        [0.6, 0.43], [-3.0, -3.0],
        color=bc4_color, lw=bc4_lw, solid_capstyle='butt', zorder=18
    )
    ax.plot(
        [0.0, 0.0], [0.0, -16.0],
        color=bc4_color, lw=bc4_lw, solid_capstyle='butt', zorder=18
    )

    # ==========================================
    # 12. Layer labels
    # ==========================================
    for i, label in enumerate(layer_names):
        z_mid = (layer_boundaries[i] + layer_boundaries[i + 1]) / 2.0
        ax.text(
            11.65,
            z_mid,
            label,
            ha='right',
            va='center',
            fontsize=12,
            color='white',
            zorder=10,
            bbox=dict(
                facecolor='black',
                alpha=0.36,
                edgecolor='none',
                pad=2.5
            )
        )

    # ==========================================
    # 13. Axes limits and labels
    # ==========================================
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_bottom, z_top)

    # Remove axis frame, keep ticks
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel('X Position [m]', fontsize=16)
    ax.set_ylabel('Depth [m]', fontsize=16)

    # ==========================================
    # 14. Major and minor ticks
    # ==========================================
    ax.set_xticks([0, 2.5, 5.0, 7.5, 10.0])
    ax.set_yticks([0, -5, -10, -15, -20, -25, -30])

    ax.xaxis.set_minor_locator(MultipleLocator(1.0))

    y_major = [0, -5, -10, -15, -20, -25, -30]
    y_minor_ticks = [z for z in layer_boundaries if z not in y_major]
    ax.yaxis.set_minor_locator(FixedLocator(y_minor_ticks))

    ax.tick_params(axis='both', which='major', length=6, width=1.0)
    ax.tick_params(axis='both', which='minor', length=4, width=0.8)

    # ==========================================
    # 15. Legend outside the plot (right upper)
    # ==========================================
    legend_handles = [
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Dry Well'),
        Line2D([0], [0], color=bc1_color, lw=bc_lw, label='Flow: Atmospheric / O$_2$: Volatile'),
        Line2D([0], [0], color=bc2_color, lw=bc_lw, label='Flow: Var Head / O$_2$: Third-Type'),
        Line2D([0], [0], color=bc3_color, lw=bc_lw, label='Flow: Free Drainage / O$_2$: Third-Type'),
        Line2D([0], [0], color=bc4_color, lw=bc4_lw, label='No Flux')
    ]

    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=11,
        frameon=True,
        framealpha=0.95
    )

    # ==========================================
    # 16. Layout
    # ==========================================
    plt.subplots_adjust(left=0.14, right=0.78, top=0.97, bottom=0.10)

    # ==========================================
    # 17. Save figure
    # ==========================================
    save_path_png = '/Users/zhaozhe/Desktop/Project/Figure_Numerical_Domain_BC.png'
    save_path_pdf = '/Users/zhaozhe/Desktop/Project/Figure_Numerical_Domain_BC.pdf'

    plt.savefig(save_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')

    print(f"Figure saved to:\n{save_path_png}\n{save_path_pdf}")

    plt.show()

except Exception as e:
    import traceback
    traceback.print_exc()