import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import matplotlib.ticker as mticker


paths = {
    '4-6': {
        'wetting': '/Users/zhaozhe/Desktop/Project/water balance/Cum_Q 4-6.out',
        'draining': '/Users/zhaozhe/Desktop/Project/water balance/Cum_Q 4-6 draining 6000.out'
    },
    '6-4': {
        'wetting': '/Users/zhaozhe/Desktop/Project/water balance/Cum_Q 6-4.out',
        'draining': '/Users/zhaozhe/Desktop/Project/water balance/Cum_Q 6-4 draining 6000.out'
    },
    '9-1': {
        'wetting': '/Users/zhaozhe/Desktop/Project/water balance/Cum_Q 9-1.out',
        'draining': '/Users/zhaozhe/Desktop/Project/water balance/Cum_Q 9-1 draining 6000.out'
    }
}


def parse_cumq(file_path):
    """Parse the HYDRUS Cum_Q.out file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r') as f:
        lines = f.readlines()

    h_idx = -1
    for i, line in enumerate(lines):
        if "Time" in line and "CumQ3" in line:
            h_idx = i
            break

    if h_idx == -1:
        return pd.DataFrame()

    headers = lines[h_idx].split()
    data = []

    for line in lines[h_idx + 2:]:
        parts = line.split()
        if len(parts) == len(headers):
            try:
                data.append({h: float(p) for h, p in zip(headers, parts)})
            except:
                continue

    return pd.DataFrame(data)


def get_processed_data(sch, split_t, p_dict):
    """Merge wetting and draining stages and convert units."""
    q_wet = parse_cumq(p_dict['wetting'])
    q_dry = parse_cumq(p_dict['draining'])

    if q_wet.empty or q_dry.empty:
        return pd.DataFrame()

    q_wet = q_wet[q_wet['Time'] <= split_t].copy()
    q_wet['Inj'] = -q_wet['CumQ3']
    q_wet['Rec'] = q_wet['CumQ6']

    last_inj = q_wet['Inj'].iloc[-1]
    last_rec = q_wet['Rec'].iloc[-1]

    q_dry['Time'] += split_t
    q_dry['Inj'] = last_inj - q_dry['CumQ3']
    q_dry['Rec'] = last_rec + q_dry['CumQ6']

    df = pd.concat(
        [q_wet[['Time', 'Inj', 'Rec']], q_dry[['Time', 'Inj', 'Rec']]],
        ignore_index=True
    )

    df['Inj'] *= 1e-6
    df['Rec'] *= 1e-6
    df['Sto'] = df['Inj'] - df['Rec']
    df['Efficiency'] = np.where(df['Inj'] > 1e-6, df['Rec'] / df['Inj'], 0)

    return df


splits = {'4-6': 576, '6-4': 624, '9-1': 696}
schedules = ['4-6', '6-4', '9-1']
sch_labels = {s: f"{s} wetting-draining (d)" for s in schedules}

data_map = {s: get_processed_data(s, splits[s], paths[s]) for s in schedules}


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 10
})

colors = {
    "4-6": "#1f77b4",
    "6-4": "#ff7f0e",
    "9-1": "#e377c2"
}


fig6, (ax_a, ax_b) = plt.subplots(
    2, 1,
    figsize=(8.8, 9.2),
    dpi=150,
    sharex=True
)


for s in schedules:
    df = data_map[s]
    if df.empty:
        continue

    t = df['Time'] / 24.0
    mask = (t <= 100)
    tm = t[mask]
    d = df.loc[mask]
    c = colors[s]

    ax_a.plot(tm, d['Inj'], color=c, lw=2.2, ls='-')
    ax_a.plot(tm, d['Rec'], color=c, lw=2.2, ls='--')
    ax_a.plot(tm, d['Sto'], color=c, lw=2.2, ls=':')

ax_a.set_xlim(0, 100)
ax_a.set_ylabel(r'Water volume, $V_i$ (m$^3$)')
ax_a.grid(True, ls='--', lw=0.5, alpha=0.35)
ax_a.text(
    -0.10, 1.03, '(a)',
    transform=ax_a.transAxes,
    fontsize=14,
    fontweight='bold',
    va='bottom'
)


for s in schedules:
    df = data_map[s]
    if df.empty:
        continue

    t = df['Time'] / 24.0
    mask = (t <= 100)
    tm = t[mask]
    c = colors[s]

    ax_b.plot(tm, df.loc[mask, 'Efficiency'], color=c, lw=2.2, ls='--')

ax_b.set_xlim(0, 100)
ax_b.set_ylim(0, 1.45)
ax_b.set_xlabel('Time (d)')
ax_b.set_ylabel(r'Efficiency, $\eta$ (-)')
ax_b.grid(True, ls='--', lw=0.5, alpha=0.35)
ax_b.text(
    -0.10, 1.03, '(b)',
    transform=ax_b.transAxes,
    fontsize=14,
    fontweight='bold',
    va='bottom'
)


mar_end_day = 30

for ax in (ax_a, ax_b):
    ax.axvspan(
        mar_end_day, 100,
        color='0.90',
        alpha=0.45,
        zorder=0
    )
    ax.axvline(
        mar_end_day,
        color='0.25',
        linestyle='--',
        linewidth=1.4,
        zorder=4
    )


schedule_handles = [
    Line2D([0], [0], color=colors[s], lw=2.2, ls='-', label=sch_labels[s])
    for s in schedules
]

component_handles_a = [
    Line2D([0], [0], color='black', lw=2.2, ls='-', label='Injection'),
    Line2D([0], [0], color='black', lw=2.2, ls='--', label='Recharge'),
    Line2D([0], [0], color='black', lw=2.2, ls=':', label='Storage')
]

component_handles_b = [
    Line2D([0], [0], color='black', lw=2.2, ls='--', label='Efficiency')
]

blank = Line2D([], [], linestyle='none', label='')

handles_a = (
    [Line2D([], [], linestyle='none', label='Schedule:')]
    + schedule_handles
    + [blank]
    + [Line2D([], [], linestyle='none', label='Component:')]
    + component_handles_a
)

handles_b = (
    [Line2D([], [], linestyle='none', label='Schedule:')]
    + schedule_handles
    + [blank]
    + [Line2D([], [], linestyle='none', label='Component:')]
    + component_handles_b
)


plt.subplots_adjust(
    left=0.12,
    right=0.72,
    bottom=0.08,
    top=0.96,
    hspace=0.32
)

leg_a = ax_a.legend(
    handles=handles_a,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.00),
    frameon=True,
    borderpad=0.8,
    labelspacing=0.6,
    handlelength=2.8
)

leg_b = ax_b.legend(
    handles=handles_b,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.00),
    frameon=True,
    borderpad=0.8,
    labelspacing=0.6,
    handlelength=2.8
)

for leg in (leg_a, leg_b):
    for h, txt in zip(leg.legendHandles, leg.texts):
        if txt.get_text() in ("Schedule:", "Component:"):
            h.set_visible(False)


plt.savefig(
    '/Users/zhaozhe/Desktop/Project/water balance/Fig6_Journal_Standard_vertical.png',
    dpi=300,
    bbox_inches='tight',
    bbox_extra_artists=(leg_a, leg_b),
    pad_inches=0.35
)


fig7, ax_bar = plt.subplots(figsize=(13, 9), constrained_layout=True)

bar_results = []

for s in schedules:
    df = data_map[s]
    if df.empty:
        continue

    idx = (df['Time'] - 720).abs().idxmin()
    row = df.loc[idx]

    bar_results.append({
        'Schedule': s,
        'Recharge': float(row['Rec']),
        'Storage': float(row['Sto']),
        'Total': float(row['Inj'])
    })

df_bar = pd.DataFrame(bar_results)

rec_c = '#1f77b4'
sto_c = '#ff7f0e'

x = np.arange(len(df_bar))
width = 0.55

ax_bar.bar(
    x,
    df_bar['Recharge'],
    width=width,
    color=rec_c,
    edgecolor='black',
    linewidth=1.5,
    label=r'Recharge ($V_{recharge}$)'
)

ax_bar.bar(
    x,
    df_bar['Storage'],
    width=width,
    bottom=df_bar['Recharge'],
    color=sto_c,
    edgecolor='black',
    linewidth=1.5,
    label=r'Storage ($V_{storage}$)'
)

xtxt = [f"{s} days" for s in df_bar['Schedule']]
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(xtxt, fontsize=26, rotation=0, ha='center')
ax_bar.tick_params(axis='x', pad=10)

ax_bar.set_ylabel('Water volume, $V_i$ (m$^3$)', fontsize=26)
ax_bar.grid(axis='y', linestyle='--', linewidth=1.1, alpha=0.45)
ax_bar.set_axisbelow(True)
ax_bar.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax_bar.tick_params(axis='y', labelsize=18)

ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_linewidth(2.2)
ax_bar.spines['bottom'].set_linewidth(2.2)

ax_bar.legend(loc='upper left', frameon=False, fontsize=22)

ymax = df_bar['Total'].max()
ax_bar.set_ylim(0, ymax * 1.12)

for i in range(len(df_bar)):
    rec = df_bar.loc[i, 'Recharge']
    sto = df_bar.loc[i, 'Storage']
    tot = df_bar.loc[i, 'Total']

    ax_bar.text(
        x[i],
        rec * 0.50,
        f'{rec:,.0f}',
        ha='center',
        va='center',
        color='white',
        fontsize=26,
        fontweight='bold'
    )

    ax_bar.text(
        x[i],
        rec + sto * 0.50,
        f'{sto:,.0f}',
        ha='center',
        va='center',
        color='white',
        fontsize=26,
        fontweight='bold'
    )

    ax_bar.text(
        x[i],
        tot + ymax * 0.03,
        f'{tot:,.0f}',
        ha='center',
        va='bottom',
        color='black',
        fontsize=26,
        fontweight='bold'
    )

plt.subplots_adjust(left=0.1, right=0.75, bottom=0.15)

plt.savefig(
    '/Users/zhaozhe/Desktop/Project/water balance/Fig7_Stacked_Bar.png',
    dpi=300
)

plt.show()