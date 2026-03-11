"""
Thailand-Cambodia Border Conflict Risk Animation (Geographic Map Version)

Animates monthly conflict probability across 4 zones from 2008-2025,
plotted over a geographic map of the Thailand-Cambodia border region.
Pauses for 2 seconds on conflict months.

REQUIRES:
    pip install pandas numpy matplotlib pillow

INPUT:
    monthly_probs.csv (precomputed by bayesian_network.py)

OUTPUT:
    conflict_risk_animation.gif
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
import matplotlib.patheffects as pe


# =============================================================
# CONFIG
# =============================================================

MONTHLY_PROBS_PATH = "../data/clean/monthly_probs.csv"
OUTPUT_PATH = "../output/conflict_risk_animation.gif"
PAUSE_FRAMES = 16  # At 8fps = 2 second pause


# =============================================================
# GEOGRAPHIC DATA (real coordinates)
# =============================================================

# Map extent (lon, lat)
LON_MIN, LON_MAX = 101.0, 106.5
LAT_MIN, LAT_MAX = 10.5, 15.5

# Thailand outline (simplified, border region only)
THAILAND_COORDS = [
    (101.0, 15.5), (102.0, 15.5), (103.0, 15.3), (104.0, 15.0),
    (105.0, 15.2), (105.6, 14.5),
    (105.1, 14.4), (104.7, 14.4), (104.1, 14.3), (103.5, 14.3),
    (103.0, 14.15), (102.6, 14.1), (102.2, 14.0),
    (102.4, 13.7), (102.5, 13.4), (102.6, 13.0),
    (102.4, 12.6), (102.3, 12.3), (102.1, 12.0),
    (101.8, 11.8), (101.5, 11.5), (101.0, 11.5),
    (101.0, 15.5),
]

# Cambodia outline (simplified, border region only)
CAMBODIA_COORDS = [
    (105.6, 14.5), (106.5, 14.5), (106.5, 13.5), (106.5, 12.5),
    (106.5, 11.5), (106.0, 10.5), (105.0, 10.5),
    (104.0, 10.5), (103.5, 10.8), (103.0, 11.0),
    (102.9, 11.5), (102.8, 11.8),
    (102.5, 12.0), (102.4, 12.3), (102.5, 12.6),
    (102.6, 13.0), (102.5, 13.4), (102.4, 13.7),
    (102.2, 14.0), (102.6, 14.1), (103.0, 14.15),
    (103.5, 14.3), (104.1, 14.3), (104.7, 14.4),
    (105.1, 14.4), (105.6, 14.5),
]

# Gulf of Thailand water
GULF_COORDS = [
    (101.0, 11.5), (101.5, 11.5), (101.8, 11.8),
    (102.1, 12.0), (102.3, 12.3), (102.5, 12.0),
    (102.8, 11.8), (102.9, 11.5), (103.0, 11.0),
    (103.5, 10.8), (104.0, 10.5), (105.0, 10.5),
    (106.0, 10.5), (106.5, 10.5),
    (101.0, 10.5), (101.0, 11.5),
]

# The actual border line (more precise points)
BORDER_LINE = [
    (105.6, 14.5), (105.1, 14.4), (104.7, 14.39),
    (104.1, 14.35), (103.5, 14.32), (103.27, 14.33),
    (103.0, 14.15), (102.6, 14.08), (102.35, 13.95),
    (102.5, 13.65), (102.55, 13.4), (102.6, 13.0),
    (102.5, 12.6), (102.4, 12.3), (102.3, 12.0),
    (102.5, 11.8),
]

# Zone centers (real lon/lat)
ZONE_GEO = {
    "Zone_1": (104.9, 14.4),
    "Zone_2": (103.3, 14.25),
    "Zone_3": (102.55, 13.6),
    "Zone_4": (102.5, 11.9),
}

# Key locations to mark on the map
KEY_LOCATIONS = {
    "Preah Vihear": (104.68, 14.39),
    "Emerald Triangle": (105.14, 14.45),
    "Ta Muen Thom": (103.27, 14.33),
    "O'Smach": (103.69, 14.42),
    "Poipet": (102.58, 13.66),
    "Ban Chamrak": (102.70, 12.22),
    "Ko Kut": (102.56, 11.67),
}

ZONE_NAMES = {
    "Zone_1": "Zone 1: Preah Vihear",
    "Zone_2": "Zone 2: Ta Muen Thom",
    "Zone_3": "Zone 3: Poipet",
    "Zone_4": "Zone 4: Trat/Maritime",
}

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# =============================================================
# HELPERS
# =============================================================

def geo_to_plot(lon, lat):
    """Convert geographic coordinates to plot coordinates (0-1)."""
    x = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    y = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    return x, y


# =============================================================
# LOAD DATA
# =============================================================

df = pd.read_csv(MONTHLY_PROBS_PATH)
df["date_str"] = df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2)
dates = sorted(df["date_str"].unique())

# Build frame list with pauses on conflict months
frame_dates = []
for date in dates:
    frame_data = df[df["date_str"] == date]
    has_conflict = frame_data["Is_Conflict"].any()
    frame_dates.append(date)
    if has_conflict:
        for _ in range(PAUSE_FRAMES):
            frame_dates.append(date)

print(f"Original frames: {len(dates)}")
print(f"With pauses: {len(frame_dates)}")


# =============================================================
# COLORMAP & FIGURE
# =============================================================

colors_risk = ["#27ae60", "#f1c40f", "#e74c3c"]
cmap = LinearSegmentedColormap.from_list("risk", colors_risk, N=256)

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0a0a1a")
fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08)


# =============================================================
# ANIMATION FUNCTION
# =============================================================

def animate(i):
    ax.clear()
    ax.set_facecolor("#0a0a1a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    date = frame_dates[i]
    frame = df[df["date_str"] == date]
    year = int(frame.iloc[0]["Year"])
    month = int(frame.iloc[0]["Month"])
    conflict_rows = frame[frame["Is_Conflict"]]
    has_conflict = len(conflict_rows) > 0

    # ---- Draw Thailand ----
    thai_plot = [geo_to_plot(lon, lat) for lon, lat in THAILAND_COORDS]
    thai_poly = Polygon(thai_plot, closed=True, facecolor="#1a2a1a",
                        edgecolor="#334433", linewidth=1, alpha=0.8)
    ax.add_patch(thai_poly)

    # ---- Draw Cambodia ----
    camb_plot = [geo_to_plot(lon, lat) for lon, lat in CAMBODIA_COORDS]
    camb_poly = Polygon(camb_plot, closed=True, facecolor="#1a1a2a",
                        edgecolor="#333344", linewidth=1, alpha=0.8)
    ax.add_patch(camb_poly)

    # ---- Draw Gulf of Thailand ----
    gulf_plot = [geo_to_plot(lon, lat) for lon, lat in GULF_COORDS]
    gulf_poly = Polygon(gulf_plot, closed=True, facecolor="#0a1525",
                        edgecolor="none", alpha=0.9)
    ax.add_patch(gulf_poly)
    gx, gy = geo_to_plot(103.5, 11.2)
    ax.text(gx, gy, "Gulf of Thailand", fontsize=9, color="#1a3050",
            ha="center", style="italic", family="sans-serif")

    # ---- Draw border line ----
    bx = [geo_to_plot(lon, lat)[0] for lon, lat in BORDER_LINE]
    by = [geo_to_plot(lon, lat)[1] for lon, lat in BORDER_LINE]
    ax.plot(bx, by, color="#ff6666", linewidth=2, alpha=0.6, linestyle="-")

    # ---- Country labels ----
    tx, ty = geo_to_plot(101.8, 15.0)
    ax.text(tx, ty, "THAILAND", fontsize=16, color="#3a5a3a",
            ha="center", fontweight="bold", family="sans-serif", alpha=0.7)
    cx, cy = geo_to_plot(105.5, 12.5)
    ax.text(cx, cy, "CAMBODIA", fontsize=16, color="#3a3a5a",
            ha="center", fontweight="bold", family="sans-serif", alpha=0.7)

    # ---- Key location dots ----
    for name, (lon, lat) in KEY_LOCATIONS.items():
        px, py = geo_to_plot(lon, lat)
        ax.plot(px, py, "o", color="#666688", markersize=3, alpha=0.5)
        ax.text(px + 0.01, py - 0.015, name, fontsize=5, color="#666688",
                alpha=0.6, family="sans-serif")

    # ---- ZONE CIRCLES ----
    for _, row in frame.iterrows():
        zone = row["Zone"]
        p = row["P_Conflict"]
        lon, lat = ZONE_GEO[zone]
        x, y = geo_to_plot(lon, lat)
        color = cmap(p)
        r = 0.06

        # Glow for high risk
        if p > 0.5:
            glow = plt.Circle((x, y), r * 1.6, color=color, alpha=0.12)
            ax.add_patch(glow)
            glow2 = plt.Circle((x, y), r * 1.35, color=color, alpha=0.18)
            ax.add_patch(glow2)

        # Main circle
        circle = plt.Circle((x, y), r, color=color, alpha=0.85,
                            ec="white", linewidth=2)
        ax.add_patch(circle)

        # Conflict flash rings
        if row["Is_Conflict"]:
            flash = plt.Circle((x, y), r * 1.3, fill=False,
                              ec="#ff0000", linewidth=3)
            ax.add_patch(flash)
            flash2 = plt.Circle((x, y), r * 1.6, fill=False,
                               ec="#ff0000", linewidth=1.5,
                               linestyle="--", alpha=0.5)
            ax.add_patch(flash2)

        # Percentage
        pct_color = "white" if p <= 0.5 else "#ffcccc"
        ax.text(x, y + 0.005, f"{p:.0%}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=pct_color,
                family="sans-serif",
                path_effects=[pe.withStroke(linewidth=3, foreground="black")])

        # Zone name below circle
        ax.text(x, y - r - 0.02, ZONE_NAMES[zone], ha="center", va="top",
                fontsize=7, color="white", family="sans-serif",
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # ---- TITLE ----
    ax.text(0.50, 0.99, "Thailand-Cambodia Border Conflict Risk",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=18, fontweight="bold", color="white", family="sans-serif")
    ax.text(0.50, 0.95, f"{MONTH_NAMES[month]} {year}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=24, fontweight="bold", color="#3498db", family="sans-serif")

    # ---- LEGEND (top left) ----
    for j, (label, c) in enumerate([("Low Risk (<20%)", "#27ae60"),
                                     ("Medium (20-50%)", "#f1c40f"),
                                     ("High Risk (>50%)", "#e74c3c")]):
        ly = 0.99 - j * 0.03
        ax.add_patch(plt.Rectangle((0.005, ly - 0.008), 0.015, 0.015,
                                    transform=ax.transAxes, facecolor=c,
                                    ec="white", linewidth=0.5))
        ax.text(0.025, ly, label, transform=ax.transAxes,
                fontsize=7, color="white", va="center", family="sans-serif")

    # ---- STATUS (bottom left, above timeline) ----
    gw = frame.iloc[0]["Gov_Weakness"]
    tl = frame.iloc[0]["Trigger_Level"]
    season = frame.iloc[0]["Season"]
    gw_c = {"Stable": "#27ae60", "Fragile": "#f1c40f",
            "Collapsed": "#e74c3c"}.get(gw, "white")
    tl_c = {"Low": "#27ae60", "Medium": "#f1c40f",
            "High": "#e74c3c"}.get(tl, "white")
    s_c = "#f39c12" if season == "Dry" else "#3498db"

    ax.text(0.01, 0.18, "STATUS", transform=ax.transAxes,
            fontsize=8, color="#888888", family="sans-serif", fontweight="bold")
    ax.text(0.01, 0.14, f"Gov Weakness: {gw}", transform=ax.transAxes,
            fontsize=9, color=gw_c, family="sans-serif", fontweight="bold")
    ax.text(0.01, 0.10, f"Trigger Level: {tl}", transform=ax.transAxes,
            fontsize=9, color=tl_c, family="sans-serif", fontweight="bold")
    ax.text(0.01, 0.06, f"Season: {season}", transform=ax.transAxes,
            fontsize=9, color=s_c, family="sans-serif", fontweight="bold")

    # ---- CONFLICT BANNER (right side, above timeline) ----
    if has_conflict:
        cname = conflict_rows.iloc[0]["Conflict_Name"]
        ax.text(0.99, 0.12, f"ACTIVE CONFLICT:  {cname}",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=12, fontweight="bold", color="#ff4444",
                family="sans-serif",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a0000",
                          ec="#ff4444", linewidth=2.5, alpha=0.95))

    # ---- TIMELINE BAR ----
    date_idx = dates.index(date)
    progress = date_idx / max(len(dates) - 1, 1)
    ax.add_patch(plt.Rectangle((0.10, 0.015), 0.80, 0.012,
                                transform=ax.transAxes,
                                facecolor="#181828", ec="#333355",
                                linewidth=0.5))
    ax.add_patch(plt.Rectangle((0.10, 0.015), 0.80 * progress, 0.012,
                                transform=ax.transAxes,
                                facecolor="#3498db", ec="none"))
    for yr in range(2008, 2026, 2):
        xp = 0.10 + 0.80 * ((yr - 2008) / 17)
        ax.text(xp, 0.035, str(yr), transform=ax.transAxes,
                fontsize=6, color="#555577", ha="center", family="sans-serif")


# =============================================================
# RENDER
# =============================================================

total = len(frame_dates)
print(f"Rendering {total} frames...")

anim = FuncAnimation(fig, animate, frames=total, interval=125, repeat=True)

print(f"Saving to {OUTPUT_PATH}...")
anim.save(OUTPUT_PATH, writer=PillowWriter(fps=8))
print("Done!")
plt.close(fig)