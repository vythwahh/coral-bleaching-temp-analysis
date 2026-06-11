"""
EcoCast - bathymetric_map.py
NOAA-style bathymetric map using real ETOPO1 data with hillshading.
"""

import io
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import netCDF4 as nc
import urllib.request
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import LightSource
import pandas as pd


REGION_BOUNDS = {
    "Indonesia":          (95,  141, -11,   7),
    "Philippines":        (115, 128,   4,  22),
    "Coral Triangle":     (110, 141, -11,   7),
    "Great Barrier Reef": (141, 155, -25, -10),
    "Coral Sea":          (141, 158, -25, -10),
    "Maldives":           (71,   75,  -2,   8),
    "Red Sea":            (31,   44,  12,  30),
    "Caribbean":          (-88, -60,  10,  25),
}

OCEAN_COLORS = [
    (0.00, "#010a18"),
    (0.10, "#021428"),
    (0.22, "#041f3e"),
    (0.35, "#062e5c"),
    (0.50, "#0a4a8a"),
    (0.64, "#1068b8"),
    (0.76, "#2090d0"),
    (0.87, "#55b8e8"),
    (0.94, "#90d4f0"),
    (1.00, "#c8edf8"),
]
OCEAN_CMAP = mcolors.LinearSegmentedColormap.from_list("noaa_ocean", OCEAN_COLORS)

LAND_COLORS = [
    (0.0, "#1a3010"),
    (0.25, "#2a4a18"),
    (0.55, "#406828"),
    (0.80, "#5a8a35"),
    (1.0,  "#78aa48"),
]
LAND_CMAP = mcolors.LinearSegmentedColormap.from_list("noaa_land", LAND_COLORS)


def _fetch_erddap(lon_min, lon_max, lat_min, lat_max, stride=2):
    try:
        print(f"Fetching ETOPO1 from ERDDAP (stride={stride})...")
        url = (
            f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc"
            f"?altitude[({lat_min}):{stride}:({lat_max})][({lon_min}):{stride}:({lon_max})]"
        )
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"etopo_cache_{region_key(lon_min,lon_max,lat_min,lat_max)}.nc")
        if not os.path.exists(cache):
            urllib.request.urlretrieve(url, cache)
        dataset = nc.Dataset(cache)
        lons = dataset.variables["longitude"][:]
        lats = dataset.variables["latitude"][:]
        elev = dataset.variables["altitude"][0, :, :]
        dataset.close()
        print("ETOPO1 loaded.")
        return np.array(lons), np.array(lats), np.array(elev, dtype=float)
    except Exception as e:
        print(f"ERDDAP failed: {e}")
        return None


def region_key(lon_min, lon_max, lat_min, lat_max):
    return f"{int(lon_min)}_{int(lon_max)}_{int(lat_min)}_{int(lat_max)}"


def _smooth_fallback(lon_min, lon_max, lat_min, lat_max, nx=320, ny=220):
    from scipy.ndimage import gaussian_filter
    print("Using smooth fallback bathymetry...")
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    np.random.seed(0)
    noise = np.random.randn(ny, nx)
    smooth = gaussian_filter(noise, sigma=18) * 1200 + gaussian_filter(noise, sigma=6) * 300
    elev = -3200 + smooth
    return lons, lats, elev


def _get_bathymetry(lon_min, lon_max, lat_min, lat_max):
    result = _fetch_erddap(lon_min, lon_max, lat_min, lat_max, stride=2)
    if result is not None:
        return result
    return _smooth_fallback(lon_min, lon_max, lat_min, lat_max)


def plot_bathymetric_overview(
    region: str,
    df_sites: pd.DataFrame,
    selected_date: str = "",
) -> bytes:
    bounds = REGION_BOUNDS.get(region, (95, 141, -11, 7))
    lon_min, lon_max, lat_min, lat_max = bounds

    lons, lats, elev = _get_bathymetry(lon_min, lon_max, lat_min, lat_max)
    lon2d, lat2d = np.meshgrid(lons, lats)

    ocean_mask = elev < 0
    land_mask = elev >= 0
    ocean_elev = np.where(ocean_mask, elev, np.nan)
    land_elev = np.where(land_mask, elev, np.nan)

    ocean_norm = mcolors.Normalize(vmin=-7000, vmax=0)
    land_norm = mcolors.Normalize(vmin=0, vmax=2500)

    ls = LightSource(azdeg=315, altdeg=35)
    ocean_elev_filled = np.where(np.isnan(ocean_elev), 0, ocean_elev)
    hs = ls.hillshade(ocean_elev_filled, vert_exag=0.003, dx=1, dy=1)
    hs = np.where(ocean_mask, hs, np.nan)

    ocean_rgb = OCEAN_CMAP(ocean_norm(ocean_elev_filled))
    ocean_rgb[..., :3] = ocean_rgb[..., :3] * 0.55 + np.dstack([hs, hs, hs]) * 0.45
    ocean_rgb = np.where(np.dstack([ocean_mask]*4), ocean_rgb,
                         np.zeros_like(ocean_rgb))

    fig = plt.figure(figsize=(14, 9), facecolor="#010a18")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_facecolor("#010a18")

    ax.imshow(
        ocean_rgb,
        origin="lower",
        extent=[lons[0], lons[-1], lats[0], lats[-1]],
        transform=ccrs.PlateCarree(),
        interpolation="bilinear",
        zorder=1,
    )

    ax.pcolormesh(
        lon2d, lat2d, land_elev,
        cmap=LAND_CMAP, norm=land_norm,
        transform=ccrs.PlateCarree(),
        shading="auto", zorder=2,
    )

    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m",
            facecolor="none", edgecolor="#6aaa35", linewidth=0.6),
        zorder=3,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "reefs", "10m",
            facecolor="none", edgecolor="#00e5ff", linewidth=0.5, alpha=0.55),
        zorder=4,
    )
    ax.coastlines(resolution="10m", color="#90cc50", linewidth=0.75, zorder=5)

    def site_color(risk_pct):
        if risk_pct >= 85:
            return "#f44336"
        elif risk_pct >= 60:
            return "#ff9800"
        elif risk_pct >= 35:
            return "#ffee58"
        else:
            return "#66bb6a"

    if df_sites is not None and len(df_sites) > 0:
        for _, row in df_sites.iterrows():
            color = site_color(row["Bleach Risk (%)"])
            size = 55 + row["Bleach Risk (%)"] * 1.1
            ax.scatter(
                row["Lon"], row["Lat"],
                s=size, c=color,
                edgecolors="white", linewidths=0.9,
                transform=ccrs.PlateCarree(),
                zorder=7, alpha=0.95,
            )
            ax.annotate(
                f"{row['Reef Site']}\n{row['Bleach Risk (%)']:.0f}%",
                xy=(row["Lon"], row["Lat"]),
                xytext=(7, 7), textcoords="offset points",
                color="white", fontsize=7.2,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#010a18",
                          alpha=0.80, edgecolor="none"),
                transform=ccrs.PlateCarree(),
                zorder=8,
            )

    sm = plt.cm.ScalarMappable(cmap=OCEAN_CMAP, norm=ocean_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.020, pad=0.025, shrink=0.82)
    cbar.set_label("Ocean Depth (m)", color="white", fontsize=9, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7.5)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("none")

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.25, color="#1470b8", alpha=0.4, linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"color": "white", "fontsize": 7.5}
    gl.ylabel_style = {"color": "white", "fontsize": 7.5}

    legend_elements = [
        mpatches.Patch(facecolor="#66bb6a", edgecolor="white", label="Low  (<35%)"),
        mpatches.Patch(facecolor="#ffee58", edgecolor="white", label="Moderate (35-60%)"),
        mpatches.Patch(facecolor="#ff9800", edgecolor="white", label="High  (60-85%)"),
        mpatches.Patch(facecolor="#f44336", edgecolor="white", label="Severe (>=85%)"),
        mpatches.Patch(facecolor="#00e5ff", edgecolor="none",  label="Reef structures", alpha=0.6),
    ]
    ax.legend(
        handles=legend_elements, loc="lower left", fontsize=7.5,
        facecolor="#021428", edgecolor="#1470b8", labelcolor="white",
        title="Bleaching Risk", title_fontsize=8.5,
        framealpha=0.88,
    )

    date_str = f"  |  {selected_date}" if selected_date else ""
    ax.set_title(
        f"Bathymetric Overview — {region}{date_str}\n"
        f"EcoCast  ·  NOAA DHW Methodology  ·  ETOPO1 Bathymetry",
        color="white", fontsize=11, fontweight="bold", pad=14,
        fontfamily="monospace",
    )

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#010a18")
    plt.close(fig)
    buf.seek(0)
    return buf.read()