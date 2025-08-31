import pandas as pd
import plotly.express as px
from typing import Dict, Optional
import numpy as np
import plotly.graph_objects as go

def choose_port(padd: str, target_lonlat: tuple) -> tuple:
    """Pick the port closest to the target point."""
    ports = PADD_PORTS[padd]
    tlon, tlat = target_lonlat
    dists = [ ( (tlon - lon)**2 + (tlat - lat)**2, (lon,lat) ) for lon,lat in ports ]
    return min(dists, key=lambda x: x[0])[1]

def curve_points(lon1, lat1, lon2, lat2, k=0.2, n=30):
    """
    Make n points of a simple curved arc between (lon1,lat1) and (lon2,lat2).
    k controls bulge (0=straight). Works in lon/lat space (good enough for US maps).
    """
    t = np.linspace(0,1,n)
    lon = (1-t)*lon1 + t*lon2
    lat = (1-t)*lat1 + t*lat2
    # perpendicular offset for curve
    dx, dy = lon2-lon1, lat2-lat1
    # normalize
    L = max(np.hypot(dx,dy), 1e-9)
    nx, ny = -dy/L, dx/L
    bulge = k * np.sin(np.pi*t)
    lon_curved = lon + bulge * nx
    lat_curved = lat + bulge * ny
    return lon_curved, lat_curved

def width_from_volume(v_bbl: float) -> float:
    return max(1.0, 0.7 * np.log10(max(v_bbl, 1)))

# PADD centroids (approx)
PADD_CENTROID = {
    "PADD1": (-77.0, 39.0),   # lon, lat (Mid-Atlantic-ish)
    "PADD2": (-93.0, 41.5),   # Upper Midwest
    "PADD3": (-94.5, 29.5),   # Gulf Coast (near TX/LA coast)
    "PADD4": (-111.9, 44.3),  # Rockies
    "PADD5": (-120.5, 38.5),  # West Coast
}

# Import/export “ports” — one or two per PADD for the line endpoints
PADD_PORTS = {
    "PADD1": [(-74.0, 40.7), (-75.2, 39.0)],      # NY/NJ, Delaware Bay
    "PADD2": [(-92.1, 46.8)],                     # Duluth (Great Lakes proxy) / US–Canada border
    "PADD3": [(-95.1, 29.7), (-90.1, 29.9)],      # Houston, New Orleans
    "PADD4": [(-111.5, 49.0)],                    # MT–Canada border
    "PADD5": [(-118.3, 34.0), (-122.4, 37.8), (-122.3, 47.6)],  # LA, SF, Seattle
}



# PADD → states (EIA convention)
PADD_STATES: Dict[str, list] = {
    "PADD 1 (East Coast)": [
        "CT","DE","DC","FL","GA","ME","MD","MA","NH","NJ","NY",
        "NC","PA","RI","SC","VT","VA","WV"
    ],
    "PADD 2 (Midwest)": [
        "IL","IN","IA","KS","KY","MI","MN","MO","NE","ND","OH",
        "OK","SD","TN","WI"
    ],
    "PADD 3 (Gulf Coast)": ["AL","AR","LA","MS","NM","TX"],
    "PADD 4 (Rocky Mountain)": ["CO","ID","MT","UT","WY"],
    "PADD 5 (West Coast)": ["AK","AZ","CA","HI","NV","OR","WA"],
}

DEFAULT_PADD_COLORS = {
    "PADD 1": "#636EFA",
    "PADD 2": "#EF553B",
    "PADD 3": "#00CC96",
    "PADD 4": "#AB63FA",
    "PADD 5": "#FFA15A",
}

def make_padd_choropleth(
    title: str = "U.S. States by PADD District",
    padd_colors: Optional[Dict[str, str]] = None
):
    """Return a Plotly choropleth of U.S. states colored by PADD."""
    padd_colors = padd_colors or DEFAULT_PADD_COLORS

    # Build tidy df: one row per state with its PADD label(s)
    rows = []
    for padd_full, states in PADD_STATES.items():
        padd_id = f"PADD {padd_full.split()[1]}"  # "PADD 1".."PADD 5"
        for st in states:
            rows.append({"state": st, "padd_id": padd_id, "padd_name": padd_full})
    df = pd.DataFrame(rows)

    fig = px.choropleth(
        df,
        locations="state",                 # uses state abbreviations
        locationmode="USA-states",
        color="padd_id",
        scope="usa",
        hover_data={"state": True, "padd_name": True, "padd_id": False},
        category_orders={"padd_id": ["PADD 1","PADD 2","PADD 3","PADD 4","PADD 5"]},
        color_discrete_map=padd_colors,
    )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="PADD"
    )
    fig.update_geos(showlakes=True, lakecolor="LightBlue")
    return fig

def add_flow_traces(fig, flows_df, period=None, product=None):
    # Filter
    df = flows_df.copy()
    if period is not None:
        df = df[df["period"] == period]
    if product is not None:
        df = df[df["product"] == product]

    COLOR = {"import":"#1f77b4", "export":"#d62728", "inter_padd":"#7f7f7f"}
    traces = []

    for _, r in df.iterrows():
        dirn = r["direction"]
        v    = float(r["value_bbl"])
        w    = width_from_volume(v)

        if dirn == "inter_padd":
            src = r["origin_area"]; dst = r["dest_area"]
            (lon1,lat1) = PADD_CENTROID[src]
            (lon2,lat2) = PADD_CENTROID[dst]
            lon, lat = curve_points(lon1, lat1, lon2, lat2, k=0.25, n=40)

        elif dirn == "import":
            dst = r["dest_area"]
            # use PADD centroid as destination
            (lon2,lat2) = PADD_CENTROID[dst]
            # pick a port in the dest PADD
            port = choose_port(dst, (lon2,lat2))
            (lon1,lat1) = port
            lon, lat = curve_points(lon1, lat1, lon2, lat2, k=0.18, n=35)

        elif dirn == "export":
            src = r["origin_area"]
            (lon1,lat1) = PADD_CENTROID[src]
            port = choose_port(src, (lon1,lat1))
            (lon2,lat2) = port
            lon, lat = curve_points(lon1, lat1, lon2, lat2, k=0.18, n=35)

        else:
            continue

        hover = []
        if dirn == "inter_padd":
            hover.append(f"{r.get('product','all').title()} inter-PADD")
            hover.append(f"{r['origin_area']} → {r['dest_area']}")
        elif dirn == "import":
            hover.append(f"{r.get('product','all').title()} import")
            src_lbl = r.get("origin_country") or "Foreign"
            hover.append(f"{src_lbl} → {r['dest_area']}")
        elif dirn == "export":
            hover.append(f"{r.get('product','all').title()} export")
            dst_lbl = r.get("dest_country") or "Foreign"
            hover.append(f"{r['origin_area']} → {dst_lbl}")

        hover.append(f"Volume: {v:,.0f} bbl")
        if r.get("period"):
            hover.append(f"Period: {r['period']}")
        hovertext = "<br>".join(hover)

        traces.append(go.Scattergeo(
            lon=lon, lat=lat,
            mode="lines",
            line=dict(width=w, color=COLOR[dirn]),
            hoverinfo="text",
            text=hovertext,
            opacity=0.8,
            name=dirn.capitalize(),
            showlegend=False
        ))

    for tr in traces:
        fig.add_trace(tr)
    
# Example usage:
# fig = make_padd_choropleth()
# fig.show()
fig = make_padd_choropleth()