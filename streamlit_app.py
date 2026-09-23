"""Streamlit earthquake dashboard with tectonic plate correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from shapely import wkt
from shapely.geometry import Point
from shapely.strtree import STRtree


st.set_page_config(page_title="Earthquake Dashboard", page_icon="🌍", layout="wide")


# Colors for each boundary type
PLATE_COLORS = {
    "Convergent Boundary": [255, 100, 100, 180],
    "Divergent Boundary":  [100, 150, 255, 180],
    "Transform Boundary":  [255, 200,  80, 180],
    "Other":               [180, 180, 180, 180],
}
DEFAULT_COLOR = [150, 150, 150, 180]


# Data loaders

@st.cache_data
def demo_data() -> pd.DataFrame:
    """Fallback demo data if nothing else loads."""
    rng = np.random.default_rng(42)
    count = 250
    return pd.DataFrame(
        {
            "time": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 180, count), unit="D"),
            "latitude": rng.uniform(-55, 65, count),
            "longitude": rng.uniform(-170, 170, count),
            "magnitude": np.round(rng.uniform(2.5, 7.2, count), 1),
            "depth_km": np.round(rng.uniform(1, 180, count), 1),
            "place": rng.choice(["Pacific Ocean", "Japan", "California", "Chile", "Indonesia"], count),
        }
    ).sort_values("time")


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    """Load a USGS-style earthquake CSV (uploaded file OR path)."""
    data = pd.read_csv(file).rename(
        columns={"mag": "magnitude", "lat": "latitude", "lon": "longitude", "depth": "depth_km"}
    )
    required = {"latitude", "longitude", "magnitude"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if "time" not in data:
        data["time"] = pd.Timestamp.today().normalize()
    # Force UTC-naive so resample/date ops work cleanly
    data["time"] = pd.to_datetime(data["time"], errors="coerce", utc=True).dt.tz_localize(None)
    return data.dropna(subset=["time", *required])


@st.cache_data
def load_plates(path: str = "tectonic_plates.csv") -> pd.DataFrame:
    """Load tectonic plates CSV and parse WKT geometry strings."""
    plates = pd.read_csv(path)
    plates = plates.dropna(subset=["geometry"]).copy()
    plates["geometry"] = plates["geometry"].apply(wkt.loads)

    # Convert every geometry (LineString or MultiLineString) into a list of
    # coordinate paths so pydeck's PathLayer can draw them.
    def to_paths(geom):
        if geom.geom_type == "LineString":
            return [list(geom.coords)]
        if geom.geom_type == "MultiLineString":
            return [list(line.coords) for line in geom.geoms]
        return []

    plates["paths"] = plates["geometry"].apply(to_paths)
    return plates


@st.cache_data
def annotate_with_plates(eq_df: pd.DataFrame, plates_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest plate boundary info to every earthquake."""
    if eq_df.empty:
        return eq_df.assign(
            nearest_plate=pd.Series(dtype="object"),
            plate_label=pd.Series(dtype="object"),
            distance_deg=pd.Series(dtype="float64"),
            distance_km=pd.Series(dtype="float64"),
        )

    geoms = plates_df["geometry"].tolist()
    tree = STRtree(geoms)

    names, labels, dists = [], [], []
    for lon, lat in zip(eq_df["longitude"], eq_df["latitude"]):
        try:
            p = Point(float(lon), float(lat))
            i = tree.nearest(p)
            g = geoms[i]
            names.append(plates_df.iloc[i]["NAME"])
            labels.append(plates_df.iloc[i]["LABEL"])
            dists.append(p.distance(g))
        except Exception:
            names.append(None)
            labels.append(None)
            dists.append(np.nan)

    out = eq_df.assign(
        nearest_plate=names,
        plate_label=labels,
        distance_deg=dists,
    )
    # Approximate km (shrinks with latitude)
    out["distance_km"] = out["distance_deg"] * 111 * np.cos(np.radians(out["latitude"]))
    return out


# Header

st.title("🌍 Earthquake Dashboard")
st.caption("Explore earthquake activity and how it correlates with tectonic plate boundaries.")


# Sidebar: upload + filters

with st.sidebar:
    st.header("Filters")
    upload = st.file_uploader("Upload earthquake CSV", type="csv")

    try:
        if upload:
            earthquakes = load_csv(upload)
        else:
            earthquakes = load_csv("2.5_month.csv")
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as error:
        st.error(f"Could not load earthquake data: {error}")
        st.stop()

    try:
        plates = load_plates("tectonic_plates.csv")
    except FileNotFoundError:
        plates = None
        st.warning("tectonic_plates.csv not found — plate correlation disabled.")

    # Annotate earthquakes with nearest plate boundary (cached)
    if plates is not None:
        with st.spinner("Matching earthquakes to plate boundaries…"):
            earthquakes = annotate_with_plates(earthquakes, plates)

    minimum = float(earthquakes["magnitude"].min())
    maximum = float(earthquakes["magnitude"].max())
    min_magnitude = st.slider("Minimum magnitude", 0.0, max(10.0, maximum), minimum, 0.1)
    available_dates = earthquakes["time"].dt.date
    date_range = st.date_input("Date range", (available_dates.min(), available_dates.max()))

    # Optional: filter to quakes close to a boundary
    if plates is not None:
        max_distance = st.slider(
            "Max distance to plate boundary (km)",
            min_value=0,
            max_value=3000,
            value=3000,
            step=50,
        )
    else:
        max_distance = None


# Apply filters

filtered = earthquakes[earthquakes["magnitude"] >= min_magnitude].copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[filtered["time"].dt.date.between(date_range[0], date_range[1])]
if max_distance is not None and "distance_km" in filtered:
    filtered = filtered[filtered["distance_km"].fillna(np.inf) <= max_distance]


# Metrics

col1, col2, col3, col4 = st.columns(4)
col1.metric("Earthquakes", f"{len(filtered):,}")
col2.metric("Largest magnitude", f"{filtered.magnitude.max():.1f}" if len(filtered) else "—")
col3.metric(
    "Average depth",
    f"{filtered.depth_km.mean():.1f} km" if "depth_km" in filtered and len(filtered) else "—",
)
if "distance_km" in filtered and len(filtered):
    col4.metric("Median dist. to boundary", f"{filtered.distance_km.median():.0f} km")
else:
    col4.metric("Median dist. to boundary", "—")


# Map + activity over time

map_col, chart_col = st.columns([1.25, 1])

with map_col:
    st.subheader("Earthquake locations")

    # Legend
    st.markdown(
        "<span style='color:#ff6464'>●</span> Convergent &nbsp; "
        "<span style='color:#6496ff'>●</span> Divergent &nbsp; "
        "<span style='color:#ffc850'>●</span> Transform &nbsp; "
        "<span style='color:#b4b4b4'>●</span> Other &nbsp; "
        "— <em>dot size = magnitude</em>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No earthquakes match the selected filters.")
    else:
        map_df = filtered.dropna(subset=["latitude", "longitude"]).copy()
        if "plate_label" in map_df and map_df["plate_label"].notna().any():
            map_df["color"] = map_df["plate_label"].map(PLATE_COLORS).apply(
                lambda c: c if isinstance(c, list) else DEFAULT_COLOR
            )
            map_df["radius"] = map_df["magnitude"].clip(lower=1) * 8000

            eq_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.8,
                stroked=False,
            )
            view = pdk.ViewState(latitude=10, longitude=0, zoom=1, pitch=0)

            # Plate boundary lines layer
            layers = [eq_layer]
            if plates is not None:
                plate_layer = pdk.Layer(
                    "PathLayer",
                    data=plates,
                    get_path="paths",
                    get_color=[255, 255, 255, 110],
                    width_scale=15,
                    width_min_pixels=1,
                    pickable=False,
                )
                layers.append(plate_layer)

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view,
                    tooltip={"text": "{place}\nMag: {magnitude}\n{plate_label}"},
                )
            )
        else:
            st.map(
                map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]]
            )

with chart_col:
    st.subheader("Activity over time")
    if filtered.empty:
        st.info("No data to plot.")
    else:
        st.line_chart(
            filtered.set_index("time").resample("D").size().rename("earthquakes")
        )



# Plate boundary correlation


if "plate_label" in filtered and filtered["plate_label"].notna().any():
    st.divider()
    st.header("🌐 Correlation with tectonic plate boundaries")

    boundary_stats = (
        filtered.groupby("plate_label")
        .agg(
            count=("magnitude", "size"),
            avg_magnitude=("magnitude", "mean"),
            max_magnitude=("magnitude", "max"),
            avg_depth_km=("depth_km", "mean") if "depth_km" in filtered else ("magnitude", "size"),
            median_distance_km=("distance_km", "median"),
        )
        .round(2)
        .sort_values("count", ascending=False)
    )

    b1, b2 = st.columns([1, 1])

    with b1:
        st.subheader("Stats by boundary type")
        st.dataframe(boundary_stats, use_container_width=True)

    with b2:
        st.subheader("Average magnitude by boundary type")
        st.bar_chart(boundary_stats["avg_magnitude"])

    st.subheader("Magnitude distribution by boundary type")

    st.bar_chart(
        filtered.dropna(subset=["plate_label"]),
        x="plate_label",
        y="magnitude",
        color="plate_label",
    )

    st.subheader("Depth vs. distance to nearest plate boundary")
    scatter_df = filtered.dropna(subset=["distance_km", "depth_km"])
    if not scatter_df.empty:
        st.scatter_chart(
            scatter_df,
            x="distance_km",
            y="depth_km",
            color="plate_label",
            size="magnitude",
        )
    else:
        st.info("Not enough data for scatter plot.")

    st.subheader("Are quakes clustered near boundaries?")
    st.caption("Most earthquakes should fall within ~200 km of a plate boundary.")
    hist = pd.cut(
        filtered["distance_km"],
        bins=[0, 50, 100, 200, 500, 1000, 2000, np.inf],
        labels=["0–50", "50–100", "100–200", "200–500", "500–1k", "1k–2k", ">2k"],
    ).value_counts().sort_index()
    st.bar_chart(hist)


# Recent earthquakes table


st.divider()
st.subheader("Recent earthquakes")

columns = [
    c
    for c in [
        "time",
        "place",
        "magnitude",
        "depth_km",
        "latitude",
        "longitude",
        "nearest_plate",
        "plate_label",
        "distance_km",
    ]
    if c in filtered
]

if filtered.empty:
    st.info("No earthquakes match the selected filters.")
else:
    st.dataframe(
        filtered.sort_values("time", ascending=False)[columns].head(100),
        use_container_width=True,
        hide_index=True,
    )