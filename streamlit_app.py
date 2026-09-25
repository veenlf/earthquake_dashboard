"""Streamlit earthquake dashboard with tectonic plate correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt
from shapely import wkt
from shapely.geometry import Point
from shapely.strtree import STRtree
from pyproj import Geod 
from shapely.ops import nearest_points


st.set_page_config(page_title="Aardbevingen Dashboard", page_icon="🌍", layout="wide")


# Colors for each boundary type
PLATE_COLORS = {
    "Convergent Boundary": [255, 100, 100, 180],
    "Divergent Boundary":  [100, 150, 255, 180],
    "Transform Boundary":  [255, 200,  80, 180],
    "Other":               [180, 180, 180, 180],
}
DEFAULT_COLOR = [150, 150, 150, 180]

PLATE_LABELS_NL = {
    "Convergent Boundary": "Convergente grens",
    "Divergent Boundary":  "Divergente grens",
    "Transform Boundary":  "Transforme grens",
    "Other":               "Overig",
}


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
        raise ValueError(f"Ontbrekende kolommen: {', '.join(sorted(missing))}")
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

    # Convert each geometry into a GeoJSON-style Feature so pydeck's
    # GeoJsonLayer can render it reliably (works for LineString AND
    # MultiLineString, unlike PathLayer which is picky about format).
    def to_feature(row):
        g = row["geometry"]
        if g.geom_type == "LineString":
            coords = [list(map(float, c)) for c in g.coords]
        elif g.geom_type == "MultiLineString":
            coords = [[list(map(float, c)) for c in line.coords] for line in g.geoms]
        else:
            return None
        return {
            "type": "Feature",
            "geometry": {"type": g.geom_type, "coordinates": coords},
            "properties": {"name": row.get("NAME"), "label": row.get("LABEL")},
        }

    plates["feature"] = plates.apply(to_feature, axis=1)
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
    geom_to_idx = {geom: idx for idx, geom in enumerate(geoms)}

    tree = STRtree(geoms)

    names, labels, distances_km = [], [], []
    geod = Geod(ellps="WGS84")

    for lon, lat in zip(eq_df["longitude"], eq_df["latitude"]):
        try:
            p = Point(float(lon), float(lat))
            nearest_geom = tree.nearest(p)
            idx = geom_to_idx[nearest_geom]

            names.append(plates_df.iloc[idx]["NAME"])
            labels.append(plates_df.iloc[idx]["LABEL"])

            nearest_point = nearest_points(p, nearest_geom)[1]
            _, _, distance_m = geod.inv(
                p.x,
                p.y,
                nearest_point.x,
                nearest_point.y
            )
            distances_km.append(distance_m / 1000)

        except Exception:
            names.append(None)
            labels.append(None)
            distances_km.append(np.nan)

    out = eq_df.assign(
        nearest_plate=names,
        plate_label=labels,
        distance_km=distances_km,
    )

    return out


# Header

st.title("🌍 Aardbevingen Dashboard")
st.markdown(
    "**Onderzoeksvraag:** Wat is de relatie tussen aardbevingen en de grenzen van tektonische platen?"
)
st.caption("Verken aardbevingsactiviteit en hoe deze samenhangt met de grenzen van tektonische platen.")


# Sidebar: upload + filters

with st.sidebar:
    st.header("Filters")
    upload = st.file_uploader("Upload aardbevingen CSV", type="csv")

    try:
        if upload:
            earthquakes = load_csv(upload)
        else:
            earthquakes = load_csv("2.5_month.csv")
    except (ValueError, FileNotFoundError, pd.errors.ParserError) as error:
        st.error(f"Kon aardbevingsdata niet laden: {error}")
        st.stop()

    try:
        plates = load_plates("tectonic_plates.csv")
    except FileNotFoundError:
        plates = None
        st.warning("tectonic_plates.csv niet gevonden — plaatcorrelatie uitgeschakeld.")

    # Annotate earthquakes with nearest plate boundary (cached)
    if plates is not None:
        with st.spinner("Aardbevingen koppelen aan plaatgrenzen…"):
            earthquakes = annotate_with_plates(earthquakes, plates)

    if "plate_label" in earthquakes.columns:
        earthquakes["plate_label_nl"] = earthquakes["plate_label"].map(PLATE_LABELS_NL).fillna(earthquakes["plate_label"])

    # DYNAMIC FILTERS
    min_mag_data = float(earthquakes["magnitude"].min())
    max_mag_data = float(earthquakes["magnitude"].max())

    min_magnitude = st.slider(
        "Minimale magnitude",
        min_value=min_mag_data,
        max_value=max_mag_data,
        value=min_mag_data,
        step=0.1,
    )

    available_dates = earthquakes["time"].dt.date
    min_date_data = available_dates.min()
    max_date_data = available_dates.max()

    date_range = st.date_input(
        "Datumbereik",
        value=(min_date_data, max_date_data),
        min_value=min_date_data,
        max_value=max_date_data,
    )

    # Optional: filter to quakes close to a boundary
    if plates is not None:
        max_dist_data = float(earthquakes["distance_km"].max()) if "distance_km" in earthquakes.columns else 3000.0
        max_distance = st.slider(
            "Maximale afstand tot plaatgrens (km)",
            min_value=0.0,
            max_value=max_dist_data,
            value=max_dist_data,
            step=50.0,
        )
    else:
        max_distance = None

    if "plate_label_nl" in earthquakes.columns:
        boundary_options = ["Alle"] + sorted(earthquakes['plate_label_nl'].dropna().unique().tolist())
        boundary_type_nl = st.selectbox(
            "Grens类型",
            boundary_options
        )
    else:
        boundary_type_nl = 'Alle'

# Apply filters

filtered = earthquakes[earthquakes["magnitude"] >= min_magnitude].copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[filtered["time"].dt.date.between(date_range[0], date_range[1])]
if max_distance is not None and "distance_km" in filtered.columns:
    filtered = filtered[filtered["distance_km"].fillna(np.inf) <= max_distance]
if boundary_type_nl != 'Alle' and "plate_label_nl" in filtered.columns:
    filtered = filtered[filtered['plate_label_nl'] == boundary_type_nl]


# Metrics

col1, col2, col3, col4 = st.columns(4)
col1.metric("Aardbevingen", f"{len(filtered):,}")
col2.metric("Grootste magnitude", f"{filtered.magnitude.max():.1f}" if len(filtered) else "—")

if "depth_km" in filtered.columns and len(filtered):
    col3.metric("Gemiddelde diepte", f"{filtered['depth_km'].mean():.1f} km")
else:
    col3.metric("Gemiddelde diepte", "—")

if "distance_km" in filtered.columns and len(filtered):
    col4.metric("Mediaan afstand tot grens", f"{filtered['distance_km'].median():.0f} km")
else:
    col4.metric("Mediaan afstand tot grens", "—")


# Map + activity over time

map_col, chart_col = st.columns([1.25, 1])

with map_col:
    st.subheader("Waar komen aardbevingen voor?")

    # Legend
    st.markdown(
        "<span style='color:#ff6464'>●</span> Convergent &nbsp; "
        "<span style='color:#6496ff'>●</span> Divergent &nbsp; "
        "<span style='color:#ffc850'>●</span> Transform &nbsp; "
        "<span style='color:#b4b4b4'>●</span> Overig<br>"
        "● grootte stip = magnitude",
        unsafe_allow_html=True,
    )

    show_boundaries = st.checkbox(
        "Toon grenzen van tektonische platen",
        value=True
    )

    if filtered.empty:
        st.info("Geen aardbevingen voldoen aan de geselecteerde filters.")
    else:
        map_df = filtered.dropna(subset=["latitude", "longitude"]).copy()
        if "plate_label" in map_df.columns and map_df["plate_label"].notna().any():
            map_df["color"] = map_df["plate_label"].map(PLATE_COLORS).apply(
                lambda c: c if isinstance(c, list) else DEFAULT_COLOR
            )
            map_df["radius"] = map_df["magnitude"].clip(lower=1) * 8000

            view = pdk.ViewState(latitude=10, longitude=0, zoom=1, pitch=0)

            layers = []

            # Plate boundary lines — GeoJsonLayer (reliable, sits behind dots)
            if show_boundaries and plates is not None:
                plate_features = [
                    f for f in plates["feature"].tolist() if f is not None
                ]
                if plate_features:
                    plate_layer = pdk.Layer(
                        "GeoJsonLayer",
                        data={"type": "FeatureCollection", "features": plate_features},
                        stroked=True,
                        filled=False,
                        get_line_color=[20, 80, 120, 220],
                        get_line_width=2500,
                        line_width_min_pixels=2,
                        line_width_max_pixels=4,
                        pickable=False,
                    )
                    layers.append(plate_layer)

            # Earthquake dots — drawn on top
            eq_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=1,
                stroked=False,
            )
            layers.append(eq_layer)

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view,
                    tooltip={"text": "{place}\nMag: {magnitude}\n{plate_label_nl}"},
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                )
            )
        else:
            st.map(
                map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]]
            )

with chart_col:
    st.subheader("Wanneer komen aardbevingen voor?")
    if filtered.empty:
        st.info("Geen data om te plotten.")
    else:
        st.line_chart(
            filtered.set_index("time").resample("D").size().rename("aardbevingen")
        )


# Plate boundary correlation


if "plate_label_nl" in filtered.columns and filtered["plate_label_nl"].notna().any():
    st.divider()
    st.header("Aardbevingen en grenzen van tektonische platen")

    agg_dict = {
        "aantal": ("magnitude", "size"),
        "gem_magnitude": ("magnitude", "mean"),
        "max_magnitude": ("magnitude", "max"),
        "mediaan_afstand_km": ("distance_km", "median"),
    }
    if "depth_km" in filtered.columns:
        agg_dict["gem_diepte_km"] = ("depth_km", "mean")

    boundary_stats = (
        filtered.groupby("plate_label_nl")
        .agg(**agg_dict)
        .round(2)
        .sort_values("aantal", ascending=False)
    )

    b1, b2 = st.columns([1, 1])

    with b1:
        st.subheader("Hoe verschillen aardbevingen per grens类型?")
        st.dataframe(boundary_stats, use_container_width=True)

    with b2:
        st.subheader("Hoe sterk zijn aardbevingen bij verschillende grens类型en?")
        box_chart = alt.Chart(filtered.dropna(subset=["plate_label_nl", "magnitude"])).mark_boxplot().encode(
            x=alt.X("plate_label_nl:N", title="Grens类型", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("magnitude:Q", title="Magnitude"),
            color=alt.Color("plate_label_nl:N", legend=None)
        ).properties(height=300)
        st.altair_chart(box_chart, use_container_width=True)


    mag_df = filtered.dropna(subset=["plate_label_nl", "magnitude"]).copy()
    mag_df["magnitude_group"] = pd.cut(
        mag_df['magnitude'],
        bins=[0, 3, 4, 5, 6, 7, 8, 9, np.inf],
        labels=['<3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9+']
    )

    mag_counts = (
        mag_df.groupby(['plate_label_nl', 'magnitude_group'], observed=True).size().reset_index(name='aantal')
    )
    mag_counts['percentage'] = (
        mag_counts['aantal'] / mag_counts.groupby("plate_label_nl")['aantal'].transform('sum') * 100
    )

    st.subheader("Hoe zijn de magnitudes verdeeld per grens类型?")

    chart = alt.Chart(mag_counts).mark_bar().encode(
        x=alt.X("plate_label_nl:N", title="Grens类型", axis=alt.Axis(labelAngle=-45, labelLimit=250)),
        y=alt.Y("percentage:Q", title="percentage aardbevingen"),
        color=alt.Color(
            "magnitude_group:N",
            scale=alt.Scale(
                domain=["<3", "3-4", "4-5", "5-6", "6-7"],
                range=["#deebf7", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
            ),
            title="magnitude groep"
        ),
        tooltip=[
            alt.Tooltip("plate_label_nl:N", title="grens类型"),
            alt.Tooltip("magnitude_group:N", title='magnitude'),
            alt.Tooltip('aantal:Q', title='aantal aardbevingen'),
            alt.Tooltip('percentage:Q', title='percentage', format='.1f')
        ]
    )

    st.altair_chart(chart, use_container_width=True)

    st.subheader("Hangt de afstand tot een plaatgrens samen met de diepte van aardbevingen?")

    scatter_df = filtered.dropna(subset=["distance_km", "depth_km"])

    st.write(f"Weergegeven aardbevingen: {len(scatter_df)}")

    if not scatter_df.empty:
        st.scatter_chart(
            scatter_df,
            x="distance_km",
            y="depth_km",
            color="plate_label_nl",
            size="magnitude",
            opacity=0.6,
        )
    else:
        st.info("Niet genoeg data voor spreidingsdiagram.")

    st.subheader("Hoe dicht liggen aardbevingen bij plaatgrenzen?")
    st.caption("De meeste aardbevingen zouden binnen ~200 km van een plaatgrens moeten liggen.")
    if not filtered.empty and "distance_km" in filtered.columns:
        dist_data = filtered["distance_km"].dropna()
        if not dist_data.empty:
            hist = pd.cut(
                dist_data,
                bins=[0, 50, 100, 200, 500, 1000, 2000, np.inf],
                labels=["0–50", "50–100", "100–200", "200–500", "500–1k", "1k–2k", ">2k"],
            ).value_counts().sort_index()
            st.bar_chart(hist)
        else:
            st.info("Geen afstandsdata beschikbaar voor histogram.")
    else:
        st.info("Afstandsdata niet beschikbaar.")


# Recent earthquakes table


st.divider()
st.subheader("Recente aardbevingen")

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
        "plate_label_nl",
        "distance_km",
    ]
    if c in filtered.columns
]

if filtered.empty:
    st.info("Geen aardbevingen voldoen aan de geselecteerde filters.")
else:
    st.dataframe(
        filtered.sort_values("time", ascending=False)[columns].head(100),
        use_container_width=True,
        hide_index=True,
    )