"""Streamlit earthquake dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Earthquake Dashboard", page_icon="🌍", layout="wide")


@st.cache_data
def demo_data() -> pd.DataFrame:
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
	data = pd.read_csv(file).rename(
		columns={"mag": "magnitude", "lat": "latitude", "lon": "longitude", "depth": "depth_km"}
	)
	required = {"latitude", "longitude", "magnitude"}
	missing = required - set(data.columns)
	if missing:
		raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
	if "time" not in data:
		data["time"] = pd.Timestamp.today().normalize()
	data["time"] = pd.to_datetime(data["time"], errors="coerce")
	return data.dropna(subset=["time", *required])


st.title("🌍 Earthquake Dashboard")
st.caption("Explore earthquake activity by location, magnitude, and time.")

with st.sidebar:
	st.header("Filters")
	upload = st.file_uploader("Upload earthquake CSV", type="csv")
	try:
		earthquakes = load_csv(upload) if upload else demo_data()
	except (ValueError, pd.errors.ParserError) as error:
		st.error(str(error))
		st.stop()

	minimum = float(earthquakes["magnitude"].min())
	maximum = float(earthquakes["magnitude"].max())
	min_magnitude = st.slider("Minimum magnitude", 0.0, max(10.0, maximum), minimum, 0.1)
	available_dates = earthquakes["time"].dt.date
	date_range = st.date_input("Date range", (available_dates.min(), available_dates.max()))

filtered = earthquakes[earthquakes["magnitude"] >= min_magnitude].copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
	filtered = filtered[filtered["time"].dt.date.between(date_range[0], date_range[1])]

col1, col2, col3 = st.columns(3)
col1.metric("Earthquakes", f"{len(filtered):,}")
col2.metric("Largest magnitude", f"{filtered.magnitude.max():.1f}" if len(filtered) else "—")
col3.metric("Average depth", f"{filtered.depth_km.mean():.1f} km" if "depth_km" in filtered and len(filtered) else "—")

map_col, chart_col = st.columns([1.25, 1])
with map_col:
	st.subheader("Earthquake locations")
	if filtered.empty:
		st.info("No earthquakes match the selected filters.")
	else:
		st.map(filtered.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
with chart_col:
	st.subheader("Activity over time")
	st.line_chart(filtered.set_index("time").resample("D").size().rename("earthquakes"))

st.subheader("Recent earthquakes")
columns = [c for c in ["time", "place", "magnitude", "depth_km", "latitude", "longitude"] if c in filtered]
st.dataframe(filtered.sort_values("time", ascending=False)[columns].head(100), use_container_width=True, hide_index=True)
