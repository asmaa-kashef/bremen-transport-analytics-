
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Bremen Transport Analytics",
    page_icon="🚌",
    layout="wide"
)

DATA_PATH = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    raw = pd.read_csv(
        DATA_PATH / "vbn_delays.csv",
        low_memory=False
    )

    original_rows = len(raw)
    duplicates = int(raw.duplicated().sum())

    df = raw.drop_duplicates().copy()
    df["collected_at"] = pd.to_datetime(
        df["collected_at"], utc=True, errors="coerce"
    )

    for col in ["arrival_delay_s", "departure_delay_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, original_rows, duplicates

df, original_rows, duplicates = load_data()

st.title("🚌 Bremen Public Transport Analytics")
st.caption("Interactive exploration of VBN real-time delay observations")

st.sidebar.header("Dashboard Filters")

snapshots = sorted(df["collected_at"].dropna().unique())

selected = st.sidebar.multiselect(
    "Collection snapshots",
    options=snapshots,
    default=snapshots,
    format_func=lambda x: pd.Timestamp(x).strftime("%H:%M:%S UTC")
)

if not selected:
    st.warning("Select at least one collection snapshot.")
    st.stop()

data = df[df["collected_at"].isin(selected)].copy()

st.warning(
    "The current real-time sample covers a short collection period. "
    "Treat results as exploratory, not as long-term service performance."
)

overview_tab, delay_tab, quality_tab = st.tabs([
    "📊 Data Overview",
    "⏱️ Delay Analysis",
    "🔍 Data Quality"
])

with overview_tab:
    st.header("Data Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Realtime Records", f"{len(data):,}")
    c2.metric("Unique Trips", f"{data['trip_id'].nunique():,}")
    c3.metric("Unique Routes", f"{data['route_id'].nunique():,}")
    c4.metric("Unique Stops", f"{data['stop_id'].nunique():,}")

    st.subheader("Records collected over time")
    snapshot_counts = (
        data.groupby("collected_at")
        .size()
        .reset_index(name="Records")
    )

    fig = px.bar(
        snapshot_counts,
        x="collected_at",
        y="Records",
        title="Number of records in each snapshot"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Realtime data preview")
    st.dataframe(data.head(30), use_container_width=True)

with delay_tab:
    st.header("Delay Analysis")

    arrival = data["arrival_delay_s"].dropna()
    departure = data["departure_delay_s"].dropna()

    arrival_mean = arrival.mean() if len(arrival) else None
    departure_mean = departure.mean() if len(departure) else None

    arrival_pct = (arrival.gt(0).mean() * 100) if len(arrival) else None
    departure_pct = (departure.gt(0).mean() * 100) if len(departure) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Mean Arrival Delay",
        f"{arrival_mean / 60:.2f} min" if arrival_mean is not None else "N/A"
    )
    c2.metric(
        "Mean Departure Delay",
        f"{departure_mean / 60:.2f} min" if departure_mean is not None else "N/A"
    )
    c3.metric(
        "Arrival Records Delayed",
        f"{arrival_pct:.1f}%" if arrival_pct is not None else "N/A"
    )
    c4.metric(
        "Departure Records Delayed",
        f"{departure_pct:.1f}%" if departure_pct is not None else "N/A"
    )

    st.divider()

    st.subheader("1. Delay Distribution")

    arrival_plot = pd.DataFrame({
        "Delay (minutes)": arrival / 60,
        "Type": "Arrival"
    })
    departure_plot = pd.DataFrame({
        "Delay (minutes)": departure / 60,
        "Type": "Departure"
    })
    distribution = pd.concat(
        [arrival_plot, departure_plot],
        ignore_index=True
    )

    if not distribution.empty:
        fig = px.histogram(
            distribution,
            x="Delay (minutes)",
            color="Type",
            barmode="overlay",
            opacity=0.6,
            nbins=50,
            title="Arrival and Departure Delay Distribution"
        )
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. Arrival vs Departure: Box Plot")

    if not distribution.empty:
        fig = px.box(
            distribution,
            x="Delay (minutes)",
            y="Type",
            color="Type",
            points="outliers",
            title="Delay Spread and Outliers"
        )
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. Average Delay Across Snapshots")

    trend = data.groupby("collected_at").agg(
        Arrival=("arrival_delay_s", "mean"),
        Departure=("departure_delay_s", "mean")
    ).reset_index()

    trend_long = trend.melt(
        id_vars="collected_at",
        var_name="Type",
        value_name="Average Delay (seconds)"
    )

    fig = px.line(
        trend_long,
        x="collected_at",
        y="Average Delay (seconds)",
        color="Type",
        markers=True,
        title="Average Delay per Collection Snapshot"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("4. Delay Categories")

    bins = [-float("inf"), 0, 60, 300, 600, 1800, float("inf")]
    labels = [
        "Early or on time",
        "1–60 seconds",
        "1–5 minutes",
        "5–10 minutes",
        "10–30 minutes",
        "More than 30 minutes"
    ]

    category_frames = []

    for column, name in [
        ("arrival_delay_s", "Arrival"),
        ("departure_delay_s", "Departure")
    ]:
        values = data[column].dropna()
        categories = pd.cut(
            values,
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True
        )
        counts = categories.value_counts(sort=False)

        category_frames.append(pd.DataFrame({
            "Category": labels,
            "Records": counts.reindex(labels, fill_value=0).values,
            "Type": name
        }))

    category_data = pd.concat(category_frames, ignore_index=True)

    fig = px.bar(
        category_data,
        x="Category",
        y="Records",
        color="Type",
        barmode="group",
        title="Records by Delay Category"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.header("💡 Automatic Insights")

    if arrival_pct is not None:
        st.write(
            f"- **{arrival_pct:.1f}%** of available arrival-delay "
            "records have a positive delay."
        )

    if departure_pct is not None:
        st.write(
            f"- **{departure_pct:.1f}%** of available departure-delay "
            "records have a positive delay."
        )

    if arrival_mean is not None and departure_mean is not None:
        difference = arrival_mean - departure_mean

        if difference > 0:
            st.write(
                f"- Mean arrival delay is **{difference:.1f} seconds "
                "higher** than mean departure delay."
            )
        elif difference < 0:
            st.write(
                f"- Mean departure delay is **{abs(difference):.1f} seconds "
                "higher** than mean arrival delay."
            )
        else:
            st.write("- Mean arrival and departure delays are equal.")

    if not trend.empty:
        combined = trend[["Arrival", "Departure"]].mean(axis=1)
        peak_index = combined.idxmax()
        peak_time = trend.loc[peak_index, "collected_at"]
        st.write(
            "- The highest combined average delay in the selected "
            f"snapshots occurred at **{peak_time.strftime('%H:%M:%S UTC')}**."
        )

    st.caption(
        "Insights describe the selected observations only. "
        "They do not explain the causes of delays."
    )

with quality_tab:
    st.header("Data Quality")

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Rows", f"{original_rows:,}")
    c2.metric("Duplicate Rows", f"{duplicates:,}")
    c3.metric("Rows After Deduplication", f"{len(df):,}")

    quality = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": [df[col].isna().sum() for col in df.columns],
        "Missing Percentage": [
            df[col].isna().mean() * 100 for col in df.columns
        ]
    }).sort_values("Missing Percentage", ascending=False)

    st.subheader("Missing Values by Column")

    fig = px.bar(
        quality,
        x="Missing Percentage",
        y="Column",
        orientation="h",
        text="Missing Percentage",
        title="Missing Values Percentage"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(quality, use_container_width=True, hide_index=True)

    st.info(
        "Missing delay values are not treated as zero. "
        "They may indicate that a realtime delay value was unavailable."
    )
