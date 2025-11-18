import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dateutil import parser
from datetime import datetime

# ----------------------
# Caching & Models
# ----------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    # Lightweight general-purpose model; can be swapped if needed
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def load_data_from_upload(file):
    df = pd.read_csv(file)
    return df


def coalesce_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        c for c in df.columns
        if c.lower() in ["date", "published_at"] or "date" in c.lower() or "time" in c.lower()
    ]
    dt = None
    for c in date_cols:
        try:
            tmp = pd.to_datetime(df[c], errors="coerce")
            if tmp.notna().sum() > 0:
                dt = tmp
                break
        except Exception:
            continue
    if dt is None:
        # fallback: create synthetic date if missing
        dt = pd.Series(pd.date_range(end=pd.Timestamp.today(), periods=len(df)))
    df = df.copy()
    df["_datetime"] = dt
    df["week"] = df["_datetime"].dt.to_period("W").apply(lambda r: r.start_time)
    return df


def get_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "asset_summary", "combined_text", "text", "body", "content", "title", "asset_title"
    ]
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    # fallback: first object column
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    return obj_cols[0] if obj_cols else df.columns[0]


def get_source_column(df: pd.DataFrame):
    for col in df.columns:
        if col.lower() in ["source", "platform"]:
            return col
    return None


def apply_filters(df: pd.DataFrame):
    """Apply sidebar filters to the dataframe"""
    df_filtered = df.copy()
    
    # Source filter
    if "source" in df.columns:
        sources = sorted(df["source"].dropna().unique())
        if sources:
            selected_sources = st.sidebar.multiselect("Filter by Source", sources, default=sources)
            if selected_sources:
                df_filtered = df_filtered[df_filtered["source"].isin(selected_sources)]
    
    # Asset type filter
    if "asset_type" in df.columns:
        asset_types = sorted(df["asset_type"].dropna().unique())
        if asset_types:
            selected_types = st.sidebar.multiselect("Filter by Asset Type", asset_types, default=asset_types)
            if selected_types:
                df_filtered = df_filtered[df_filtered["asset_type"].isin(selected_types)]
    
    # Language filter
    if "language" in df.columns:
        languages = sorted(df["language"].dropna().unique())
        if languages:
            selected_langs = st.sidebar.multiselect("Filter by Language", languages, default=languages)
            if selected_langs:
                df_filtered = df_filtered[df_filtered["language"].isin(selected_langs)]
    
    # Purchase stage filter
    if "purchase_stage" in df.columns:
        stages = sorted(df["purchase_stage"].dropna().unique())
        if stages:
            selected_stages = st.sidebar.multiselect("Filter by Purchase Stage", stages, default=stages)
            if selected_stages:
                df_filtered = df_filtered[df_filtered["purchase_stage"].isin(selected_stages)]
    
    # Date range filter
    if "Date" in df.columns:
        df_filtered["Date"] = pd.to_datetime(df_filtered["Date"], errors="coerce")
        min_date = df_filtered["Date"].min()
        max_date = df_filtered["Date"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df_filtered[
                    (df_filtered["Date"].dt.date >= start_date) & 
                    (df_filtered["Date"].dt.date <= end_date)
                ]
    
    return df_filtered


@st.cache_data(show_spinner=False)
def compute_embeddings_and_clusters(df: pd.DataFrame, text_col: str, n_clusters: int):
    model = load_embedding_model()
    texts = df[text_col].astype(str).fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    df_out = df.copy()
    df_out["theme_id"] = labels
    
    # Enhanced theme naming using multiple sources
    theme_names = {}
    for k in range(n_clusters):
        cluster_df = df_out[df_out["theme_id"] == k]
        
        # Try to use existing themes column if available
        if "themes" in cluster_df.columns:
            existing_themes = cluster_df["themes"].dropna().astype(str)
            if len(existing_themes) > 0:
                theme_words = []
                for themes_str in existing_themes:
                    theme_words.extend(themes_str.split(";"))
                theme_counter = pd.Series(theme_words).value_counts()
                if len(theme_counter) > 0:
                    theme_names[k] = theme_counter.index[0].strip()
                    continue
        
        # Try contextual_drivers if themes not available
        if "contextual_drivers" in cluster_df.columns:
            drivers = cluster_df["contextual_drivers"].dropna().astype(str)
            if len(drivers) > 0:
                driver_words = []
                for driver_str in drivers:
                    driver_words.extend(driver_str.split(";"))
                driver_counter = pd.Series(driver_words).value_counts()
                if len(driver_counter) > 0:
                    theme_names[k] = driver_counter.index[0].strip()
                    continue
        
        # Fallback to text analysis
        cluster_text = " ".join(cluster_df[text_col].astype(str).tolist())
        words = pd.Series(cluster_text.lower().split())
        words = words[~words.isin(["the", "and", "or", "to", "of", "a", "is", "it", "this", "that", "card", "cards"])]
        top_words = words.value_counts().head(3).index.tolist()
        theme_names[k] = ", ".join(top_words) if top_words else f"Theme {k}"
    
    df_out["theme_name"] = df_out["theme_id"].map(theme_names)
    return df_out, theme_names


@st.cache_data(show_spinner=False)
def compute_sentiment(df: pd.DataFrame, text_col: str):
    df_out = df.copy()

    # If dataset already has a sentiment column, map it to numeric and standardized labels
    existing_sent_col = None
    for c in df_out.columns:
        if c.lower() == "sentiment":
            existing_sent_col = c
            break

    if existing_sent_col is not None:
        raw = df_out[existing_sent_col].astype(str).str.lower().fillna("")
        label_map = {
            "positive": "Positive",
            "pos": "Positive",
            "negative": "Negative",
            "neg": "Negative",
            "neutral": "Neutral",
        }
        df_out["sentiment_label"] = raw.map(label_map).fillna("Neutral")
        score_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
        df_out["sentiment_score"] = df_out["sentiment_label"].map(score_map).astype(float)
        return df_out

    sentiments = []
    labels = []
    for text in df[text_col].astype(str).fillna(""):
        tb = TextBlob(text)
        score = tb.sentiment.polarity
        sentiments.append(score)
        if score > 0.1:
            labels.append("Positive")
        elif score < -0.1:
            labels.append("Negative")
        else:
            labels.append("Neutral")
    df_out["sentiment_score"] = sentiments
    df_out["sentiment_label"] = labels
    return df_out


def page_data_overview(df: pd.DataFrame):
    st.header("Data Overview – Column Profiling")
    st.markdown("Automatically profile every column: types, missingness, distributions, and example values.")

    st.subheader("Schema Snapshot")
    summary_rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        non_null = series.notna().sum()
        n_unique = series.nunique(dropna=True)
        missing = series.isna().sum()
        example = series.dropna().astype(str).head(3).tolist()
        summary_rows.append({
            "column": col,
            "dtype": dtype,
            "non_null": int(non_null),
            "missing": int(missing),
            "unique_values": int(n_unique),
            "examples": "; ".join(example),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    st.subheader("Column Explorer")
    col_selected = st.selectbox("Select a column to inspect", df.columns)
    s = df[col_selected]
    st.write(f"**Dtype:** {s.dtype}")
    st.write(f"**Non-null:** {s.notna().sum()} | **Missing:** {s.isna().sum()} | **Unique:** {s.nunique(dropna=True)}")

    if pd.api.types.is_numeric_dtype(s):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Summary statistics**")
            st.write(s.describe())
        with col2:
            fig = px.histogram(df, x=col_selected, nbins=30, title=f"Distribution of {col_selected}")
            st.plotly_chart(fig, use_container_width=True)
    elif pd.api.types.is_datetime64_any_dtype(s):
        s_dt = pd.to_datetime(s, errors="coerce")
        ts = s_dt.value_counts().sort_index().reset_index()
        ts.columns = [col_selected, "count"]
        fig = px.line(ts, x=col_selected, y="count", title=f"Time series of {col_selected}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # treat as categorical/text
        value_counts = s.astype(str).value_counts().head(30).reset_index()
        value_counts.columns = ["value", "count"]
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top values**")
            st.dataframe(value_counts, use_container_width=True)
        with col2:
            fig = px.bar(value_counts, x="value", y="count", title=f"Top values of {col_selected}")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


def compute_weekly_metrics(df: pd.DataFrame):
    # theme-week frequency and sentiment
    grp = df.groupby(["theme_name", "week"]).agg(
        volume=("theme_name", "size"),
        avg_sentiment=("sentiment_score", "mean"),
    ).reset_index()
    # velocity: week-over-week change in volume (percent) for the latest 2 weeks
    latest_weeks = sorted(df["week"].dropna().unique())
    metrics = {}
    if len(latest_weeks) >= 2:
        w_latest, w_prev = latest_weeks[-1], latest_weeks[-2]
        latest = grp[grp["week"] == w_latest].set_index("theme_name")
        prev = grp[grp["week"] == w_prev].set_index("theme_name")
        all_themes = sorted(set(latest.index) | set(prev.index))
        for t in all_themes:
            vol_latest = latest.loc[t, "volume"] if t in latest.index else 0
            vol_prev = prev.loc[t, "volume"] if t in prev.index else 0
            if vol_prev == 0:
                velocity = np.inf if vol_latest > 0 else 0
            else:
                velocity = (vol_latest - vol_prev) / vol_prev
            sent_latest = latest.loc[t, "avg_sentiment"] if t in latest.index else 0
            sent_prev = prev.loc[t, "avg_sentiment"] if t in prev.index else 0
            sentiment_momentum = sent_latest - sent_prev
            metrics[t] = {
                "velocity": velocity,
                "sentiment_momentum": sentiment_momentum,
                "vol_latest": vol_latest,
                "vol_prev": vol_prev,
            }
    return grp, metrics


def classify_theme_opportunity(theme_name: str, metrics: dict):
    m = metrics.get(theme_name)
    if not m:
        return "Unknown", "Insufficient data"
    velocity = m["velocity"]
    sentiment_momentum = m["sentiment_momentum"]
    vol_latest = m["vol_latest"]
    # business rules (can be tuned)
    if vol_latest < 5 and (velocity == 0 or np.isfinite(velocity) and velocity < 0.2):
        return "Declining – low priority", "Low recent volume and weak growth."
    if (np.isinf(velocity) or velocity > 0.5) and sentiment_momentum >= 0:
        return "Trending – notify vetted sellers", "Strong week-over-week growth with stable or improving sentiment."
    if (0.2 <= velocity <= 0.5) or (sentiment_momentum > 0.05 and vol_latest >= 3):
        return "Emerging – watchlist", "Early signs of increasing conversation or sentiment."
    if velocity < 0 and sentiment_momentum <= 0:
        return "Declining – low priority", "Volume and sentiment are both softening."
    return "Stable", "Theme is relatively steady without strong momentum signals."


def plot_theme_overview(df: pd.DataFrame):
    theme_counts = df["theme_name"].value_counts().reset_index()
    theme_counts.columns = ["theme_name", "count"]
    fig_tree = px.treemap(
        theme_counts,
        path=["theme_name"],
        values="count",
        color="count",
        color_continuous_scale="Blues",
    )
    fig_bar = px.bar(
        theme_counts.head(20),
        x="theme_name",
        y="count",
        title="Top Themes by Volume",
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    return fig_tree, fig_bar, theme_counts


def plot_theme_deep_dive(df_theme: pd.DataFrame, source_col: str | None):
    # Sentiment distribution
    sent_counts = df_theme["sentiment_label"].value_counts().reset_index()
    sent_counts.columns = ["sentiment_label", "count"]
    fig_sent = px.pie(
        sent_counts,
        names="sentiment_label",
        values="count",
        hole=0.4,
        title="Sentiment Distribution",
    )

    # Source distribution
    fig_source = None
    if source_col and source_col in df_theme.columns:
        src_counts = df_theme[source_col].value_counts().reset_index()
        src_counts.columns = ["source", "count"]
        fig_source = px.bar(
            src_counts,
            x="source",
            y="count",
            title="Source Distribution",
        )

    return fig_sent, fig_source


def generate_wordcloud(df_theme: pd.DataFrame, text_col: str):
    text = " ".join(df_theme[text_col].astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def page_theme_discovery(df: pd.DataFrame, text_col: str):
    st.header("Theme Discovery – Overview")
    st.markdown("Discover emerging and trending themes across social platforms.")

    n_clusters = st.sidebar.slider("Number of themes (clusters)", min_value=3, max_value=20, value=8)
    df_clustered, theme_names = compute_embeddings_and_clusters(df, text_col, n_clusters)
    df_clustered = compute_sentiment(df_clustered, text_col)
    df_clustered = coalesce_date_columns(df_clustered)

    fig_tree, fig_bar, theme_counts = plot_theme_overview(df_clustered)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Trending Themes – Treemap")
        st.plotly_chart(fig_tree, use_container_width=True)
    with col2:
        st.subheader("Theme Frequency")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Top 5 Themes – Quick Insight")
    top5 = theme_counts.head(5)
    for _, row in top5.iterrows():
        theme = row["theme_name"]
        count = row["count"]
        st.markdown(f"**{theme}** – {count} mentions")

    return df_clustered


def get_engagement_metrics(df: pd.DataFrame):
    """Detect engagement-like columns and return the best one"""
    engagement_candidates = [
        "score", "upvotes", "engagement", "like_count", "comment_count", 
        "views", "shares", "interactions", "relevance_to_use"
    ]
    for candidate in engagement_candidates:
        for col in df.columns:
            if col.lower() == candidate.lower():
                return col
    return None


def page_theme_statistics(df: pd.DataFrame, text_col: str):
    st.header("Theme Statistics – Deep Dive")
    st.markdown("Drill into a specific theme to understand sentiment, sources, and example posts.")

    source_col = get_source_column(df)
    engagement_col = get_engagement_metrics(df)
    themes = sorted(df["theme_name"].unique())
    theme_selected = st.selectbox("Select a theme", themes)
    df_theme = df[df["theme_name"] == theme_selected]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total posts", len(df_theme))
    with col2:
        st.metric("Avg sentiment", f"{df_theme['sentiment_score'].mean():.2f}")
    with col3:
        st.metric("Time span", f"{df_theme['_datetime'].min().date()} to {df_theme['_datetime'].max().date()}")
    with col4:
        if engagement_col and engagement_col in df_theme.columns:
            try:
                avg_engagement = pd.to_numeric(df_theme[engagement_col], errors="coerce").mean()
                st.metric(f"Avg {engagement_col}", f"{avg_engagement:.2f}")
            except:
                st.metric("Engagement", "N/A")
        else:
            st.metric("Engagement", "N/A")

    fig_sent, fig_source = plot_theme_deep_dive(df_theme, source_col)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Distribution")
        st.plotly_chart(fig_sent, use_container_width=True)
    with col2:
        st.subheader("Source Distribution")
        if fig_source is not None:
            st.plotly_chart(fig_source, use_container_width=True)
        else:
            st.info("No explicit source column found.")

    st.subheader("Top Keywords & Phrases (Word Cloud)")
    fig_wc = generate_wordcloud(df_theme, text_col)
    st.pyplot(fig_wc, clear_figure=True)

    st.subheader("Sample High-Engagement Posts")
    engagement_col = get_engagement_metrics(df_theme)
    if engagement_col and engagement_col in df_theme.columns:
        try:
            df_theme[f"{engagement_col}_numeric"] = pd.to_numeric(df_theme[engagement_col], errors="coerce")
            df_sample = df_theme.sort_values(f"{engagement_col}_numeric", ascending=False, na_position="last").head(10)
        except:
            df_sample = df_theme.sample(min(10, len(df_theme))) if len(df_theme) > 0 else df_theme
    else:
        df_sample = df_theme.sample(min(10, len(df_theme))) if len(df_theme) > 0 else df_theme

    # Show key columns for readability
    display_cols = []
    priority_cols = ["asset_title", "asset_summary", "source", "Date", "sentiment_label", "themes", "contextual_drivers"]
    for col in priority_cols:
        if col in df_sample.columns:
            display_cols.append(col)
    
    # Add engagement column if available
    if engagement_col and engagement_col in df_sample.columns:
        display_cols.append(engagement_col)
    
    # Fill remaining space with other interesting columns
    other_cols = [c for c in df_sample.columns if c not in display_cols and c not in ["sentiment_score", "theme_id", "_datetime", "week"]]
    display_cols.extend(other_cols[:3])  # Add up to 3 more columns
    
    st.dataframe(df_sample[display_cols].head(10), use_container_width=True)


def page_weekly_impact(df: pd.DataFrame):
    st.header("Weekly Theme Impact & Sentiment Movement")
    st.markdown("Track how each theme evolves week by week.")

    weekly, metrics = compute_weekly_metrics(df)
    themes = sorted(weekly["theme_name"].unique())
    theme_selected = st.selectbox("Select a theme", themes)
    df_theme_week = weekly[weekly["theme_name"] == theme_selected]

    col1, col2 = st.columns(2)
    with col1:
        fig_vol = px.line(
            df_theme_week,
            x="week",
            y="volume",
            markers=True,
            title="Weekly Volume",
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    with col2:
        fig_sent = px.line(
            df_theme_week,
            x="week",
            y="avg_sentiment",
            markers=True,
            title="Weekly Avg Sentiment",
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    st.subheader("Theme × Week Activity Heatmap")
    heat = weekly.pivot_table(index="theme_name", columns="week", values="volume", fill_value=0)
    fig_heat = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(color="Volume"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("KPI Flags")
    m = metrics.get(theme_selected)
    if m:
        velocity_pct = "∞" if np.isinf(m["velocity"]) else f"{m['velocity']*100:.1f}%"
        st.metric("WoW volume change", velocity_pct)
        st.metric("Sentiment momentum", f"{m['sentiment_momentum']:+.2f}")
    else:
        st.info("Not enough data to compute week-over-week metrics.")


def page_recommendations(df: pd.DataFrame):
    st.header("Opportunity Recommendations – eBay Seller Enablement")
    st.markdown("Identify which themes should be surfaced to vetted sellers.")

    weekly, metrics = compute_weekly_metrics(df)
    theme_latest = weekly.sort_values("week").groupby("theme_name").tail(1)

    rows = []
    for _, row in theme_latest.iterrows():
        theme = row["theme_name"]
        status, rationale = classify_theme_opportunity(theme, metrics)
        m = metrics.get(theme, {})
        velocity = m.get("velocity", 0)
        velocity_pct = "∞" if np.isinf(velocity) else f"{velocity*100:.1f}%"
        rows.append(
            {
                "Theme": theme,
                "Latest Week Volume": row["volume"],
                "Latest Avg Sentiment": f"{row['avg_sentiment']:.2f}",
                "WoW Velocity": velocity_pct,
                "Sentiment Momentum": f"{m.get('sentiment_momentum', 0):+.2f}",
                "Recommendation": status,
                "Rationale": rationale,
            }
        )

    rec_df = pd.DataFrame(rows)

    # priority ordering
    priority_order = [
        "Trending – notify vetted sellers",
        "Emerging – watchlist",
        "Stable",
        "Declining – low priority",
        "Unknown",
    ]
    rec_df["_priority_rank"] = rec_df["Recommendation"].apply(lambda x: priority_order.index(x) if x in priority_order else len(priority_order))
    rec_df = rec_df.sort_values(["_priority_rank", "Latest Week Volume"], ascending=[True, False])

    st.subheader("Automated Insights")
    for _, r in rec_df.head(10).iterrows():
        if r["Recommendation"] == "Trending – notify vetted sellers":
            st.markdown(
                f"**{r['Theme']}** has grown {r['WoW Velocity']} in volume recently with positive sentiment ({r['Latest Avg Sentiment']}). **Recommend notifying vetted sellers.**"
            )
        elif r["Recommendation"] == "Emerging – watchlist":
            st.markdown(
                f"Early chatter around **{r['Theme']}** with {r['Latest Week Volume']} mentions and sentiment {r['Latest Avg Sentiment']}. **Add to watchlist.**"
            )
        elif r["Recommendation"] == "Declining – low priority":
            st.markdown(
                f"Conversation around **{r['Theme']}** is softening. {r['Latest Week Volume']} recent mentions, sentiment {r['Latest Avg Sentiment']}. **Low urgency.**"
            )
        else:
            st.markdown(
                f"**{r['Theme']}** – {r['Latest Week Volume']} recent mentions. Sentiment {r['Latest Avg Sentiment']}. Status: {r['Recommendation']}."
            )

    st.subheader("Theme Opportunity Table")
    st.dataframe(
        rec_df.drop(columns=["_priority_rank"]),
        use_container_width=True,
        hide_index=True,
    )


def main():
    st.set_page_config(
        page_title="eBay Social Listening – Theme Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Data & Controls")
    st.sidebar.markdown("The app will try to auto-load `sample_data.csv` from the project folder. You can also upload another CSV.")

    default_path = os.path.join(os.getcwd(), "sample_data.csv")
    df_raw = None
    if os.path.exists(default_path):
        try:
            df_raw = pd.read_csv(default_path)
            st.sidebar.success(f"Loaded default dataset: `sample_data.csv` ({len(df_raw):,} rows)")
        except Exception as e:
            st.sidebar.error(f"Failed to load `sample_data.csv`: {e}")

    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        df_raw = load_data_from_upload(uploaded)
        st.sidebar.success(f"Using uploaded dataset ({len(df_raw):,} rows)")

    if df_raw is None:
        st.info("Place `sample_data.csv` in the app folder or upload a CSV to begin exploring themes.")
        return

    text_col = get_text_column(df_raw)
    st.sidebar.markdown(f"**Detected text column:** `{text_col}`")
    
    # Apply filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    df_filtered = apply_filters(df_raw)
    
    if len(df_filtered) != len(df_raw):
        st.sidebar.info(f"Filtered to {len(df_filtered):,} rows (from {len(df_raw):,})")
    
    # Use filtered data for all downstream processing
    df_raw = df_filtered

    # Pipeline shared across pages
    page = st.sidebar.radio(
        "Navigation",
        [
            "Data Overview",
            "Theme Discovery",
            "Theme Statistics",
            "Weekly Theme Impact",
            "Opportunity Recommendations",
        ],
    )

    # Ensure we compute core fields once for theme-related pages
    if len(df_raw) == 0:
        st.warning("No data remaining after applying filters. Please adjust your filter settings.")
        return
        
    n_clusters_default = min(8, max(2, len(df_raw) // 10))  # Adaptive cluster count
    df_clustered, theme_names = compute_embeddings_and_clusters(df_raw, text_col, n_clusters_default)
    df_clustered = compute_sentiment(df_clustered, text_col)
    df_clustered = coalesce_date_columns(df_clustered)

    # Route pages (Theme Discovery re-runs with its own cluster slider for fine tuning)
    if page == "Data Overview":
        page_data_overview(df_raw)
    elif page == "Theme Discovery":
        page_theme_discovery(df_raw, text_col)
    elif page == "Theme Statistics":
        page_theme_statistics(df_clustered, text_col)
    elif page == "Weekly Theme Impact":
        page_weekly_impact(df_clustered)
    elif page == "Opportunity Recommendations":
        page_recommendations(df_clustered)


if __name__ == "__main__":
    main()
