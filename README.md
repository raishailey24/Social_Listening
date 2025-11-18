# eBay Social Listening Dashboard

An advanced, interactive Streamlit dashboard for analyzing social media trends and identifying emerging opportunities for eBay sellers.

## Features

### 🔍 **Data Overview**
- Automatic column profiling and schema detection
- Interactive column explorer with distributions and statistics
- Handles numeric, datetime, and categorical data types

### 🎯 **Theme Discovery**
- AI-powered theme clustering using sentence embeddings
- Intelligent theme naming using existing metadata (`themes`, `contextual_drivers`)
- Interactive treemap and bar charts of trending themes
- Adjustable cluster count (3-20 themes)

### 📊 **Theme Statistics**
- Deep dive into individual themes
- Sentiment analysis and distribution
- Source platform breakdown
- Word clouds and keyword extraction
- High-engagement post samples

### 📈 **Weekly Impact Analysis**
- Time-series tracking of theme evolution
- Week-over-week velocity calculations
- Sentiment momentum tracking
- Interactive heatmaps of theme activity

### 🎯 **Opportunity Recommendations**
- Automated seller opportunity classification:
  - **Trending – notify vetted sellers**: High growth + positive sentiment
  - **Emerging – watchlist**: Early signals of growth
  - **Declining – low priority**: Weakening trends
- Business-ready insights and rationale

### 🔧 **Advanced Filtering**
- Filter by source platform (Reddit, TikTok, etc.)
- Asset type filtering (posts, comments, videos)
- Language and purchase stage filters
- Date range selection
- Real-time filter impact tracking

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your data:**
   - Put `sample_data.csv` in the project folder, OR
   - Use the sidebar uploader for any CSV file

3. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **Navigate through pages:**
   - **Data Overview**: Explore your dataset schema
   - **Theme Discovery**: Find trending topics
   - **Theme Statistics**: Deep dive analysis
   - **Weekly Impact**: Track evolution over time
   - **Opportunity Recommendations**: Get seller insights

## Data Requirements

The dashboard works with any CSV containing social media data. It automatically detects:

- **Text columns**: `asset_summary`, `combined_text`, `text`, `content`, etc.
- **Date columns**: `Date`, `published_at`, `asset_date`, etc.
- **Sentiment**: Uses existing `sentiment` column or computes with TextBlob
- **Sources**: `source`, `platform` columns for breakdown analysis
- **Engagement**: `score`, `upvotes`, `like_count`, `relevance_to_use`, etc.

### Expected Schema (sample_data.csv format)
```
video_id, source, asset_type, asset_title, Date, asset_summary, 
themes, contextual_drivers, sentiment, language, purchase_stage, 
relevance_to_use, [other columns...]
```

## Technical Architecture

- **Frontend**: Streamlit with Plotly visualizations
- **NLP**: SentenceTransformers (all-MiniLM-L6-v2) + scikit-learn KMeans
- **Sentiment**: TextBlob with existing data integration
- **Caching**: Streamlit cache for embeddings and computations
- **Filtering**: Real-time data filtering with automatic recomputation

## Business Logic

### Theme Classification Rules
- **Trending**: >50% WoW growth + non-negative sentiment momentum
- **Emerging**: 20-50% growth OR positive sentiment shift + volume ≥3
- **Declining**: Negative growth + negative sentiment momentum
- **Stable**: Steady patterns without strong signals

### Adaptive Features
- Cluster count adapts to data size (min 2, max 20, default 8)
- Theme naming prioritizes existing metadata over text analysis
- Engagement metrics auto-detected from multiple column patterns
- Date handling with multiple format support and synthetic fallbacks

## Customization

The dashboard is designed to work out-of-the-box but can be customized:

1. **Business Rules**: Modify `classify_theme_opportunity()` thresholds
2. **Theme Naming**: Adjust priority in `compute_embeddings_and_clusters()`
3. **Filters**: Add new filter types in `apply_filters()`
4. **Visualizations**: Extend plotting functions for new chart types

## Performance Notes

- First run downloads the sentence transformer model (~90MB)
- Embeddings are cached per dataset and cluster count
- Filtering recomputes themes but preserves embeddings when possible
- Recommended max dataset size: 10K rows for real-time performance

## Troubleshooting

**Model download issues**: Ensure internet connection for first-time setup
**Memory errors**: Reduce cluster count or filter data size
**Empty results**: Check filter settings and data quality
**Slow performance**: Consider sampling large datasets before upload

---

Built for eBay business stakeholders to identify trending collectibles, seasonal items, and emerging market opportunities from social media conversations.
