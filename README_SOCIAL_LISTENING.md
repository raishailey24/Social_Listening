# Social Listening Dashboard

A comprehensive Streamlit dashboard for analyzing social media conversations, market trends, and community sentiment across collectible categories.

## 🎯 Overview

This Social Listening dashboard provides real-time insights into community conversations and market trends. It aggregates data from social media discussions, forum posts, and marketplace activities to reveal what people are talking about, buying, and selling in the collectibles market.

## ✨ Features

### 📊 **Executive Overview**
- High-level KPIs and trending themes
- Real-time sentiment metrics
- Week-over-week growth tracking
- Top trending themes with dynamic filtering

### 🔥 **Trend Deep Dive**
- Detailed analysis of trending themes and sentiment
- Bubble charts with solid color visualization
- Action priority table with recommendation filters
- Comprehensive trend scoring system

### 💭 **Sentiment Journey**
- Multi-theme sentiment timeline analysis
- All themes displayed by default with selective filtering
- Sentiment distribution across categories
- Community sentiment tracking over time

### 💰 **Seller Opportunities**
- Identify high-value opportunities for sellers
- Recommendation-based filtering system
- Opportunity scoring and prioritization

### 📦 **Category Performance**
- Performance analysis across different categories
- Year-based filtering for trend analysis
- Properly capitalized category displays

### 📖 **Theme Decoder**
- Enhanced theme explanations and meanings
- Business-friendly interpretations
- Seller action recommendations

## 🚀 Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/raishailey24/Social_Listening.git
   cd Social_Listening
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run ebay_trend_intelligence_dashboard.py
   ```

4. **Access the dashboard:**
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
Social_Listening/
├── ebay_trend_intelligence_dashboard.py  # Main dashboard application
├── app.py                               # Alternative entry point
├── requirements.txt                     # Python dependencies
├── sample_data.csv                      # Sample dataset
├── README.md                           # This file
├── README_EBAY_DASHBOARD.md            # Detailed dashboard documentation
└── ENHANCEMENT_SUMMARY.md              # Feature enhancement history
```

## 📊 Data Requirements

The dashboard expects the following CSV files in the specified directory:
- `CLEAN_theme_dimension_ENHANCED.csv`
- `CLEAN_social_listening_fact_table.csv`
- `CLEAN_theme_weekly_trends_TOP20.csv`
- `CLEAN_weekly_category_aggregation.csv`
- `CLEAN_weekly_totals.csv`
- `CLEAN_product_performance_ENHANCED.csv`
- `CLEAN_specific_themes_only.csv`
- `THEME_DECODER_REFERENCE.csv`

## 🎨 Key Features

### **Enhanced Visualizations**
- ✅ Solid color schemes for better visibility
- ✅ Professional category capitalization
- ✅ Dynamic filtering and multi-selection
- ✅ Responsive chart layouts

### **Advanced Analytics**
- ✅ Real-time sentiment analysis
- ✅ Trend scoring algorithms
- ✅ Opportunity identification
- ✅ Multi-dimensional filtering

### **User Experience**
- ✅ Intuitive navigation
- ✅ Comprehensive explanations
- ✅ Professional branding
- ✅ Mobile-responsive design

## 🔧 Configuration

Update the `base_path` variable in the `load_data()` function to point to your data directory:

```python
base_path = "path/to/your/data/directory"
```

## 📈 Usage

1. **Executive Overview**: Start here for high-level insights and trending themes
2. **Trend Deep Dive**: Analyze specific trends and sentiment patterns
3. **Sentiment Journey**: Track how community feelings evolve over time
4. **Seller Opportunities**: Identify actionable business opportunities
5. **Category Performance**: Compare performance across different categories
6. **Theme Decoder**: Understand what themes mean for your business

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

**Rai Shailey** - raishailey24@gmail.com

Project Link: [https://github.com/raishailey24/Social_Listening](https://github.com/raishailey24/Social_Listening)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/)

---

**Social Listening Intelligence Platform** | *Data updated: November 2025*
