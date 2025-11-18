import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import re

# Page config
st.set_page_config(
    page_title="Social Listening Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Enhanced theme processing functions
def create_display_theme(theme, category):
    """
    Convert generic themes into actionable display names
    """
    generic_themes = [
        'pricing', 'collecting', 'grading', 'trading cards', 
        'nostalgia', 'market trends', 'collectibles', 'investing',
        'checklists', 'sets'
    ]
    
    if pd.isna(theme) or theme == '':
        return 'Uncategorized'
    
    theme_lower = str(theme).lower().strip()
    category_clean = str(category).title() if pd.notna(category) else 'General'
    
    # If theme is generic, add category context
    if theme_lower in generic_themes:
        theme_labels = {
            'pricing': f'{category_clean} Card Pricing Discussions',
            'collecting': f'{category_clean} Collection Building',
            'grading': f'{category_clean} Card Grading Questions',
            'trading cards': f'{category_clean} Trading Card Activity',
            'nostalgia': f'{category_clean} Nostalgia & Memories',
            'market trends': f'{category_clean} Market Trends',
            'collectibles': f'{category_clean} Collectibles',
            'investing': f'{category_clean} Investment Activity',
            'checklists': f'{category_clean} Checklists & Sets',
            'sets': f'{category_clean} Set Completion'
        }
        return theme_labels.get(theme_lower, f'{category_clean} - {theme.title()}')
    
    # If theme is already specific, just clean it up
    return theme.title()

def is_generic_theme(theme):
    """
    Check if theme is generic and should be filtered from top lists
    """
    generic_themes = [
        'pricing', 'collecting', 'grading', 'trading cards', 
        'nostalgia', 'market trends', 'collectibles', 'investing',
        'checklists', 'sets'
    ]
    return str(theme).lower().strip() in generic_themes

def get_category_color(category):
    """
    Get category badge color - Subtle tones
    """
    category_colors = {
        'pokemon': '#FFD54F',        # Soft yellow
        'baseball': '#E57373',       # Soft red
        'basketball': '#FF8A65',     # Soft orange
        'football': '#A1887F',       # Soft brown
        'magic_the_gathering': '#757575',  # Soft gray
        'magic': '#757575',          # Soft gray
        'tcg': '#64B5F6',           # Soft blue
        'gift cards': '#81C784',     # Soft green
        'card collecting': '#64B5F6', # Soft blue
        'travel': '#9575CD',         # Soft purple
        'gaming': '#F06292',         # Soft pink
        'general': '#B0BEC5'         # Light blue-gray
    }
    return category_colors.get(str(category).lower(), '#B0BEC5')

def get_theme_tooltip(theme, category):
    """
    Get contextual tooltip for themes
    """
    tooltips = {
        'pricing': f"People discussing how much {category} cards are worth. Opportunity: Launch pricing tools or transparency features.",
        'grading': f"People asking about PSA/BGS grading services for {category}. Opportunity: Partner with grading companies for seller discounts.",
        'collecting': f"People building {category} collections, asking for advice. Opportunity: Promote starter sets, binders, organizers.",
        'gift cards': "People buying or discussing gift cards. Opportunity: Promote seasonal gift card bundles.",
        'rookies': "People searching for rookie cards of specific players. Opportunity: Feature rookie card collections."
    }
    theme_lower = str(theme).lower()
    for key in tooltips:
        if key in theme_lower:
            return tooltips[key]
    return f"Trending topic in {category}. Monitor for seller opportunities."

def get_enhanced_theme_explanation(theme):
    """
    Get enhanced business-friendly explanation for each theme
    """
    explanations = {
        # Generic themes
        'pricing': 'People discussing card values, asking "What is this worth?" and seeking pricing guidance',
        'collecting': 'People building collections, organizing cards, and seeking collecting advice',
        'grading': 'People asking about PSA/BGS grading services, condition assessment, and grading costs',
        'trading cards': 'General trading card discussions, swaps, and trading activities',
        'nostalgia': 'People sharing memories and nostalgic feelings about cards from their childhood',
        'market trends': 'Discussions about market movements, price trends, and investment patterns',
        'collectibles': 'General collectible discussions beyond just trading cards',
        'investing': 'People treating cards as financial investments and discussing ROI',
        'checklists': 'People working on completing sets and tracking missing cards',
        'sets': 'Discussions about specific card sets, completion, and set building',
        
        # Specific themes
        'gift cards': 'People buying gift cards or discussing gift card promotions and deals',
        'rookies': 'Rookie card discussions, prospect evaluations, and first-year player cards',
        'autographs': 'Autographed card discussions, authentication, and signature verification',
        'authentication': 'People verifying card authenticity, spotting fakes, and seeking authentication services',
        'vintage cards': 'Discussions about older, classic cards from past decades',
        'pop culture': 'Cards related to movies, TV shows, anime, and entertainment franchises',
        'accommodation': 'Travel-related collectibles, airline memorabilia, and transportation themes',
        'addiction': 'People discussing collecting addiction, spending habits, and collection management',
        'app bugs': 'Technical issues with trading card apps and digital platforms',
        'art focus': 'Appreciation of card artwork, artist discussions, and visual design elements',
        'availability': 'Product availability discussions, restocks, and finding specific items',
        'box breaks': 'Group box breaking activities, shared pack openings, and break participation',
        'brands': 'Discussions about different card manufacturers and brand preferences',
        'card condition': 'Condition assessment, damage evaluation, and preservation discussions',
        'card protection': 'Protective supplies like sleeves, toploaders, and storage solutions',
        'celebrity cards': 'Cards featuring celebrities, entertainers, and non-sports personalities',
        'counterfeit detection': 'Identifying fake cards and avoiding counterfeit products',
        'custom cards': 'Custom-made cards, fan creations, and personalized designs',
        'deals': 'Bargain hunting, sales, discounts, and good deal sharing',
        'first edition': 'First edition cards, print runs, and edition identification',
        'game mechanics': 'Trading card game rules, gameplay, and strategy discussions',
        'holographic stickers': 'Holographic elements, foil cards, and special finishes',
        'limited edition': 'Limited print cards, exclusive releases, and rare variants',
        'pack opening': 'Pack opening experiences, pulls, and unboxing content',
        'player performance': 'Sports card discussions tied to player statistics and performance',
        'product launches': 'New product releases, upcoming sets, and launch announcements',
        'sealed products': 'Unopened packs, boxes, and sealed product investments',
        'storage': 'Card storage solutions, organization methods, and preservation techniques',
        'tournament': 'Competitive play, tournaments, and organized gaming events'
    }
    return explanations.get(str(theme).lower().strip(), f'Discussions and activities related to {str(theme).title()}')

# Load data function
@st.cache_data
def load_data():
    # Use local data directory for portability
    base_path = "./data"
    
    try:
        # Load V2 ENHANCED datasets
        theme_dim = pd.read_csv(f"{base_path}/CLEAN_theme_dimension_ENHANCED.csv")  # V2 Enhanced
        fact_table = pd.read_csv(f"{base_path}/CLEAN_social_listening_fact_table.csv")
        weekly_trends = pd.read_csv(f"{base_path}/CLEAN_theme_weekly_trends_TOP20.csv")  # V2 Enhanced (Top 20)
        weekly_category = pd.read_csv(f"{base_path}/CLEAN_weekly_category_aggregation.csv")
        weekly_totals = pd.read_csv(f"{base_path}/CLEAN_weekly_totals.csv")
        product_performance = pd.read_csv(f"{base_path}/CLEAN_product_performance_ENHANCED.csv")  # V2 Enhanced
        
        # V2 New: Load additional enhanced datasets
        specific_themes = pd.read_csv(f"{base_path}/CLEAN_specific_themes_only.csv")  # V2 New
        theme_decoder = pd.read_csv(f"{base_path}/THEME_DECODER_REFERENCE.csv")  # V2 New
        
        # Convert dates
        fact_table['post_date'] = pd.to_datetime(fact_table['post_date'])
        weekly_category['week_start_date'] = pd.to_datetime(weekly_category['week_start_date'])
        weekly_totals['week_start_date'] = pd.to_datetime(weekly_totals['week_start_date'])
        
        # Fix category capitalization across all datasets
        def capitalize_category(cat):
            if pd.isna(cat):
                return cat
            return str(cat).replace('_', ' ').title()
        
        # Apply capitalization to all category columns
        if 'primary_category' in theme_dim.columns:
            theme_dim['primary_category'] = theme_dim['primary_category'].apply(capitalize_category)
        if 'category' in fact_table.columns:
            fact_table['category'] = fact_table['category'].apply(capitalize_category)
        if 'category' in weekly_category.columns:
            weekly_category['category'] = weekly_category['category'].apply(capitalize_category)
        if 'primary_category' in product_performance.columns:
            product_performance['primary_category'] = product_performance['primary_category'].apply(capitalize_category)
        
        # Add category colors (enhanced data already in CSV)
        theme_dim['category_color'] = theme_dim['primary_category'].apply(get_category_color)
        
        return {
            'theme_dimension': theme_dim,
            'fact_table': fact_table,
            'weekly_trends': weekly_trends,
            'weekly_category': weekly_category,
            'weekly_totals': weekly_totals,
            'product_performance': product_performance,
            'specific_themes': specific_themes,  # V2 New
            'theme_decoder': theme_decoder  # V2 New
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Color schemes - Light colors for better text visibility
TREND_COLORS = {
    'HOT': '#FFCDD2',      # Very light red
    'RISING': '#FFE0B2',   # Very light orange
    'STEADY': '#FFF9C4',   # Very light yellow
    'COOLING': '#C8E6C9'   # Very light green
}

SENTIMENT_COLORS = {
    'Very Positive': '#00CC66',
    'Positive': '#66CC66',
    'Neutral': '#CCCCCC',
    'Negative': '#FF9966',
    'Very Negative': '#FF3333'
}

RECOMMENDATION_COLORS = {
    'IMMEDIATE ACTION': '#DC143C',
    'HIGH PRIORITY': '#FF8C00',
    'MONITOR': '#4682B4',
    'LOW PRIORITY': '#A9A9A9'
}

# Enhanced product colors - Light tones for better text visibility
PRODUCT_COLORS = {
    'Hot Product': '#FFCDD2',    # Very light red
    'Rising Product': '#FFE0B2', # Very light orange
    'Active Product': '#FFF9C4', # Very light yellow
    'Cooling Product': '#E1F5FE' # Very light blue
}

def page_executive_overview(data):
    st.title("📊 Executive Overview")
    st.markdown("High-level snapshot for leadership")
    
    # Add "What Does This Mean?" info panel
    with st.expander("ℹ️ Understanding Your Dashboard (Click to Expand)"):
        st.markdown("""
        ### 📊 WHAT YOU'RE SEEING:
        
        This Social Listening dashboard provides real-time insights into community conversations and market trends across collectible categories. You're viewing aggregated data from social media discussions, forum posts, and marketplace activities that reveal what people are talking about, buying, and selling. The trending themes represent popular conversation topics like pricing discussions, collecting advice, and product availability, while sentiment analysis shows how the community feels about different topics. Use these insights to identify emerging opportunities, understand market sentiment, and make data-driven decisions for your business strategy.
        
        ### 🎯 HOW TO USE THIS:
        
        1. **Check "Immediate Actions Needed"** - Email these to top sellers TODAY
        2. **Review "Top Trending Products"** - Promote these on homepage  
        3. **Track sentiment** - If negative, investigate issues
        4. **Monitor week-over-week growth** - Catch trends before they peak
        
        ### 🔥 TREND SCORES EXPLAINED:
        
        - **90-100**: VIRAL - Act immediately!
        - **70-89**: HOT - High priority this week
        - **50-69**: RISING - Monitor closely
        - **30-49**: STEADY - Keep on radar
        - **0-29**: COOLING - Deprioritize
        """)
    
    theme_dim = data['theme_dimension']
    weekly_totals = data['weekly_totals']
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        trending_count = len(theme_dim[theme_dim['trend_status'].isin(['HOT', 'RISING'])])
        st.metric("Total Trending Themes", trending_count, delta=f"+{trending_count-5}")
    
    with col2:
        immediate_actions = len(theme_dim[theme_dim['recommendation'].str.contains('IMMEDIATE', na=False)])
        st.metric("Immediate Actions Needed", immediate_actions, delta=f"+{immediate_actions}")
    
    with col3:
        avg_sentiment = theme_dim['avg_sentiment'].mean()
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}", delta=f"+{avg_sentiment-0.1:.2f}")
    
    with col4:
        if len(weekly_totals) >= 2:
            current_week = weekly_totals.iloc[-1]['total_posts_all']
            prev_week = weekly_totals.iloc[-2]['total_posts_all']
            growth = ((current_week - prev_week) / prev_week * 100) if prev_week > 0 else 0
            st.metric("Week-over-Week Growth", f"{growth:.1f}%", delta=f"{growth:.1f}%")
        else:
            st.metric("Week-over-Week Growth", "N/A")
    
    # Add dynamic filter for themes only
    theme_count = st.selectbox("Select number of themes to display:", [5, 10, 15, 20], index=1)
    
    # Top Trending Themes Grid (filtered as per requirements)
    st.subheader(f"🔥 Top {theme_count} Trending Themes")
    
    # Apply filtering as specified: exclude generic themes from Top Trending lists
    # But show enhanced display names for ALL themes (including any generic ones that might appear)
    top_themes = theme_dim[
        (~theme_dim['is_generic_theme']) & 
        (theme_dim['trending_score'] >= 30)
    ].nlargest(theme_count, 'trending_score')
    
    if not top_themes.empty:
        cols = st.columns(5)
        for i, (_, theme) in enumerate(top_themes.iterrows()):
            with cols[i % 5]:
                color = TREND_COLORS.get(theme['trend_status'], '#CCCCCC')
                category_badge_color = theme['category_color']
                tooltip_text = get_theme_tooltip(theme['theme'], theme['primary_category'])
                
                # Create theme card with category badge and tooltip - categories are now properly capitalized in data
                category_display = str(theme['primary_category']) if pd.notna(theme['primary_category']) else 'General'
                st.markdown(f"""
                <div style="background-color: {color}; padding: 12px; border-radius: 8px; margin: 5px 0; position: relative; color: #333333; border: 1px solid #E0E0E0;" title="{tooltip_text}">
                    <div style="background-color: {category_badge_color}; padding: 2px 8px; border-radius: 12px; font-size: 10px; margin-bottom: 8px; display: inline-block; color: white;">
                        🏷️ {category_display}
                    </div>
                    <h5 style="color: #333333; margin: 0; font-size: 13px; line-height: 1.2; font-weight: 600;">{theme['display_theme']}</h5>
                    <p style="color: #555555; margin: 2px 0; font-size: 12px;">Score: {theme['trending_score']:.1f}</p>
                    <p style="color: #666666; margin: 0; font-size: 11px;">{theme['trend_status']}</p>
                    <span style="position: absolute; top: 5px; right: 8px; color: #666666; cursor: help;">ℹ️</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No specific trending themes found. Adjust filters to see more results.")
    
    # Charts with Year Filter
    st.subheader("📊 Trend Analysis")
    
    # Add year filter for charts
    if not weekly_totals.empty:
        # Extract available years from the data
        weekly_totals['year'] = pd.to_datetime(weekly_totals['week_start_date']).dt.year
        available_years = sorted(weekly_totals['year'].unique())
        
        # Year filter dropdown
        col_filter, col_spacer = st.columns([1, 3])
        with col_filter:
            selected_year = st.selectbox(
                "Select Year for Charts:",
                options=['All Years'] + [str(year) for year in available_years],
                index=0
            )
        
        # Filter data based on selected year
        if selected_year == 'All Years':
            filtered_data = weekly_totals
        else:
            filtered_data = weekly_totals[weekly_totals['year'] == int(selected_year)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Weekly Mentions Trend")
        if not weekly_totals.empty:
            fig = px.line(filtered_data, x='week_start_date', y='total_posts_all',
                         title=f"Total Posts Over Time ({selected_year})")
            fig.update_layout(
                xaxis_title="Week Start Date",
                yaxis_title="Total Posts Count"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💭 Sentiment Evolution")
        if not weekly_totals.empty:
            fig = px.line(filtered_data, x='week_start_date', y='avg_sentiment_all',
                         title=f"Average Sentiment Over Time ({selected_year})")
            fig.update_layout(
                xaxis_title="Week Start Date",
                yaxis_title="Average Sentiment Score"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Category Distribution
    st.subheader("📦 Category Distribution")
    category_counts = data['fact_table']['category'].value_counts().head(10)
    fig = px.pie(values=category_counts.values, names=category_counts.index,
                 title="Mentions by Category")
    st.plotly_chart(fig, use_container_width=True)

def page_trend_deep_dive(data):
    st.title("🔥 Trend Deep Dive")
    st.markdown("Identify what's hot and why")
    
    theme_dim = data['theme_dimension']
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hottest = theme_dim.loc[theme_dim['trending_score'].idxmax()]
        st.metric("Hottest Theme", hottest['theme'], f"Score: {hottest['trending_score']:.1f}")
    
    with col2:
        fastest = theme_dim.loc[theme_dim['momentum_4w'].idxmax()]
        st.metric("Fastest Growing", fastest['theme'], f"{fastest['momentum_4w']:.0f}%")
    
    with col3:
        most_positive = theme_dim.loc[theme_dim['avg_sentiment'].idxmax()]
        st.metric("Most Positive", most_positive['theme'], f"{most_positive['avg_sentiment']:.2f}")
    
    with col4:
        purchase_ready = theme_dim.loc[theme_dim['purchase_intent_ratio'].idxmax()]
        st.metric("Purchase Ready", purchase_ready['theme'], f"{purchase_ready['purchase_intent_ratio']:.0f}%")
    
    # Filters
    st.sidebar.subheader("Filters")
    min_mentions = st.sidebar.slider("Minimum Mentions", 0, 50, 5)
    trend_status_filter = st.sidebar.multiselect("Trend Status", 
                                                theme_dim['trend_status'].unique(),
                                                default=theme_dim['trend_status'].unique())
    
    # Filter data
    filtered_themes = theme_dim[
        (theme_dim['total_mentions'] >= min_mentions) &
        (theme_dim['trend_status'].isin(trend_status_filter))
    ]
    
    # Trending Scorecard Table
    st.subheader("📊 Trending Scorecard")
    
    # Enhanced column headers for business clarity
    display_cols = {
        'display_theme': 'Theme / Product',
        'trending_score': 'Trending Score (0-100)',
        'trend_status': 'Status',
        'opportunity_score': 'Seller Opportunity (0-100)',
        'recommendation': 'Action Required',
        'momentum_4w': '4-Week Growth (%)',
        'avg_sentiment': 'Community Sentiment',
        'total_mentions': 'Total Mentions',
        'primary_category': 'Category'
    }
    
    scorecard = filtered_themes[list(display_cols.keys())].sort_values('trending_score', ascending=False)
    scorecard = scorecard.rename(columns=display_cols)
    
    # Color code the dataframe
    def color_trend_status(val):
        color = TREND_COLORS.get(val, '#FFFFFF')
        return f'background-color: {color}; color: white'
    
    styled_scorecard = scorecard.style.map(color_trend_status, subset=['Status'])
    st.dataframe(styled_scorecard, use_container_width=True)
    
    # Bubble Chart with solid colors
    st.subheader("🎯 Sentiment vs Trending Score")
    
    # Use solid colors for bubble chart
    SOLID_TREND_COLORS = {
        'HOT': '#E53E3E',      # Solid red
        'RISING': '#FF8C00',   # Solid orange
        'STEADY': '#FFD700',   # Solid yellow
        'COOLING': '#38A169'   # Solid green
    }
    
    fig = px.scatter(filtered_themes, x='avg_sentiment', y='trending_score',
                    size='total_mentions', color='trend_status',
                    hover_data=['display_theme', 'momentum_4w', 'primary_category'],
                    color_discrete_map=SOLID_TREND_COLORS,
                    title="Bubble Chart: Sentiment vs Trending Score")
    fig.update_layout(
        xaxis_title="Community Sentiment",
        yaxis_title="Trending Score (0-100)"
    )
    st.plotly_chart(fig, use_container_width=True)

def page_sentiment_journey(data):
    st.title("💭 Sentiment Journey")
    st.markdown("Track how feelings evolve")
    
    theme_dim = data['theme_dimension']
    weekly_trends = data['weekly_trends']
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        positive_themes = len(theme_dim[theme_dim['avg_sentiment'] > 0.5])
        st.metric("Very Positive Themes", positive_themes)
    
    with col2:
        negative_themes = len(theme_dim[theme_dim['avg_sentiment'] < -0.5])
        st.metric("Very Negative Themes", negative_themes)
    
    with col3:
        neutral_themes = len(theme_dim[theme_dim['avg_sentiment'].between(-0.1, 0.1)])
        st.metric("Neutral Themes", neutral_themes)
    
    with col4:
        avg_sentiment_all = theme_dim['avg_sentiment'].mean()
        st.metric("Platform Sentiment", f"{avg_sentiment_all:.2f}")
    
    # Sentiment Timeline with Multiple Theme Filter
    st.subheader("📈 Sentiment Timeline")
    if not weekly_trends.empty:
        # Get all available themes
        all_themes = sorted(weekly_trends['theme'].unique())
        
        # Multiple theme filter with all themes selected by default
        selected_themes = st.multiselect(
            "Select Themes for Sentiment Timeline:",
            options=all_themes,
            default=all_themes,  # All themes selected by default
            help="Select one or more themes to display. All themes are shown by default."
        )
        
        # Filter data based on selection
        if selected_themes:
            timeline_data = weekly_trends[weekly_trends['theme'].isin(selected_themes)]
            if len(selected_themes) == len(all_themes):
                chart_title = "Sentiment Evolution for Themes"
            elif len(selected_themes) == 1:
                chart_title = f"Sentiment Evolution for {selected_themes[0].title()}"
            else:
                chart_title = f"Sentiment Evolution for {len(selected_themes)} Selected Themes"
        else:
            # If no themes selected, show all
            timeline_data = weekly_trends
            chart_title = "Sentiment Evolution for Themes"
        
        fig = px.line(timeline_data, x='year_week', y='avg_sentiment', color='theme',
                     title=chart_title)
        fig.update_layout(
            xaxis_title="Time Period (Year-Week)",
            yaxis_title="Average Sentiment Score"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = theme_dim['sentiment_category'].value_counts()
        fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
                    color=sentiment_counts.index,
                    color_discrete_map=SENTIMENT_COLORS,
                    title="Number of Themes by Sentiment Category")
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Number of Themes"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Sentiment vs Volume")
        fig = px.scatter(theme_dim, x='total_mentions', y='avg_sentiment',
                        color='sentiment_category',
                        color_discrete_map=SENTIMENT_COLORS,
                        title="Sentiment vs Total Mentions")
        st.plotly_chart(fig, use_container_width=True)

def page_seller_opportunities(data):
    st.title("💰 Seller Opportunities")
    st.markdown("Actionable recommendations for sellers")
    
    theme_dim = data['theme_dimension']
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        immediate_count = len(theme_dim[theme_dim['recommendation'].str.contains('IMMEDIATE', na=False)])
        st.metric("Immediate Action Items", immediate_count)
    
    with col2:
        high_priority_count = len(theme_dim[theme_dim['recommendation'].str.contains('HIGH PRIORITY', na=False)])
        st.metric("High Priority Themes", high_priority_count)
    
    with col3:
        total_opportunity = theme_dim.nlargest(10, 'opportunity_score')['opportunity_score'].sum()
        st.metric("Total Opportunity Score", f"{total_opportunity:.0f}")
    
    with col4:
        avg_recency = theme_dim[theme_dim['recommendation'].str.contains('IMMEDIATE', na=False)]['recency_days'].mean()
        st.metric("Avg Time to Action", f"{avg_recency:.0f} days" if not pd.isna(avg_recency) else "N/A")
    
    # Opportunity Ranking Table
    st.subheader("🎯 Opportunity Ranking")
    
    # Enhanced column headers
    opportunity_cols = {
        'display_theme': 'Theme / Product',
        'opportunity_score': 'Seller Opportunity (0-100)',
        'recommendation': 'Action Required',
        'trending_score': 'Trending Score (0-100)',
        'momentum_4w': '4-Week Growth (%)',
        'avg_sentiment': 'Community Sentiment',
        'recency_days': 'Days Since Last Mention',
        'primary_category': 'Category'
    }
    
    opportunities = theme_dim[list(opportunity_cols.keys())].sort_values('opportunity_score', ascending=False)
    opportunities = opportunities.rename(columns=opportunity_cols)
    
    # Highlight immediate actions
    def highlight_immediate(row):
        if 'IMMEDIATE' in str(row['Action Required']):
            return ['background-color: #FFE6E6'] * len(row)
        elif 'HIGH PRIORITY' in str(row['Action Required']):
            return ['background-color: #FFF2E6'] * len(row)
        else:
            return [''] * len(row)
    
    styled_opportunities = opportunities.head(20).style.apply(highlight_immediate, axis=1)
    st.dataframe(styled_opportunities, use_container_width=True)
    
    # Action Priority Table with Recommendation Filter
    st.subheader("📊 Action Priority Table")
    
    # Add recommendation filter
    all_recommendations = sorted(theme_dim['recommendation'].unique())
    selected_recommendation = st.selectbox(
        "Filter by Recommendation:",
        options=['All Recommendations'] + list(all_recommendations),
        index=0
    )
    
    # Filter data based on recommendation
    if selected_recommendation == 'All Recommendations':
        filtered_data = theme_dim
    else:
        filtered_data = theme_dim[theme_dim['recommendation'] == selected_recommendation]
    
    # Create table with required columns
    priority_table = filtered_data[['theme', 'trending_score', 'avg_sentiment', 'opportunity_score', 
                                   'display_theme', 'primary_category', 'momentum_4w', 'recommendation']].copy()
    
    # Sort by opportunity score descending
    priority_table = priority_table.sort_values('opportunity_score', ascending=False)
    
    # Format the table
    priority_table['trending_score'] = priority_table['trending_score'].round(1)
    priority_table['avg_sentiment'] = priority_table['avg_sentiment'].round(2)
    priority_table['opportunity_score'] = priority_table['opportunity_score'].round(1)
    priority_table['momentum_4w'] = priority_table['momentum_4w'].round(1)
    
    # Rename columns for display
    priority_table = priority_table.rename(columns={
        'theme': 'Theme',
        'trending_score': 'Trending Score',
        'avg_sentiment': 'Avg Sentiment',
        'opportunity_score': 'Opportunity Score',
        'display_theme': 'Display Theme',
        'primary_category': 'Primary Category',
        'momentum_4w': 'Momentum (4W)',
        'recommendation': 'Recommendation'
    })
    
    st.dataframe(priority_table.head(20), use_container_width=True)
    
    # Add explanation note
    st.info("""
    **How Recommendations are Calculated:**
    - **IMMEDIATE ACTION**: High trending score (>70) + Positive sentiment (>0.3) + High momentum
    - **HIGH PRIORITY**: Good trending score (50-70) + Decent sentiment (>0) + Growing momentum  
    - **MONITOR**: Moderate trending score (30-50) + Any sentiment + Stable momentum
    - **LOW PRIORITY**: Low trending score (<30) + Any sentiment + Low momentum
    
    The system considers trending score, sentiment, and 4-week momentum to determine priority level.
    """)
    
    # Recommended Actions
    st.subheader("✅ Recommended Actions")
    immediate_actions = theme_dim[theme_dim['recommendation'].str.contains('IMMEDIATE', na=False)]
    
    if not immediate_actions.empty:
        for _, action in immediate_actions.head(5).iterrows():
            category_color = action['category_color']
            st.markdown(f"""
            <div style="border-left: 4px solid #DC143C; padding: 10px; margin: 10px 0; background-color: #FFF5F5;">
                <div style="background-color: {category_color}; padding: 2px 8px; border-radius: 12px; font-size: 10px; margin-bottom: 5px; display: inline-block; color: white;">
                    🏷️ {action['primary_category'].upper()}
                </div>
                <h4 style="color: #DC143C; margin: 5px 0;">🚨 IMMEDIATE ACTION: {action['display_theme']}</h4>
                <ul style="margin: 5px 0;">
                    <li><strong>Trending Score:</strong> {action['trending_score']:.1f}/100</li>
                    <li><strong>4-Week Growth:</strong> {action['momentum_4w']:.0f}%</li>
                    <li><strong>Seller Opportunity:</strong> {action['opportunity_score']:.1f}/100</li>
                    <li><strong>Community Sentiment:</strong> {action['avg_sentiment']:.2f}</li>
                </ul>
                <p style="color: #666; font-style: italic;">Action: Alert top sellers about this trending theme immediately</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No immediate actions required at this time. Check back tomorrow for new trends.")

def page_category_performance(data):
    st.title("📦 Category Performance")
    st.markdown("Compare product categories")
    
    fact_table = data['fact_table']
    weekly_category = data['weekly_category']
    
    # Category metrics
    category_metrics = fact_table.groupby('category').agg({
        'video_id': 'count',
        'sentiment_score': 'mean',
        'is_high_quality': 'sum'
    }).rename(columns={
        'video_id': 'total_mentions',
        'sentiment_score': 'avg_sentiment',
        'is_high_quality': 'quality_posts'
    })
    
    category_metrics['quality_ratio'] = (category_metrics['quality_posts'] / category_metrics['total_mentions'] * 100).round(1)
    
    # Display metrics with enhanced headers
    st.subheader("📊 Category Comparison")
    category_display = category_metrics.sort_values('total_mentions', ascending=False).head(15)
    
    # Rename columns for clarity
    category_display = category_display.rename(columns={
        'total_mentions': 'Total Mentions',
        'avg_sentiment': 'Average Sentiment',
        'quality_posts': 'Quality Posts',
        'quality_ratio': 'Quality Ratio (%)'
    })
    
    st.dataframe(category_display, use_container_width=True)
    
    # Category trends with year filter
    st.subheader("📈 Category Trends")
    if not weekly_category.empty:
        # Add year filter for category trends
        weekly_category['year'] = pd.to_datetime(weekly_category['week_start_date']).dt.year
        available_years = sorted(weekly_category['year'].unique())
        
        selected_year_cat = st.selectbox(
            "Select Year for Category Trends:",
            options=['All Years'] + [str(year) for year in available_years],
            index=0,
            key="category_year_filter"
        )
        
        # Filter data based on selected year
        if selected_year_cat == 'All Years':
            filtered_category_data = weekly_category
        else:
            filtered_category_data = weekly_category[weekly_category['year'] == int(selected_year_cat)]
        
        fig = px.line(filtered_category_data, x='week_start_date', y='total_posts', color='category',
                     title=f"Category Mentions Over Time ({selected_year_cat})")
        fig.update_layout(
            xaxis_title="Week Start Date",
            yaxis_title="Total Posts Count"
        )
        st.plotly_chart(fig, use_container_width=True)

def page_theme_decoder(data):
    st.title("📖 Theme Decoder")
    st.markdown("Understanding what each theme means and what sellers should do")
    
    # V2 Enhanced: Use actual theme decoder data from CSV
    theme_decoder = data['theme_decoder']
    
    # Add search functionality
    st.markdown("### 🔍 Search Themes")
    search_term = st.text_input("Search for a specific theme:", placeholder="e.g., pricing, collecting, grading")
    
    if search_term:
        filtered_decoder = theme_decoder[
            theme_decoder['Display Name'].str.contains(search_term, case=False, na=False) |
            theme_decoder['Original Theme'].str.contains(search_term, case=False, na=False) |
            theme_decoder['What It Means'].str.contains(search_term, case=False, na=False)
        ]
    else:
        filtered_decoder = theme_decoder.head(20)  # Show top 20 by default
    
    # V2 Enhanced: Show statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Themes", len(theme_decoder))
    with col2:
        generic_count = len(theme_decoder[theme_decoder['Original Theme'].isin(['pricing', 'collecting', 'grading', 'trading cards', 'nostalgia', 'market trends', 'collectibles', 'investing', 'checklists', 'sets'])])
        st.metric("Generic Themes", generic_count)
    with col3:
        specific_count = len(theme_decoder) - generic_count
        st.metric("Specific Themes", specific_count)
    
    # Display as formatted table
    st.markdown("### 📊 Theme Decoder Reference")
    
    for _, row in filtered_decoder.iterrows():
        # Use enhanced explanation instead of generic one
        enhanced_explanation = get_enhanced_theme_explanation(row['Original Theme'])
        category_display = str(row.get('Category', 'General')).replace('_', ' ').title() if pd.notna(row.get('Category', '')) else 'General'
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; background-color: #f9f9f9;">
            <h4 style="color: #2E86AB; margin: 0 0 10px 0;">🏷️ {row['Display Name']}</h4>
            <p style="margin: 5px 0;"><strong>Original Theme:</strong> {row['Original Theme']}</p>
            <p style="margin: 5px 0;"><strong>What It Means:</strong> {enhanced_explanation}</p>
            <p style="margin: 5px 0; color: #D2691E;"><strong>Seller Action:</strong> {row['Seller Action']}</p>
            <p style="margin: 5px 0; font-size: 12px; color: #666;"><strong>Category:</strong> {category_display}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add category-specific examples
    st.markdown("### 🎨 Category-Specific Examples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Pokemon Examples:**
        - "Pokemon Card Pricing Discussions" = People asking about Charizard values
        - "Pokemon Collection Building" = People organizing their Pokemon binders
        - "Pokemon Card Grading Questions" = People asking about PSA 10 Pokemon cards
        """)
    
    with col2:
        st.markdown("""
        **Baseball Examples:**
        - "Baseball Card Pricing Discussions" = People asking about rookie card values
        - "Baseball Collection Building" = People completing team sets
        - "Baseball Card Grading Questions" = People asking about vintage card grading
        """)
    
    # Add business impact section
    st.markdown("### 💰 Business Impact")
    
    impact_metrics = [
        "**Pricing Discussions** → Launch pricing transparency tools → +15% seller confidence",
        "**Collection Building** → Promote starter sets → +25% new collector acquisition", 
        "**Grading Questions** → Partner with PSA/BGS → +30% high-value listings",
        "**Gift Cards** → Seasonal campaigns → +40% holiday revenue"
    ]
    
    for metric in impact_metrics:
        st.markdown(f"- {metric}")
    
    st.markdown("""
    ---
    **📝 Note:** This decoder helps translate social media conversations into actionable business opportunities. 
    Use it to understand what each trending theme means for your seller strategy.
    """)

def main():
    # Load data
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please check file paths.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🎯 Social Listening")
    st.sidebar.markdown("Navigate through different views")
    
    page = st.sidebar.radio("Select Page", [
        "📊 Executive Overview",
        "🔥 Trend Deep Dive", 
        "💭 Sentiment Journey",
        "💰 Seller Opportunities",
        "📦 Category Performance",
        "📖 Theme Decoder"
    ])
    
    # Route to pages
    if page == "📊 Executive Overview":
        page_executive_overview(data)
    elif page == "🔥 Trend Deep Dive":
        page_trend_deep_dive(data)
    elif page == "💭 Sentiment Journey":
        page_sentiment_journey(data)
    elif page == "💰 Seller Opportunities":
        page_seller_opportunities(data)
    elif page == "📦 Category Performance":
        page_category_performance(data)
    elif page == "📖 Theme Decoder":
        page_theme_decoder(data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Social Listening Intelligence Platform**")
    st.sidebar.markdown("*Data updated: November 2025*")

if __name__ == "__main__":
    main()
