# NASA Climate Change Sentiment Analysis Dashboard
# Professional Streamlit application for interactive climate data exploration

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from myth_detector import ClimateMythDetector
from alerts import SmartAlertSystem
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NASA Climate Sentiment Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .insight-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed dataset"""
    try:
        df = pd.read_csv('nasa_climate_with_topics.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please run the data processing pipeline first.")
        return None

def create_sidebar_filters(df):
    """Create sidebar filters for data exploration"""
    st.sidebar.markdown("## Data Filters")
    
    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Sentiment filter
    sentiment_options = ['All'] + list(df['sentiment_category'].unique())
    selected_sentiment = st.sidebar.selectbox("Filter by Sentiment", sentiment_options)
    
    # Topic filter
    topic_options = ['All'] + list(df['topic_label'].unique())
    selected_topic = st.sidebar.selectbox("Filter by Topic", topic_options)
    
    # Engagement filter
    min_engagement = int(df['engagement_score'].min())
    max_engagement = int(df['engagement_score'].max())
    
    engagement_range = st.sidebar.slider(
        "Engagement Score Range",
        min_value=min_engagement,
        max_value=max_engagement,
        value=(min_engagement, max_engagement)
    )
    
    # Climate stance filter
    stance_options = ['All'] + list(df['climate_stance'].unique())
    selected_stance = st.sidebar.selectbox("Filter by Climate Stance", stance_options)
    
    return {
        'date_range': date_range,
        'sentiment': selected_sentiment,
        'topic': selected_topic,
        'engagement_range': engagement_range,
        'stance': selected_stance
    }

def filter_data(df, filters):
    """Apply filters to the dataset"""
    filtered_df = df.copy()
    
    # Date filter
    if len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Sentiment filter
    if filters['sentiment'] != 'All':
        filtered_df = filtered_df[filtered_df['sentiment_category'] == filters['sentiment']]
    
    # Topic filter
    if filters['topic'] != 'All':
        filtered_df = filtered_df[filtered_df['topic_label'] == filters['topic']]
    
    # Engagement filter
    min_eng, max_eng = filters['engagement_range']
    filtered_df = filtered_df[
        (filtered_df['engagement_score'] >= min_eng) & 
        (filtered_df['engagement_score'] <= max_eng)
    ]
    
    # Stance filter
    if filters['stance'] != 'All':
        filtered_df = filtered_df[filtered_df['climate_stance'] == filters['stance']]
    
    return filtered_df

def display_key_metrics(df):
    """Display key metrics in a professional layout"""
    st.markdown('<div class="sub-header">Key Metrics Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Comments",
            value=f"{len(df):,}",
            delta=f"Filtered from {st.session_state.get('total_comments', len(df)):,}"
        )
    
    with col2:
        avg_sentiment = df['vader_compound'].mean()
        st.metric(
            label="Average Sentiment",
            value=f"{avg_sentiment:.3f}",
            delta=f"{'Positive' if avg_sentiment > 0 else 'Negative'} trend"
        )
    
    with col3:
        total_engagement = df['engagement_score'].sum()
        st.metric(
            label="Total Engagement",
            value=f"{total_engagement:,.0f}",
            delta=f"Avg: {df['engagement_score'].mean():.1f}"
        )
    
    with col4:
        date_span = (df['date'].max() - df['date'].min()).days
        st.metric(
            label="Analysis Period",
            value=f"{date_span} days",
            delta=f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
        )

def create_sentiment_timeline(df):
    """Create interactive sentiment timeline"""
    st.markdown('<div class="sub-header">Sentiment Trends Over Time</div>', unsafe_allow_html=True)
    
    # Group by month for cleaner visualization
    monthly_data = df.groupby(df['date'].dt.to_period('M')).agg({
        'vader_compound': 'mean',
        'engagement_score': 'mean',
        'text': 'count'
    }).reset_index()
    
    monthly_data['date_str'] = monthly_data['date'].astype(str)
    monthly_data = monthly_data.rename(columns={'text': 'comment_count'})
    
    # Create dual-axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Sentiment line
    fig.add_trace(
        go.Scatter(
            x=monthly_data['date_str'],
            y=monthly_data['vader_compound'],
            mode='lines+markers',
            name='Average Sentiment',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=8)
        ),
        secondary_y=False
    )
    
    # Comment volume bars
    fig.add_trace(
        go.Bar(
            x=monthly_data['date_str'],
            y=monthly_data['comment_count'],
            name='Comment Volume',
            marker_color='rgba(70, 130, 180, 0.6)',
            yaxis='y2'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Sentiment and Comment Volume Trends",
        height=400,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
    fig.update_yaxes(title_text="Number of Comments", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

def create_topic_analysis(df):
    """Create topic distribution and engagement analysis"""
    st.markdown('<div class="sub-header">Topic Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Topic distribution
        topic_counts = df['topic_label'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=topic_counts.index,
            values=topic_counts.values,
            hole=0.4,
            textinfo='label+percent'
        )])
        
        fig_pie.update_layout(title="Topic Distribution", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Topic engagement
        topic_engagement = df.groupby('topic_label')['engagement_score'].mean().sort_values(ascending=True)
        
        fig_bar = go.Figure(data=[go.Bar(
            x=topic_engagement.values,
            y=topic_engagement.index,
            orientation='h',
            marker_color='#228B22'
        )])
        
        fig_bar.update_layout(
            title="Average Engagement by Topic",
            height=400,
            xaxis_title="Average Engagement Score"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def create_sentiment_analysis(df):
    """Create comprehensive sentiment analysis visualization"""
    st.markdown('<div class="sub-header"> Sentiment Analysis Deep Dive</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment vs engagement scatter
        fig_scatter = px.scatter(
            df.sample(min(500, len(df))),  # Sample for performance
            x='vader_compound',
            y='engagement_score',
            color='topic_label',
            size='text_length',
            hover_data=['sentiment_category', 'climate_stance'],
            title='Sentiment vs Engagement by Topic'
        )
        
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Climate stance analysis
        stance_sentiment = df.groupby(['climate_stance', 'sentiment_category']).size().unstack(fill_value=0)
        
        fig_stack = go.Figure()
        colors = {'Positive': '#2E8B57', 'Neutral': '#4682B4', 'Negative': '#DC143C'}
        
        for sentiment in stance_sentiment.columns:
            fig_stack.add_trace(go.Bar(
                name=sentiment,
                x=stance_sentiment.index,
                y=stance_sentiment[sentiment],
                marker_color=colors.get(sentiment, '#808080')
            ))
        
        fig_stack.update_layout(
            title='Climate Stance vs Sentiment',
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig_stack, use_container_width=True)

def display_sample_comments(df):
    """Display sample comments with analysis"""
    st.markdown('<div class="sub-header">Sample Comments Analysis</div>', unsafe_allow_html=True)
    
    # Top engaged comments
    st.markdown("** Most Engaged Comments:**")
    top_comments = df.nlargest(5, 'engagement_score')[['text', 'engagement_score', 'sentiment_category', 'topic_label']]
    
    for idx, row in top_comments.iterrows():
        with st.expander(f" Engagement Score: {row['engagement_score']:.0f} | {row['sentiment_category']} | {row['topic_label']}"):
            st.write(row['text'][:500] + "..." if len(row['text']) > 500 else row['text'])
    
    st.markdown("---")
    
    # Most positive and negative comments
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**😊 Most Positive Comment:**")
        most_positive = df.loc[df['vader_compound'].idxmax()]
        st.success(f"**Sentiment Score:** {most_positive['vader_compound']:.3f}")
        st.write(most_positive['text'][:300] + "..." if len(most_positive['text']) > 300 else most_positive['text'])
    
    with col2:
        st.markdown("**😞 Most Negative Comment:**")
        most_negative = df.loc[df['vader_compound'].idxmin()]
        st.error(f"**Sentiment Score:** {most_negative['vader_compound']:.3f}")
        st.write(most_negative['text'][:300] + "..." if len(most_negative['text']) > 300 else most_negative['text'])

def generate_insights(df):
    """Generate and display key insights"""
    st.markdown('<div class="sub-header"> Key Insights & Recommendations</div>', unsafe_allow_html=True)
    
    # Calculate insights
    total_comments = len(df)
    avg_sentiment = df['vader_compound'].mean()
    sentiment_dist = df['sentiment_category'].value_counts()
    positive_pct = sentiment_dist.get('Positive', 0) / total_comments * 100
    negative_pct = sentiment_dist.get('Negative', 0) / total_comments * 100
    
    stance_dist = df['climate_stance'].value_counts()
    skeptical_pct = stance_dist.get('Skeptical', 0) / total_comments * 100
    supportive_pct = stance_dist.get('Supportive', 0) / total_comments * 100
    
    most_discussed_topic = df['topic_label'].value_counts().index[0]
    most_engaging_topic = df.groupby('topic_label')['engagement_score'].mean().idxmax()
    
    # Display insights
    insights = []
    
    if avg_sentiment > 0.1:
        insights.append("🎉 **Overall sentiment is POSITIVE** - NASA's climate content resonates well with the audience")
    elif avg_sentiment < -0.1:
        insights.append("⚠️ **Overall sentiment is NEGATIVE** - Consider addressing common concerns in future content")
    else:
        insights.append("😐 **Overall sentiment is NEUTRAL** - Opportunity to create more engaging, positive content")
    
    if positive_pct > negative_pct:
        insights.append(f"✅ **Positive comments ({positive_pct:.1f}%) outnumber negative ones ({negative_pct:.1f}%)**")
    else:
        insights.append(f"⚠️ **Negative comments ({negative_pct:.1f}%) are more prevalent than positive ones ({positive_pct:.1f}%)**")
    
    if supportive_pct > skeptical_pct:
        insights.append(f"🌍 **Climate supporters ({supportive_pct:.1f}%) outnumber skeptics ({skeptical_pct:.1f}%)**")
    else:
        insights.append(f"🚨 **Climate skeptics ({skeptical_pct:.1f}%) are more vocal than supporters ({supportive_pct:.1f}%)**")
    
    insights.append(f" **Most discussed topic:** {most_discussed_topic}")
    insights.append(f" **Most engaging topic:** {most_engaging_topic}")
    
    if most_discussed_topic != most_engaging_topic:
        insights.append(f"**Recommendation:** Focus more content on '{most_engaging_topic}' for higher engagement")
    
    # Display insights in styled boxes
    for insight in insights:
        if "⚠️" in insight or "🚨" in insight:
            st.markdown(f'<div class="warning-box">{insight}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

def display_myth_detection_analysis(df):
    """Display climate myth detection analysis"""
    
    st.markdown("## Climate Myth Detection Analysis")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_myths = df['myths_detected'].sum()
        st.metric("Total Myths Detected", total_myths)
    
    with col2:
        comments_with_myths = df['has_myths'].sum()
        myth_percentage = comments_with_myths / len(df) * 100
        st.metric("Comments with Myths", f"{comments_with_myths} ({myth_percentage:.1f}%)")
    
    with col3:
        high_priority = df['high_priority_myth'].sum()
        st.metric("High Priority Responses", high_priority)
    
    with col4:
        avg_myths_per_comment = df[df['has_myths']]['myths_detected'].mean()
        st.metric("Avg Myths per Flagged Comment", f"{avg_myths_per_comment:.1f}")
    
    # Myth type breakdown
    st.subheader("Myth Types Distribution")

    # Safe myth distribution analysis
    if 'myth_severity' in df.columns and df['has_myths'].sum() > 0:
        # Use myth severity instead of complex details parsing
        myth_comments = df[df['has_myths']]
        severity_counts = myth_comments['myth_severity'].value_counts()
    
        if len(severity_counts) > 0:
            fig_bar = go.Figure(data=[go.Bar(
                x=severity_counts.values,
                y=severity_counts.index,
                orientation='h',
                marker_color='#DC143C'
            )])
        
            fig_bar.update_layout(
                title="Myth Severity Distribution",
                xaxis_title="Number of Comments",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
            # Show breakdown stats
            st.write("**Severity Breakdown:**")
            for severity, count in severity_counts.items():
                st.write(f"• {severity.title()}: {count} comments")
        else:
            st.info("No myth severity data to display.")
    else:
        st.info("No myths detected in current data selection.")
    
    # Sample detected myths
    st.subheader("Sample Detected Myths")
    
    myth_comments = df[df['has_myths']].nlargest(5, 'myths_detected')
    
    for idx, row in myth_comments.iterrows():
        with st.expander(f"Comment with {row['myths_detected']} myth(s) - {row['myth_severity'].title()} Priority"):
            st.write("**Original Comment:**")
            st.write(row['text'][:300] + "..." if len(row['text']) > 300 else row['text'])
            
            st.write("**Detected Myths:**")
            st.warning(f"**Climate Myth Detected** (Severity: {row['myth_severity']})")
            st.info("NASA research confirms climate change is real and human-caused.")
            st.caption("Source: NASA Climate Research")
    
    # Myth detection over time
    if 'date' in df.columns:
        st.subheader("Myth Detection Trends")
        
        monthly_myths = df.groupby(df['date'].dt.to_period('M')).agg({
            'myths_detected': 'sum',
            'has_myths': 'sum'
        }).reset_index()
        
        monthly_myths['date_str'] = monthly_myths['date'].astype(str)
        
        fig_line = go.Figure()
        
        fig_line.add_trace(go.Scatter(
            x=monthly_myths['date_str'],
            y=monthly_myths['myths_detected'],
            mode='lines+markers',
            name='Total Myths Detected',
            line=dict(color='#DC143C', width=3)
        ))
        
        fig_line.update_layout(
            title="Climate Myths Detection Over Time",
            xaxis_title="Month",
            yaxis_title="Number of Myths",
            height=400
        )
        
        st.plotly_chart(fig_line, use_container_width=True)

def show_automated_responses(df):
    """Show how automated responses would work"""
    
    st.markdown("## Automated Response System")
    
    st.info("How it works: When myths are detected, the system can automatically generate educational responses using NASA's authoritative data.")
    
    # Get a comment with myths for demo
    myth_comment = df[df['has_myths']].iloc[0] if len(df[df['has_myths']]) > 0 else None
    
    if myth_comment is not None:
        st.subheader("Response Generation Example")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Comment:**")
            st.text_area("", myth_comment['text'], height=100, disabled=True)
            
            st.write("**Detected Issues:**")
            try:
                if isinstance(myth_comment['myth_details'], list) and len(myth_comment['myth_details']) > 0:
                    for myth in myth_comment['myth_details']:
                        if isinstance(myth, dict):
                            myth_type = myth.get('myth_type', 'Climate Myth')
                            trigger = myth.get('trigger_phrase', 'misconception')
                            st.error(f"{myth_type}: '{trigger}'")
                        else:
                            st.error("Climate misinformation detected")
                else:
                    st.error("Climate misinformation patterns detected")
            except:
                st.error("Climate misinformation detected")
        
        with col2:
            st.write("**Generated Response:**")
            
            # Safe response generation
            sample_response = """Thanks for engaging with NASA's climate content! 

NASA's research is based on decades of data from satellites, ground stations, and ice cores, providing comprehensive evidence for climate change.

For more information, check out NASA's research at: NASA Climate Change Evidence

We appreciate your interest in climate science!"""
            
            st.text_area("", sample_response, height=150, disabled=True)
            
            st.success("Response Quality: Educational, respectful, fact-based")
            st.success("NASA Authority: Backed by official NASA sources")


def display_smart_alerts(df):
    """Display intelligent alert system"""
    
    st.markdown("## Smart Alert System")
    
    # Initialize alert system
    alert_system = SmartAlertSystem(df)
    alerts = alert_system.get_all_alerts()
    
    # Alert summary
    col1, col2, col3, col4 = st.columns(4)
    
    high_alerts = [a for a in alerts if a['severity'] == 'HIGH']
    medium_alerts = [a for a in alerts if a['severity'] == 'MEDIUM']
    positive_alerts = [a for a in alerts if a['severity'] == 'POSITIVE']
    
    with col1:
        st.metric("🔴 High Priority", len(high_alerts))
    with col2:
        st.metric("🟡 Medium Priority", len(medium_alerts))
    with col3:
        st.metric("🟢 Positive Signals", len(positive_alerts))
    with col4:
        st.metric("📊 Total Alerts", len(alerts))
    
    # Display alerts
    if alerts:
        st.subheader("Active Alerts")
        
        for alert in alerts:
            # Color code by severity
            if alert['severity'] == 'HIGH':
                alert_color = "error"
            elif alert['severity'] == 'MEDIUM':
                alert_color = "warning"
            elif alert['severity'] == 'POSITIVE':
                alert_color = "success"
            else:
                alert_color = "info"
            
            with st.container():
                if alert_color == "error":
                    st.error(f"🔴 **{alert['title']}**")
                elif alert_color == "warning":
                    st.warning(f"🟡 **{alert['title']}**")
                elif alert_color == "success":
                    st.success(f"🟢 **{alert['title']}**")
                else:
                    st.info(f"🔵 **{alert['title']}**")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Issue**: {alert['message']}")
                    st.write(f"**Recommended Action**: {alert['action']}")
                
                with col2:
                    if alert['data']:
                        st.write("**Alert Data**:")
                        for key, value in alert['data'].items():
                            if isinstance(value, float):
                                st.caption(f"{key}: {value:.3f}")
                            else:
                                st.caption(f"{key}: {value}")
                
                st.markdown("---")
    else:
        st.success("✅ No alerts detected - all systems operating normally!")
    
    # Alert configuration
    with st.expander("⚙️ Alert Configuration"):
        st.markdown("### Configure Alert Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_threshold = st.slider("Sentiment Drop Alert Threshold", 0.1, 0.5, 0.3, 0.05)
            myth_threshold = st.slider("Myth Alert Threshold", 1, 10, 3, 1)
        
        with col2:
            engagement_threshold = st.slider("Low Engagement Alert (%)", 0.3, 0.8, 0.6, 0.05)
            st.info("💡 Thresholds can be customized based on NASA's communication goals")

def create_alert_summary_widget():
    """Create a small alert summary for main dashboard"""
    
    # This can be added to your main dashboard sidebar
    st.sidebar.markdown("### 🚨 Alert Summary")
    
    # Quick alert counts (you can make this dynamic)
    alert_counts = {
        'High Priority': 2,
        'Medium Priority': 1,
        'Positive Signals': 3
    }
    
    for alert_type, count in alert_counts.items():
        if count > 0:
            if 'High' in alert_type:
                st.sidebar.error(f"🔴 {alert_type}: {count}")
            elif 'Medium' in alert_type:
                st.sidebar.warning(f"🟡 {alert_type}: {count}")
            else:
                st.sidebar.success(f"🟢 {alert_type}: {count}")

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<div class="main-header">NASA Climate Change Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Analyzing public sentiment and engagement with NASA's climate change communications**")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Store total comments for metrics
    if 'total_comments' not in st.session_state:
        st.session_state.total_comments = len(df)
    
    # Sidebar filters
    filters = create_sidebar_filters(df)
    
    # Apply filters
    filtered_df = filter_data(df, filters)
    
    # ALERT SUMMARY TO SIDEBAR 
    #create_alert_summary_widget(filtered_df)

    # Display filter summary
    if len(filtered_df) != len(df):
        st.info(f"Showing {len(filtered_df):,} of {len(df):,} comments based on your filters")
    
    # Main dashboard content
    if len(filtered_df) == 0:
        st.warning("No data matches your current filters. Please adjust your selection.")
        st.stop()
    
    # Key metrics
    display_key_metrics(filtered_df)
    st.markdown("---")
    
    # Sentiment timeline
    create_sentiment_timeline(filtered_df)
    st.markdown("---")
    
    # Topic analysis
    create_topic_analysis(filtered_df)
    st.markdown("---")
    
    # Sentiment analysis
    create_sentiment_analysis(filtered_df)
    st.markdown("---")
    
    # Sample comments
    display_sample_comments(filtered_df)
    st.markdown("---")
    
    # Insights and recommendations
    generate_insights(filtered_df)

    # NEW: Myth Detection Section
    if st.sidebar.checkbox("🕵️ Show Myth Detection"):
        st.markdown("---")
        display_myth_detection_analysis(filtered_df)
    
    # NEW: Smart Alerts Section  
    if st.sidebar.checkbox("🚨 Show Smart Alerts"):
        st.markdown("---")
        display_smart_alerts(filtered_df)
    
    # NEW: Automated Responses Section
    if st.sidebar.checkbox("🤖 Show Automated Responses"):
        st.markdown("---")
        show_automated_responses(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("### Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"nasa_climate_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Generate Report"):
            # Create a summary report
            report_data = {
                'metric': ['Total Comments', 'Average Sentiment', 'Positive %', 'Negative %', 'Most Discussed Topic', 'Most Engaging Topic'],
                'value': [
                    len(filtered_df),
                    f"{filtered_df['vader_compound'].mean():.3f}",
                    f"{(filtered_df['sentiment_category'] == 'Positive').mean() * 100:.1f}%",
                    f"{(filtered_df['sentiment_category'] == 'Negative').mean() * 100:.1f}%",
                    filtered_df['topic_label'].value_counts().index[0],
                    filtered_df.groupby('topic_label')['engagement_score'].mean().idxmax()
                ]
            }
            report_df = pd.DataFrame(report_data)
            csv_report = report_df.to_csv(index=False)
            st.download_button(
                label="Download Report",
                data=csv_report,
                file_name=f"nasa_climate_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        st.markdown("**Dashboard Stats:**")
        st.write(f"• {len(df):,} total comments analyzed")
        st.write(f"• {df['topic_label'].nunique()} topics identified")
        st.write(f"• {df['profileName'].nunique():,} unique users")


if __name__ == "__main__":
    main()