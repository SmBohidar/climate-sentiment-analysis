# NASA Climate Change Interactive Visualizations
# Professional interactive charts and graphs for climate sentiment analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for static plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClimateVisualizer:
    """Professional visualization suite for climate sentiment analysis"""
    
    def __init__(self, df):
        self.df = df
        self.color_palette = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4',   # Steel Blue
            'climate': '#228B22',   # Forest Green
            'skeptical': '#B22222', # Fire Brick
            'concerned': '#FF8C00', # Dark Orange
            'supportive': '#32CD32' # Lime Green
        }
    
    def create_sentiment_timeline(self):
        """Create interactive sentiment trends over time"""
        print("Creating Sentiment Timeline...")
        
        # Prepare monthly data
        monthly_data = self.df.groupby([self.df['date'].dt.to_period('M')]).agg({
            'vader_compound': 'mean',
            'engagement_score': 'mean',
            'text': 'count'
        }).reset_index()
        
        monthly_data['date'] = monthly_data['date'].astype(str)
        monthly_data = monthly_data.rename(columns={'text': 'comment_count'})
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Average Sentiment Over Time', 'Comment Volume & Engagement'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # Sentiment line
        fig.add_trace(
            go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['vader_compound'],
                mode='lines+markers',
                name='Average Sentiment',
                line=dict(color=self.color_palette['climate'], width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Comment volume bars
        fig.add_trace(
            go.Bar(
                x=monthly_data['date'],
                y=monthly_data['comment_count'],
                name='Comments',
                marker_color=self.color_palette['neutral'],
                opacity=0.7,
                hovertemplate='<b>%{x}</b><br>Comments: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Engagement line on secondary axis
        fig.add_trace(
            go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['engagement_score'],
                mode='lines+markers',
                name='Avg Engagement',
                line=dict(color=self.color_palette['positive'], width=2),
                marker=dict(size=6),
                yaxis='y3',
                hovertemplate='<b>%{x}</b><br>Engagement: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='NASA Climate Comments: Sentiment & Engagement Trends',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
        fig.update_yaxes(title_text="Number of Comments", row=2, col=1)
        fig.update_yaxes(title_text="Average Engagement", secondary_y=True, row=2, col=1)
        
        return fig
    
    def create_topic_distribution(self):
        """Create interactive topic distribution charts"""
        print("Creating Topic Distribution...")
        
        # Topic counts
        topic_counts = self.df['topic_label'].value_counts()
        
        # Create pie chart and bar chart side by side
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Topic Distribution', 'Topic Engagement'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=topic_counts.index,
                values=topic_counts.values,
                hole=0.4,
                hovertemplate='<b>%{label}</b><br>Comments: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Engagement by topic
        topic_engagement = self.df.groupby('topic_label')['engagement_score'].mean().sort_values(ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=topic_engagement.values,
                y=topic_engagement.index,
                orientation='h',
                marker_color=self.color_palette['climate'],
                hovertemplate='<b>%{y}</b><br>Avg Engagement: %{x:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Climate Discussion Topics Analysis',
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Average Engagement Score", row=1, col=2)
        
        return fig
    
    def create_sentiment_engagement_scatter(self):
        """Create sentiment vs engagement scatter plot"""
        print("Creating Sentiment-Engagement Scatter Plot...")
        
        # Sample data if too large
        plot_df = self.df.sample(min(1000, len(self.df))) if len(self.df) > 1000 else self.df
        
        fig = px.scatter(
            plot_df,
            x='vader_compound',
            y='engagement_score',
            color='topic_label',
            size='text_length',
            hover_data=['sentiment_category', 'climate_stance'],
            title='Sentiment vs Engagement by Topic',
            labels={
                'vader_compound': 'Sentiment Score',
                'engagement_score': 'Engagement Score',
                'topic_label': 'Topic',
                'text_length': 'Comment Length'
            }
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Sentiment Score (-1 = Negative, +1 = Positive)",
            yaxis_title="Engagement Score (Likes + 2.5×Comments)"
        )
        
        return fig
    
    def create_climate_stance_analysis(self):
        """Analyze climate stance distribution and engagement"""
        print("Creating Climate Stance Analysis...")
        
        # Stance distribution with sentiment
        stance_sentiment = self.df.groupby(['climate_stance', 'sentiment_category']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        for sentiment in stance_sentiment.columns:
            fig.add_trace(go.Bar(
                name=sentiment,
                x=stance_sentiment.index,
                y=stance_sentiment[sentiment],
                marker_color=self.color_palette.get(sentiment.lower(), '#808080')
            ))
        
        fig.update_layout(
            title='Climate Stance vs Sentiment Distribution',
            xaxis_title='Climate Stance',
            yaxis_title='Number of Comments',
            barmode='stack',
            height=400
        )
        
        return fig
    
    def create_word_cloud(self):
        """Create word cloud from most frequent terms"""
        print("Creating Word Cloud...")
        
        # Combine all text
        all_text = ' '.join(self.df['text'].fillna('').astype(str))
        
        # Clean text for word cloud
        import re
        clean_text = re.sub(r'http\S+|www\S+|https\S+', '', all_text)
        clean_text = re.sub(r'[^a-zA-Z\s]', ' ', clean_text)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
            stopwords=set(['nasa', 'climate', 'change', 'post', 'comment', 'page'])
        ).generate(clean_text)
        
        # Convert to plotly figure
        fig = go.Figure()
        
        fig.add_layout_image(
            dict(
                source=wordcloud.to_image(),
                xref="x", yref="y",
                x=0, y=1,
                sizex=1, sizey=1,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )
        
        fig.update_layout(
            title="Most Frequent Words in Climate Discussions",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400
        )
        
        return fig
    
    def create_temporal_heatmap(self):
        """Create heatmap of activity patterns"""
        print("Creating Temporal Activity Heatmap...")
        
        # Create hour vs day of week heatmap
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_name'] = self.df['date'].dt.day_name()
        
        heatmap_data = self.df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            hovertemplate='<b>%{y} %{x}:00</b><br>Comments: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Comment Activity Heatmap (Day vs Hour)',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        
        return fig
    
    def create_engagement_distribution(self):
        """Create engagement score distribution"""
        print("Creating Engagement Distribution...")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Engagement Distribution', 'Top Engaged Comments by Topic')
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=self.df['engagement_score'],
                nbinsx=30,
                name='Engagement Distribution',
                marker_color=self.color_palette['climate'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Top comments by topic
        top_comments = self.df.nlargest(20, 'engagement_score')
        topic_engagement = top_comments.groupby('topic_label')['engagement_score'].sum().sort_values(ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=topic_engagement.values,
                y=topic_engagement.index,
                orientation='h',
                name='Topic Engagement',
                marker_color=self.color_palette['positive']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Engagement Analysis Overview',
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Engagement Score", row=1, col=1)
        fig.update_xaxes(title_text="Total Engagement", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        return fig

def create_summary_dashboard(df):
    """Create a comprehensive summary dashboard"""
    print("Creating Summary Dashboard...")
    
    visualizer = ClimateVisualizer(df)
    
    # Calculate key metrics
    total_comments = len(df)
    avg_sentiment = df['vader_compound'].mean()
    total_engagement = df['engagement_score'].sum()
    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    
    # Sentiment distribution
    sentiment_dist = df['sentiment_category'].value_counts()
    positive_pct = sentiment_dist.get('Positive', 0) / total_comments * 100
    negative_pct = sentiment_dist.get('Negative', 0) / total_comments * 100
    
    # Topic insights
    top_topic = df['topic_label'].value_counts().index[0]
    most_engaging_topic = df.groupby('topic_label')['engagement_score'].mean().idxmax()
    
    # Create metrics cards
    metrics = {
        'Total Comments': f"{total_comments:,}",
        'Average Sentiment': f"{avg_sentiment:.3f}",
        'Total Engagement': f"{total_engagement:,.0f}",
        'Date Range': date_range,
        'Positive Comments': f"{positive_pct:.1f}%",
        'Negative Comments': f"{negative_pct:.1f}%",
        'Most Discussed Topic': top_topic,
        'Most Engaging Topic': most_engaging_topic
    }
    
    return metrics

def generate_all_visualizations(df):
    """Generate all visualizations and save them"""
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 50)
    
    visualizer = ClimateVisualizer(df)
    figures = {}
    
    # Generate all charts
    print("Creating visualizations...")
    
    try:
        figures['sentiment_timeline'] = visualizer.create_sentiment_timeline()
        print("Sentiment timeline created")
    except Exception as e:
        print(f"Error creating sentiment timeline: {e}")
    
    try:
        figures['topic_distribution'] = visualizer.create_topic_distribution()
        print("Topic distribution created")
    except Exception as e:
        print(f"Error creating topic distribution: {e}")
    
    try:
        figures['sentiment_scatter'] = visualizer.create_sentiment_engagement_scatter()
        print("Sentiment-engagement scatter created")
    except Exception as e:
        print(f"Error creating sentiment scatter: {e}")
    
    try:
        figures['climate_stance'] = visualizer.create_climate_stance_analysis()
        print("Climate stance analysis created")
    except Exception as e:
        print(f"Error creating climate stance analysis: {e}")
    
    try:
        figures['word_cloud'] = visualizer.create_word_cloud()
        print("Word cloud created")
    except Exception as e:
        print(f"Error creating word cloud: {e}")
    
    try:
        figures['activity_heatmap'] = visualizer.create_temporal_heatmap()
        print("Activity heatmap created")
    except Exception as e:
        print(f"Error creating activity heatmap: {e}")
    
    try:
        figures['engagement_dist'] = visualizer.create_engagement_distribution()
        print("Engagement distribution created")
    except Exception as e:
        print(f"Error creating engagement distribution: {e}")
    
    # Generate summary metrics
    summary_metrics = create_summary_dashboard(df)
    
    return figures, summary_metrics

def save_visualizations(figures, output_dir='climate_visualizations'):
    """Save all visualizations as HTML files"""
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nSaving visualizations to {output_dir}/...")
    
    for name, fig in figures.items():
        if fig is not None:
            filename = f"{output_dir}/{name}.html"
            fig.write_html(filename)
            print(f"Saved {name}.html")
    
    print(f"\nAll visualizations saved in {output_dir}/ directory")

def create_static_summary_plots(df):
    """Create additional static plots for comprehensive analysis"""
    print("\nCreating Additional Static Plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NASA Climate Change Comments - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Sentiment distribution
    sentiment_counts = df['sentiment_category'].value_counts()
    colors = ['#2E8B57', '#4682B4', '#DC143C']  # Green, Blue, Red
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
    axes[0, 0].set_title('Sentiment Distribution')
    
    # 2. Monthly comment volume
    monthly_counts = df.groupby(df['date'].dt.to_period('M')).size()
    axes[0, 1].plot(range(len(monthly_counts)), monthly_counts.values, marker='o', linewidth=2, markersize=6)
    axes[0, 1].set_title('Monthly Comment Volume')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Number of Comments')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Topic engagement comparison
    topic_engagement = df.groupby('topic_label')['engagement_score'].mean().sort_values()
    axes[0, 2].barh(range(len(topic_engagement)), topic_engagement.values, color='#228B22')
    axes[0, 2].set_yticks(range(len(topic_engagement)))
    axes[0, 2].set_yticklabels(topic_engagement.index, fontsize=8)
    axes[0, 2].set_title('Average Engagement by Topic')
    axes[0, 2].set_xlabel('Average Engagement Score')
    
    # 4. Sentiment vs text length
    axes[1, 0].scatter(df['text_length'], df['vader_compound'], alpha=0.6, c=df['engagement_score'], 
                      cmap='viridis', s=30)
    axes[1, 0].set_title('Sentiment vs Comment Length')
    axes[1, 0].set_xlabel('Comment Length (characters)')
    axes[1, 0].set_ylabel('Sentiment Score')
    
    # 5. Climate stance distribution
    stance_counts = df['climate_stance'].value_counts()
    axes[1, 1].bar(stance_counts.index, stance_counts.values, color=['#32CD32', '#FF8C00', '#B22222', '#4682B4'])
    axes[1, 1].set_title('Climate Stance Distribution')
    axes[1, 1].set_ylabel('Number of Comments')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Hourly activity pattern
    hourly_activity = df.groupby(df['date'].dt.hour).size()
    axes[1, 2].plot(hourly_activity.index, hourly_activity.values, marker='o', linewidth=2, color='#DC143C')
    axes[1, 2].set_title('Comments by Hour of Day')
    axes[1, 2].set_xlabel('Hour of Day')
    axes[1, 2].set_ylabel('Number of Comments')
    axes[1, 2].set_xticks(range(0, 24, 4))
    
    plt.tight_layout()
    plt.savefig('climate_analysis_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Static summary plots created and saved as 'climate_analysis_summary.png'")

def create_insights_report(df, summary_metrics):
    """Generate a comprehensive insights report"""
    print("\nGENERATING INSIGHTS REPORT...")
    
    insights = []
    
    # Basic statistics insights
    insights.append("=== NASA CLIMATE CHANGE COMMENTS ANALYSIS REPORT ===\n")
    insights.append(f"Analysis Period: {summary_metrics['Date Range']}")
    insights.append(f"Total Comments Analyzed: {summary_metrics['Total Comments']}")
    insights.append(f"Overall Sentiment: {summary_metrics['Average Sentiment']} (Range: -1 to +1)")
    insights.append(f"Total Engagement: {summary_metrics['Total Engagement']} points\n")
    
    # Sentiment insights
    positive_pct = float(summary_metrics['Positive Comments'].rstrip('%'))
    negative_pct = float(summary_metrics['Negative Comments'].rstrip('%'))
    neutral_pct = 100 - positive_pct - negative_pct
    
    insights.append("=== SENTIMENT ANALYSIS INSIGHTS ===")
    insights.append(f"• {positive_pct:.1f}% of comments express positive sentiment")
    insights.append(f"• {negative_pct:.1f}% of comments express negative sentiment")
    insights.append(f"• {neutral_pct:.1f}% of comments are neutral")
    
    if positive_pct > negative_pct:
        insights.append("• Overall public sentiment leans POSITIVE toward NASA's climate content")
    else:
        insights.append("• Overall public sentiment leans NEGATIVE toward NASA's climate content")
    
    # Topic insights
    insights.append(f"\n=== TOPIC ANALYSIS INSIGHTS ===")
    insights.append(f"• Most discussed topic: {summary_metrics['Most Discussed Topic']}")
    insights.append(f"• Most engaging topic: {summary_metrics['Most Engaging Topic']}")
    
    topic_counts = df['topic_label'].value_counts()
    insights.append(f"• {len(topic_counts)} distinct topics identified in discussions")
    
    # Engagement insights
    high_engagement = (df['engagement_score'] > df['engagement_score'].quantile(0.8)).sum()
    insights.append(f"\n=== ENGAGEMENT INSIGHTS ===")
    insights.append(f"• {high_engagement} comments ({high_engagement/len(df)*100:.1f}%) received high engagement")
    
    avg_likes = df['likesCount'].mean()
    avg_comments = df['commentsCount'].mean()
    insights.append(f"• Average likes per comment: {avg_likes:.1f}")
    insights.append(f"• Average replies per comment: {avg_comments:.1f}")
    
    # Temporal insights
    peak_month = df.groupby(df['date'].dt.month_name())['engagement_score'].sum().idxmax()
    peak_hour = df.groupby(df['date'].dt.hour).size().idxmax()
    
    insights.append(f"\n=== TEMPORAL INSIGHTS ===")
    insights.append(f"• Peak engagement month: {peak_month}")
    insights.append(f"• Most active hour: {peak_hour}:00")
    
    # Climate-specific insights
    skeptical_pct = (df['climate_stance'] == 'Skeptical').sum() / len(df) * 100
    supportive_pct = (df['climate_stance'] == 'Supportive').sum() / len(df) * 100
    
    insights.append(f"\n=== CLIMATE STANCE INSIGHTS ===")
    insights.append(f"• {skeptical_pct:.1f}% of comments express climate skepticism")
    insights.append(f"• {supportive_pct:.1f}% of comments are supportive of climate science")
    
    if supportive_pct > skeptical_pct:
        insights.append("• Climate science supporters outnumber skeptics in discussions")
    else:
        insights.append("• Climate skeptics are more vocal in discussions")
    
    # Recommendations
    insights.append(f"\n=== RECOMMENDATIONS FOR NASA ===")
    
    if negative_pct > 30:
        insights.append("• Consider addressing common concerns in climate skeptic comments")
    
    if summary_metrics['Most Engaging Topic'] != summary_metrics['Most Discussed Topic']:
        insights.append(f"• Focus more content on '{summary_metrics['Most Engaging Topic']}' (high engagement topic)")
    
    insights.append(f"• Optimal posting time: {peak_hour}:00 for maximum visibility")
    insights.append("• Encourage more positive discussions to improve overall sentiment")
    
    # Save report
    report_text = '\n'.join(insights)
    with open('climate_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print("Comprehensive insights report saved as 'climate_analysis_report.txt'")
    print("\nKEY INSIGHTS PREVIEW:")
    for insight in insights[:10]:
        print(insight)

def main():
    """Main visualization generation pipeline"""
    print("CLIMATE VISUALIZATION GENERATION")
    print("=" * 60)
    
    # Load data with topics
    try:
        df = pd.read_csv('nasa_climate_with_topics.csv')
        print(f"Loaded topic data: {len(df)} comments")
    except FileNotFoundError:
        print("Topic data not found. Please run Hour 3 topic analysis first.")
        return None
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Generate all interactive visualizations
    figures, summary_metrics = generate_all_visualizations(df)
    
    # Save interactive visualizations
    save_visualizations(figures)
    
    # Create additional static plots
    create_static_summary_plots(df)
    
    # Generate comprehensive insights report
    create_insights_report(df, summary_metrics)
    
    print(f"\nAll visualizations and insights generated.")
    print(f"\nGenerated Files:")
    print("  • Interactive HTML charts in 'climate_visualizations/' folder")
    print("  • Static summary plot: 'climate_analysis_summary.png'")
    print("  • Insights report: 'climate_analysis_report.txt'")
    
    return figures, summary_metrics

# Run the visualization pipeline
if __name__ == "__main__":
    viz_figures, metrics = main()