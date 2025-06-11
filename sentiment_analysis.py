# NASA Climate Change Sentiment Analysis
# Advanced sentiment analysis specifically for climate change discussions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ClimateSentimentAnalyzer:
    """Custom sentiment analyzer for climate change discussions"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.climate_positive_words = [
            'solution', 'hope', 'progress', 'innovation', 'renewable', 'clean',
            'sustainable', 'future', 'action', 'positive', 'improvement', 'success',
            'breakthrough', 'opportunity', 'optimistic', 'beneficial', 'effective'
        ]
        self.climate_negative_words = [
            'crisis', 'disaster', 'catastrophe', 'danger', 'threat', 'destruction',
            'devastating', 'alarming', 'urgent', 'serious', 'concern', 'problem',
            'failure', 'damage', 'risk', 'emergency', 'critical'
        ]
        self.skeptical_words = [
            'fake', 'hoax', 'lie', 'scam', 'fraud', 'conspiracy', 'myth',
            'exaggerated', 'wrong', 'false', 'nonsense', 'ridiculous'
        ]
        
    def get_vader_sentiment(self, text):
        """Get VADER sentiment scores"""
        if pd.isna(text) or text == '':
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
        
        scores = self.vader.polarity_scores(str(text))
        return scores
    
    def get_textblob_sentiment(self, text):
        """Get TextBlob sentiment scores"""
        if pd.isna(text) or text == '':
            return {'polarity': 0, 'subjectivity': 0}
        
        blob = TextBlob(str(text))
        return {'polarity': blob.sentiment.polarity, 
                'subjectivity': blob.sentiment.subjectivity}
    
    def get_climate_specific_sentiment(self, text):
        """Custom climate-aware sentiment analysis"""
        if pd.isna(text) or text == '':
            return {'climate_sentiment': 0, 'skeptical_score': 0, 'urgency_score': 0}
        
        text_lower = str(text).lower()
        
        # Count positive climate words
        positive_count = sum(1 for word in self.climate_positive_words if word in text_lower)
        
        # Count negative climate words
        negative_count = sum(1 for word in self.climate_negative_words if word in text_lower)
        
        # Count skeptical words
        skeptical_count = sum(1 for word in self.skeptical_words if word in text_lower)
        
        # Calculate scores
        climate_sentiment = (positive_count - negative_count) / max(len(text_lower.split()), 1)
        skeptical_score = skeptical_count / max(len(text_lower.split()), 1)
        urgency_score = negative_count / max(len(text_lower.split()), 1)
        
        return {
            'climate_sentiment': climate_sentiment,
            'skeptical_score': skeptical_score,
            'urgency_score': urgency_score
        }
    
    def analyze_emotion_patterns(self, text):
        """Detect emotional patterns in climate discussions"""
        if pd.isna(text) or text == '':
            return {'fear': 0, 'anger': 0, 'hope': 0, 'concern': 0}
        
        text_lower = str(text).lower()
        
        # Emotion word dictionaries
        fear_words = ['afraid', 'scared', 'fear', 'terrified', 'worried', 'anxious']
        anger_words = ['angry', 'mad', 'furious', 'outraged', 'frustrated', 'disgusted']
        hope_words = ['hope', 'optimistic', 'positive', 'confident', 'believe', 'faith']
        concern_words = ['concerned', 'worried', 'troubled', 'serious', 'important']
        
        emotions = {
            'fear': sum(1 for word in fear_words if word in text_lower),
            'anger': sum(1 for word in anger_words if word in text_lower),
            'hope': sum(1 for word in hope_words if word in text_lower),
            'concern': sum(1 for word in concern_words if word in text_lower)
        }
        
        return emotions

def analyze_sentiment_comprehensive(df):
    """Comprehensive sentiment analysis of the dataset"""
    print("HOUR 2: COMPREHENSIVE SENTIMENT ANALYSIS")
    print("=" * 50)
    
    analyzer = ClimateSentimentAnalyzer()
    
    print("Analyzing sentiment for", len(df), "comments...")
    
    # 1. VADER Sentiment Analysis
    print("\nRunning VADER Sentiment Analysis...")
    vader_results = df['text'].apply(analyzer.get_vader_sentiment)
    
    df['vader_compound'] = [result['compound'] for result in vader_results]
    df['vader_positive'] = [result['pos'] for result in vader_results]
    df['vader_neutral'] = [result['neu'] for result in vader_results]
    df['vader_negative'] = [result['neg'] for result in vader_results]
    
    # 2. TextBlob Sentiment Analysis
    print("Running TextBlob Sentiment Analysis...")
    textblob_results = df['text'].apply(analyzer.get_textblob_sentiment)
    
    df['textblob_polarity'] = [result['polarity'] for result in textblob_results]
    df['textblob_subjectivity'] = [result['subjectivity'] for result in textblob_results]
    
    # 3. Climate-Specific Sentiment
    print("Running Climate-Specific Sentiment Analysis...")
    climate_results = df['text'].apply(analyzer.get_climate_specific_sentiment)
    
    df['climate_sentiment'] = [result['climate_sentiment'] for result in climate_results]
    df['skeptical_score'] = [result['skeptical_score'] for result in climate_results]
    df['urgency_score'] = [result['urgency_score'] for result in climate_results]
    
    # 4. Emotion Analysis
    print("Running Emotion Pattern Analysis...")
    emotion_results = df['text'].apply(analyzer.analyze_emotion_patterns)
    
    df['emotion_fear'] = [result['fear'] for result in emotion_results]
    df['emotion_anger'] = [result['anger'] for result in emotion_results]
    df['emotion_hope'] = [result['hope'] for result in emotion_results]
    df['emotion_concern'] = [result['concern'] for result in emotion_results]
    
    return df

def create_sentiment_categories(df):
    """Create categorical sentiment labels"""
    print("\nCreating Sentiment Categories...")
    
    # Primary sentiment categories based on VADER compound score
    def categorize_sentiment(score):
        if score >= 0.3:
            return 'Positive'
        elif score <= -0.3:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment_category'] = df['vader_compound'].apply(categorize_sentiment)
    
    # Climate stance based on multiple indicators
    def determine_climate_stance(row):
        if row['skeptical_score'] > 0.01:  # High skeptical words
            return 'Skeptical'
        elif row['climate_sentiment'] > 0.01:  # Positive climate sentiment
            return 'Supportive'
        elif row['urgency_score'] > 0.01:  # High urgency words
            return 'Concerned'
        else:
            return 'Neutral'
    
    df['climate_stance'] = df.apply(determine_climate_stance, axis=1)
    
    # Engagement-weighted sentiment (considers likes/comments)
    df['weighted_sentiment'] = df['vader_compound'] * (1 + np.log1p(df['engagement_score']))
    
    return df

def analyze_sentiment_patterns(df):
    """Analyze patterns in sentiment data"""
    print("\nSENTIMENT PATTERN ANALYSIS")
    print("=" * 40)
    
    # Overall sentiment distribution
    sentiment_dist = df['sentiment_category'].value_counts()
    print("Overall Sentiment Distribution:")
    for category, count in sentiment_dist.items():
        percentage = count / len(df) * 100
        print(f"  {category}: {count} comments ({percentage:.1f}%)")
    
    # Climate stance distribution
    stance_dist = df['climate_stance'].value_counts()
    print("\nClimate Stance Distribution:")
    for stance, count in stance_dist.items():
        percentage = count / len(df) * 100
        print(f"  {stance}: {count} comments ({percentage:.1f}%)")
    
    # Sentiment by engagement level
    print("\nSentiment by Engagement Level:")
    engagement_sentiment = df.groupby('engagement_level')['vader_compound'].mean()
    for level, avg_sentiment in engagement_sentiment.items():
        print(f"  {level} engagement: {avg_sentiment:.3f} average sentiment")
    
    # Temporal sentiment trends
    print("\nSentiment Trends by Year:")
    yearly_sentiment = df.groupby('year')['vader_compound'].mean()
    for year, avg_sentiment in yearly_sentiment.items():
        print(f"  {year}: {avg_sentiment:.3f} average sentiment")
    
    # Most positive and negative comments
    most_positive = df.loc[df['vader_compound'].idxmax()]
    most_negative = df.loc[df['vader_compound'].idxmin()]
    
    print(f"\nMost Positive Comment ({most_positive['vader_compound']:.3f}):")
    print(f"  Text: {most_positive['text'][:100]}...")
    print(f"  Engagement: {most_positive['engagement_score']} points")
    
    print(f"\nMost Negative Comment ({most_negative['vader_compound']:.3f}):")
    print(f"  Text: {most_negative['text'][:100]}...")
    print(f"  Engagement: {most_negative['engagement_score']} points")

def create_sentiment_insights(df):
    """Generate key insights from sentiment analysis"""
    print("\nKEY SENTIMENT INSIGHTS")
    print("=" * 30)
    
    # Correlation between sentiment and engagement
    sentiment_engagement_corr = df['vader_compound'].corr(df['engagement_score'])
    print(f"Sentiment-Engagement Correlation: {sentiment_engagement_corr:.3f}")
    
    if abs(sentiment_engagement_corr) > 0.1:
        if sentiment_engagement_corr > 0:
            print("  → More positive comments tend to get higher engagement")
        else:
            print("  → More negative comments tend to get higher engagement")
    else:
        print("  → Sentiment and engagement are weakly correlated")
    
    # Skeptical vs supportive engagement
    stance_engagement = df.groupby('climate_stance')['engagement_score'].mean()
    print("\nAverage Engagement by Climate Stance:")
    for stance, avg_engagement in stance_engagement.items():
        print(f"  {stance}: {avg_engagement:.2f} average engagement")
    
    # Emotional patterns
    emotion_cols = ['emotion_fear', 'emotion_anger', 'emotion_hope', 'emotion_concern']
    emotion_totals = df[emotion_cols].sum()
    print("\nEmotion Pattern Distribution:")
    for emotion, total in emotion_totals.items():
        emotion_name = emotion.replace('emotion_', '').title()
        print(f"  {emotion_name}: {total} mentions")
    
    # Time-based sentiment patterns
    monthly_sentiment = df.groupby('month_name')['vader_compound'].mean().sort_values(ascending=False)
    print(f"\nMost Positive Month: {monthly_sentiment.index[0]} ({monthly_sentiment.iloc[0]:.3f})")
    print(f"Most Negative Month: {monthly_sentiment.index[-1]} ({monthly_sentiment.iloc[-1]:.3f})")

def save_sentiment_results(df):
    """Save the sentiment analysis results"""
    print("\nSaving Sentiment Analysis Results...")
    
    # Save full dataset with sentiment features
    output_file = 'nasa_climate_with_sentiment.csv'
    df.to_csv(output_file, index=False)
    
    # Create a summary report
    summary_data = {
        'total_comments': len(df),
        'avg_sentiment': df['vader_compound'].mean(),
        'positive_comments': (df['sentiment_category'] == 'Positive').sum(),
        'negative_comments': (df['sentiment_category'] == 'Negative').sum(),
        'neutral_comments': (df['sentiment_category'] == 'Neutral').sum(),
        'skeptical_comments': (df['climate_stance'] == 'Skeptical').sum(),
        'supportive_comments': (df['climate_stance'] == 'Supportive').sum(),
        'most_engaged_sentiment': df.loc[df['engagement_score'].idxmax()]['vader_compound']
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv('sentiment_summary.csv', index=False)
    
    print(f"Full dataset saved as: {output_file}")
    print(f"Summary report saved as: sentiment_summary.csv")
    print(f"   Total features: {len(df.columns)} columns")
    
    return df

def main():
    """Main sentiment analysis pipeline"""
    print("CLIMATE SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Load processed data from Hour 1
    try:
        df = pd.read_csv('nasa_climate_processed.csv')
        print(f"Loaded processed data: {len(df)} comments")
    except FileNotFoundError:
        print("Processed data not found. Please run Hour 1 data processing first.")
        return None
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Run comprehensive sentiment analysis
    df = analyze_sentiment_comprehensive(df)
    
    # Create sentiment categories and labels
    df = create_sentiment_categories(df)
    
    # Analyze patterns and generate insights
    analyze_sentiment_patterns(df)
    create_sentiment_insights(df)
    
    # Save results
    df = save_sentiment_results(df)
    
    print(f"\nSentiment analysis finished.")
    
    return df

# Run the sentiment analysis pipeline
if __name__ == "__main__":
    sentiment_df = main()