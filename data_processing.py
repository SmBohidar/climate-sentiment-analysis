# NASA Climate Change Comments Data Processor
# Complete data cleaning and feature engineering pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from myth_detector import ClimateMythDetector
from alerts import SmartAlertSystem
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load NASA climate data and show basic info"""
    print("Loading Climate Change Dataset...")
    
    # Load the data
    df = pd.read_csv('data/climate_nasa.csv')
    
    print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Show data types and missing values
    print("\nData Info:")
    print(df.info())
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    for col in missing.index:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} missing ({missing[col]/len(df)*100:.1f}%)")
    
    return df

def clean_data(df):
    """Clean and prepare the data"""
    print("\nCleaning Data...")
    
    # Handle missing values strategically
    df['likesCount'] = df['likesCount'].fillna(0).astype(int)
    df['commentsCount'] = df['commentsCount'].fillna(0).astype(int)
    df['text'] = df['text'].fillna('').astype(str)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove rows with empty text (unusable for analysis)
    initial_count = len(df)
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    removed = initial_count - len(df)
    print(f"Removed {removed} empty comments. Remaining: {len(df)} comments")
    
    return df

def create_engagement_features(df):
    """Create engagement-related features"""
    print("\nCreating Engagement Features...")
    
    # Primary engagement score (weighted: comments worth more than likes)
    df['engagement_score'] = df['likesCount'] + (df['commentsCount'] * 2.5)
    
    # Alternative engagement metrics
    df['total_interactions'] = df['likesCount'] + df['commentsCount']
    df['like_to_comment_ratio'] = np.where(df['commentsCount'] > 0, 
                                          df['likesCount'] / df['commentsCount'], 
                                          df['likesCount'])
    
    # FIXED: Engagement categories with proper binning
    engagement_quartiles = df['engagement_score'].quantile([0.25, 0.5, 0.75])
    
    # Create unique bins to avoid duplicate edges
    q25, q75 = engagement_quartiles[0.25], engagement_quartiles[0.75]
    min_score, max_score = df['engagement_score'].min(), df['engagement_score'].max()
    
    # Ensure bins are unique and meaningful
    if q25 == 0 and q75 == 0:
        # Most comments have 0 engagement - use fixed bins
        bins = [-1, 0, 1, max(2, max_score * 0.5), max_score + 1]
    elif q25 == q75:
        # Same quartile values - create manual bins
        bins = [min_score - 1, 0, max(1, q25), max(2, q25 + 1), max_score + 1]
    else:
        # Normal case with unique quartiles
        bins = [-1, 0, q25, q75, max_score + 1]
    
    # Remove any remaining duplicates and ensure proper ordering
    bins = sorted(list(set(bins)))
    
    # Create engagement levels
    try:
        df['engagement_level'] = pd.cut(df['engagement_score'], 
                                       bins=bins,
                                       labels=['None', 'Low', 'Medium', 'High'][:len(bins)-1],
                                       duplicates='drop')
    except ValueError:
        # Fallback to simple categorization
        print("  Using fallback engagement categorization...")
        df['engagement_level'] = 'Low'
        df.loc[df['engagement_score'] == 0, 'engagement_level'] = 'None'
        df.loc[df['engagement_score'] > df['engagement_score'].median(), 'engagement_level'] = 'Medium'
        df.loc[df['engagement_score'] > df['engagement_score'].quantile(0.8), 'engagement_level'] = 'High'
    
    print(f"Average engagement score: {df['engagement_score'].mean():.2f}")
    print(f"Max engagement score: {df['engagement_score'].max()}")
    
    return df

def create_temporal_features(df):
    """Extract time-based features"""
    print("\nCreating Temporal Features...")
    
    # Basic time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Seasonal features
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df['season'] = df['month'].map(season_map)
    
    # Time periods
    df['time_period'] = pd.cut(df['hour'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              include_lowest=True)
    
    # Show temporal distribution
    print("Comments by year:")
    print(df['year'].value_counts().sort_index())
    
    return df

def create_text_features(df):
    """Create text-based features"""
    print("\nCreating Text Features...")
    
    # Basic text metrics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['sentence_count'] = df['text'].str.split('.').str.len()
    df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
    
    # Text complexity indicators
    df['has_questions'] = df['text'].str.contains(r'\?', regex=True)
    df['has_exclamations'] = df['text'].str.contains(r'!', regex=True)
    df['has_caps'] = df['text'].str.contains(r'[A-Z]{3,}', regex=True)
    df['has_urls'] = df['text'].str.contains(r'http|www\.', regex=True, case=False)
    
    print(f"Average text length: {df['text_length'].mean():.0f} characters")
    print(f"Average word count: {df['word_count'].mean():.1f} words")
    
    return df

def create_climate_keyword_features(df):
    """Create climate change specific keyword features"""
    print("\nCreating Climate Keyword Features...")
    
    # Define keyword categories
    keyword_categories = {
        'climate_science': ['climate', 'warming', 'temperature', 'greenhouse', 'atmosphere', 'carbon cycle'],
        'emissions': ['co2', 'carbon dioxide', 'emissions', 'fossil fuels', 'methane', 'pollution'],
        'skeptical': ['fake', 'hoax', 'lie', 'scam', 'fraud', 'conspiracy', 'myth', 'exaggerated'],
        'supportive': ['real', 'true', 'evidence', 'proof', 'scientific', 'data', 'research'],
        'impact': ['flood', 'drought', 'hurricane', 'wildfire', 'sea level', 'ice', 'polar', 'extreme weather'],
        'action': ['renewable', 'solar', 'wind', 'policy', 'government', 'action', 'solution', 'reduce'],
        'emotional': ['worried', 'scared', 'hope', 'future', 'children', 'planet', 'crisis', 'urgent']
    }
    
    # Create boolean features for each category
    for category, keywords in keyword_categories.items():
        pattern = '|'.join(keywords)
        df[f'mentions_{category}'] = df['text'].str.lower().str.contains(pattern, regex=True, na=False)
    
    # Overall climate relevance score
    climate_columns = [col for col in df.columns if col.startswith('mentions_')]
    df['climate_relevance_score'] = df[climate_columns].sum(axis=1)
    
    # Show keyword statistics
    print("Keyword category frequencies:")
    for col in climate_columns:
        count = df[col].sum()
        percentage = count / len(df) * 100
        print(f"  {col.replace('mentions_', '')}: {count} comments ({percentage:.1f}%)")
    
    return df

def create_user_features(df):
    """Create user engagement features"""
    print("\nCreating User Features...")
    
    # User activity metrics
    user_stats = df.groupby('profileName').agg({
        'engagement_score': ['count', 'mean', 'sum'],
        'text_length': 'mean',
        'date': ['min', 'max']
    }).round(2)
    
    user_stats.columns = ['comment_count', 'avg_engagement', 'total_engagement', 
                         'avg_text_length', 'first_comment', 'last_comment']
    
    # Merge back to main dataframe
    df = df.merge(user_stats[['comment_count', 'avg_engagement']], 
                  left_on='profileName', right_index=True, how='left')
    
    # FIXED: User engagement level with proper binning
    try:
        df['user_engagement_level'] = pd.cut(df['avg_engagement'], 
                                            bins=[0, 1, 5, 15, float('inf')],
                                            labels=['Low', 'Medium', 'High', 'Very_High'])
    except ValueError:
        # Fallback if binning fails
        df['user_engagement_level'] = 'Medium'
        df.loc[df['avg_engagement'] <= 1, 'user_engagement_level'] = 'Low'
        df.loc[df['avg_engagement'] > 5, 'user_engagement_level'] = 'High'
        df.loc[df['avg_engagement'] > 15, 'user_engagement_level'] = 'Very_High'
    
    print(f"Unique users: {df['profileName'].nunique()}")
    print(f"Average comments per user: {df['comment_count'].mean():.1f}")
    
    return df

def generate_summary_stats(df):
    """Generate comprehensive summary statistics"""
    print("\nDATASET SUMMARY STATISTICS")
    print("=" * 50)
    
    # Basic stats
    print(f"Total Comments: {len(df):,}")
    print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Unique Users: {df['profileName'].nunique():,}")
    print(f"Average Engagement: {df['engagement_score'].mean():.2f}")
    print(f"Most Engaged Comment: {df['engagement_score'].max()} points")
    
    # Text statistics
    print(f"\nTEXT METRICS:")
    print(f"   Average Length: {df['text_length'].mean():.0f} characters")
    print(f"   Average Words: {df['word_count'].mean():.1f} words")
    print(f"   Longest Comment: {df['text_length'].max()} characters")
    
    # Climate relevance
    print(f"\nCLIMATE RELEVANCE:")
    climate_mentions = df['climate_relevance_score'] > 0
    print(f"   Comments mentioning climate topics: {climate_mentions.sum()} ({climate_mentions.mean()*100:.1f}%)")
    
    # Temporal patterns
    print(f"\nTEMPORAL PATTERNS:")
    print(f"   Most active year: {df['year'].value_counts().index[0]}")
    print(f"   Most active month: {df['month_name'].value_counts().index[0]}")
    print(f"   Peak engagement hour: {df.groupby('hour')['engagement_score'].mean().idxmax()}:00")
    
    # Engagement distribution
    print(f"\nENGAGEMENT DISTRIBUTION:")
    engagement_dist = df['engagement_level'].value_counts()
    for level, count in engagement_dist.items():
        print(f"   {level} engagement: {count} comments ({count/len(df)*100:.1f}%)")

def add_myth_detection_features(df):
    """Add climate myth detection to existing dataset"""
    print("\nAdding Climate Myth Detection...")
    
    myth_detector = ClimateMythDetector()
    
    # Apply myth detection to all comments
    myth_results = df['text'].apply(myth_detector.detect_myths)
    
    # Extract results into separate columns
    df['myths_detected'] = myth_results.apply(lambda x: x['myth_count'])
    df['myth_severity'] = myth_results.apply(lambda x: x['highest_severity'])
    df['requires_response'] = myth_results.apply(lambda x: x['requires_response'])
    df['myth_details'] = myth_results.apply(lambda x: x['myths_detected'])
    
    # Create myth categories
    df['has_myths'] = df['myths_detected'] > 0
    df['high_priority_myth'] = df['myth_severity'] == 'high'
    
    # Summary statistics
    total_myths = df['myths_detected'].sum()
    comments_with_myths = df['has_myths'].sum()
    high_priority = df['high_priority_myth'].sum()
    
    print(f"✅ Myth Detection Complete:")
    print(f"   • Total myths detected: {total_myths}")
    print(f"   • Comments containing myths: {comments_with_myths} ({comments_with_myths/len(df)*100:.1f}%)")
    print(f"   • High priority responses needed: {high_priority}")
    
    return df

def main():
    """Main processing pipeline"""
    print(" CLIMATE CHANGE DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and explore
    df = load_and_explore_data()
    
    # Step 2: Clean data
    df = clean_data(df)
    
    # Step 3: Create features
    df = create_engagement_features(df)
    df = create_temporal_features(df)
    df = create_text_features(df)
    df = create_climate_keyword_features(df)
    df = add_myth_detection_features(df)
    df = create_user_features(df)
    
    # Step 4: Generate summary
    generate_summary_stats(df)
    
    # Step 5: Save processed data
    output_file = 'nasa_climate_processed.csv'
    df.to_csv(output_file, index=False)
    print(f"\n Processed dataset saved as: {output_file}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n Ready for sentiment analysis.")
    print("Next Step: sentiment analysis...")
    
    return df

# Run the processing pipeline
if __name__ == "__main__":
    processed_df = main()